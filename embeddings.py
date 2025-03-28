# embeddings.py

import os
import json
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoProcessor, AutoModel

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Wav2Vec2 Setup for Audio Embeddings -----
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model.to(device)
wav2vec_model.eval()

# ----- CLAP Setup for Audio Embeddings -----
# Replace "laion/CLAP" with the appropriate model identifier if needed.
clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
clap_model = AutoModel.from_pretrained("laion/clap-htsat-unfused")
clap_model.to(device)
clap_model.eval()

def get_audio_embedding(audio_tensor, sample_rate):
    """
    Returns a normalized embedding for the given audio segment using Wav2Vec2.
    The audio is normalized to 5 seconds (padding or truncating as needed).
    """
    if audio_tensor.numel() == 0:
        return np.zeros((768,), dtype=np.float32)

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_tensor = resampler(audio_tensor)
        sample_rate = 16000

    # Convert to mono if multi-channel
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.mean(dim=0)

    # Ensure the tensor is 2D: [1, samples]
    audio_tensor = audio_tensor.unsqueeze(0)

    # Normalize to 5 seconds (5 * 16000 samples)
    target_samples = 16000 * 5
    if audio_tensor.shape[1] < target_samples:
        pad = target_samples - audio_tensor.shape[1]
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad))
    else:
        audio_tensor = audio_tensor[:, :target_samples]

    # Process the audio and compute the embedding
    inputs = processor(audio_tensor.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = wav2vec_model(**inputs)

    # Take the mean of the last hidden state as the embedding vector
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

def get_clap_embedding(audio_tensor, sample_rate):
    if audio_tensor.numel() == 0:
        return np.zeros((768,), dtype=np.float32)  # fallback for empty audio

    # Resample to 48kHz (CLAP requires this)
    if sample_rate != 48000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)
        audio_tensor = resampler(audio_tensor)

    # Convert to mono if needed
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.mean(dim=0)

    # Pad or truncate to 5s
    target_samples = 48000 * 5
    if audio_tensor.shape[0] < target_samples:
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, target_samples - audio_tensor.shape[0]))
    else:
        audio_tensor = audio_tensor[:target_samples]

    audio_list = audio_tensor.cpu().numpy()  # match what processor expects

    # Process only audio
    inputs = clap_processor(audios=audio_list, sampling_rate=48000, return_tensors="pt")

    # Move to correct device
    inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

    with torch.no_grad():
        outputs = clap_model.get_audio_features(**inputs)

    embedding = outputs[0].cpu().numpy()
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def save_embeddings_to_jsonl(jsonl_path, embeddings_tensor, metadata):
    """
    Save embeddings along with their metadata to a JSONL file.
    Each line in the file represents a segment.
    """
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w") as f:
        for i, (embedding, meta) in enumerate(zip(embeddings_tensor, metadata)):
            entry = {
                "segment_id": i,
                "embedding": embedding.cpu().tolist(),
                "metadata": meta
            }
            f.write(json.dumps(entry) + "\n")
    print(f"✅ Saved {len(metadata)} segments to {jsonl_path}")

def query_from_jsonl(query_embedding, jsonl_path, top_k=10):
    """
    Load stored embeddings from a JSONL file and return the top-k entries
    most similar to the query embedding.
    """
    if isinstance(query_embedding, np.ndarray):
        query_tensor = torch.tensor(query_embedding).float()
    else:
        query_tensor = query_embedding.float()
    query_tensor = torch.nn.functional.normalize(query_tensor, dim=0).unsqueeze(0)

    entries = []
    embeddings = []

    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            emb = torch.tensor(entry["embedding"]).float()
            emb = torch.nn.functional.normalize(emb, dim=0).unsqueeze(0)
            embeddings.append(emb)
            entries.append(entry)

    if not embeddings:
        return []

    all_embeddings_tensor = torch.cat(embeddings, dim=0)
    similarities = torch.mm(query_tensor, all_embeddings_tensor.T).squeeze(0)
    top_indices = torch.topk(similarities, k=min(top_k, len(entries))).indices.tolist()

    return [entries[i] for i in top_indices]
