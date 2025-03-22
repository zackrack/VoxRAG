import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from faster_whisper import WhisperModel
import tempfile
import soundfile as sf
import os
import io
import base64

# ----- Silero VAD Setup -----
# Load Silero VAD model via torch.hub.
model_vad, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)
(get_speech_ts, _, _, _, _) = utils

# ----- Wav2Vec2 Setup for Embeddings -----
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec_model.to(device)
wav2vec_model.eval()

# ----- Faster-Whisper Setup for Transcription -----
whisper_model = WhisperModel("tiny", device="cuda", compute_type="float16")

def get_audio_embedding(audio_tensor, sample_rate):
    if audio_tensor.numel() == 0:
        return np.zeros((768,), dtype=np.float32)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        audio_tensor = resampler(audio_tensor)
        sample_rate = 16000

    # Convert to mono
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.mean(dim=0)

    # Ensure 2D tensor [1, samples]
    audio_tensor = audio_tensor.unsqueeze(0)

    # 🔁 Normalize to 5s (for consistent embeddings)
    target_samples = 16000 * 5
    if audio_tensor.shape[1] < target_samples:
        pad = target_samples - audio_tensor.shape[1]
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad))
    else:
        audio_tensor = audio_tensor[:, :target_samples]

    inputs = processor(audio_tensor.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        outputs = wav2vec_model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

def assign_speakers_to_vad_segments(
    vad_segments, vad_metadata, diarization_segments, custom_speaker_mapping
):
    """
    Matches VAD-based speech segments with diarization speaker labels
    by overlapping them in time. NO sub-segment is discarded,
    even if it's under 4 seconds.
    """
    matched_segments = []
    matched_metadata = []

    for segment, meta in zip(vad_segments, vad_metadata):
        vad_start = meta["start_time"]
        vad_end = meta["end_time"]
        sr = segment.shape[1] / (vad_end - vad_start)  # sample_rate from segment length

        # For each diarization segment, find overlap
        for dia in diarization_segments:
            dia_start = dia["start"]
            dia_end = dia["end"]

            # Skip if no overlap
            if dia_end <= vad_start or dia_start >= vad_end:
                continue

            # Clip to VAD boundaries
            start_time = max(vad_start, dia_start)
            end_time = min(vad_end, dia_end)

            if end_time <= start_time:
                continue

            # Slice out the overlapping audio
            rel_start = int((start_time - vad_start) * sr)
            rel_end = int((end_time - vad_start) * sr)
            sub_segment = segment[:, rel_start:rel_end]

            new_meta = {
                "start_time": start_time,
                "end_time": end_time,
                "speaker_label": dia["speaker"],
                "speaker": custom_speaker_mapping.get(dia["speaker"], dia["speaker"]),
            }

            matched_segments.append(sub_segment)
            matched_metadata.append(new_meta)

    return matched_segments, matched_metadata


def mmr_rerank(query_embedding, candidate_embeddings, candidate_indices, top_k=10, lambda_param=0.5):
    """
    Maximal Marginal Relevance (MMR) reranking.
    Selects diverse and relevant results from top-k candidates.

    Args:
        query_embedding (torch.Tensor): [1, dim] query embedding
        candidate_embeddings (torch.Tensor): [k, dim] candidate embeddings
        candidate_indices (List[int]): original indices of the candidates
        top_k (int): number of final results to select
        lambda_param (float): trade-off between relevance and diversity (0 = all diverse, 1 = all relevant)

    Returns:
        List[int]: reranked candidate indices
    """
    selected = []
    candidate_pool = list(range(candidate_embeddings.shape[0]))

    # Precompute similarities
    query_sim = torch.matmul(candidate_embeddings, query_embedding.squeeze(0))  # [k]
    candidate_sim = torch.matmul(candidate_embeddings, candidate_embeddings.T)  # [k, k]

    while len(selected) < top_k and candidate_pool:
        mmr_scores = []
        for idx in candidate_pool:
            sim_to_query = query_sim[idx]
            sim_to_selected = 0
            if selected:
                sim_to_selected = torch.max(candidate_sim[idx, selected])
            score = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
            mmr_scores.append(score.item())

        # Pick best-scoring candidate
        best_idx_in_pool = torch.tensor(mmr_scores).argmax().item()
        selected_idx = candidate_pool[best_idx_in_pool]
        selected.append(selected_idx)
        candidate_pool.remove(selected_idx)

    # Map back to original indices
    return [candidate_indices[i] for i in selected]

def segment_audio_silero(audio_tensor, sample_rate, threshold=0.3, min_duration=3.0, target_duration=10.0, max_chunk_size=20.0):
    """
    Smart segmentation with NO AUDIO LOSS:
    - Uses VAD to detect speech.
    - Merges short segments up to target_duration (~10s).
    - Keeps shorter segments (≥ min_duration) when needed.
    - Splits overly long segments (> max_chunk_size).
    """
    if sample_rate != 16000:
        audio_tensor = torchaudio.transforms.Resample(sample_rate, 16000)(audio_tensor)

    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)  # Convert to mono

    print(f"📏 Running VAD... Audio shape: {audio_tensor.shape}, Sample rate: {sample_rate}")
    speech_timestamps = get_speech_ts(audio_tensor, model_vad, sampling_rate=16000, threshold=threshold)
    print(f"🔍 VAD detected {len(speech_timestamps)} raw speech segments.")

    merged_segments, metadata = [], []
    temp_start, temp_end = None, None

    for i, ts in enumerate(speech_timestamps):
        start_time = ts["start"] / sample_rate
        end_time = ts["end"] / sample_rate
        duration = end_time - start_time

        print(f"⏳ Found segment {i+1}: {duration:.2f} sec ({start_time:.2f}s → {end_time:.2f}s)")

        if temp_start is None:
            temp_start, temp_end = start_time, end_time
        else:
            current_duration = temp_end - temp_start
            gap = start_time - temp_end

            if gap <= 1.0 and current_duration < target_duration:
                print(f"🔗 Merging with segment {i+1} (gap: {gap:.2f}s, current total: {current_duration:.2f}s)")
                temp_end = end_time
            else:
                final_duration = temp_end - temp_start
                if final_duration >= min_duration:
                    merged_segments.append((temp_start, temp_end))
                    metadata.append({"start_time": temp_start, "end_time": temp_end})
                    print(f"✅ Finalized merged segment: {final_duration:.2f}s")
                else:
                    print(f"⚠️ Skipped short segment: {final_duration:.2f}s")
                temp_start, temp_end = start_time, end_time

    # Add last segment
    if temp_start is not None and temp_end is not None:
        final_duration = temp_end - temp_start
        if final_duration >= min_duration:
            merged_segments.append((temp_start, temp_end))
            metadata.append({"start_time": temp_start, "end_time": temp_end})
            print(f"✅ Finalized last segment: {final_duration:.2f}s")
        else:
            print(f"⚠️ Skipped last short segment: {final_duration:.2f}s")

    print(f"🔍 Merged into {len(merged_segments)} segments (min {min_duration}s each, target ~{target_duration}s).")

    # Final chunking: split if over max_chunk_size
    final_segments, final_metadata = [], []
    for meta in metadata:
        start_time, end_time = meta["start_time"], meta["end_time"]
        duration = end_time - start_time

        if duration <= max_chunk_size:
            final_segments.append(audio_tensor[:, int(start_time * 16000):int(end_time * 16000)])
            final_metadata.append(meta)
        else:
            # Split long segments
            while start_time < end_time:
                chunk_end = min(start_time + max_chunk_size, end_time)
                start_idx = int(start_time * 16000)
                end_idx = int(chunk_end * 16000)

                final_segments.append(audio_tensor[:, start_idx:end_idx])
                final_metadata.append({"start_time": start_time, "end_time": chunk_end})
                print(f"🔪 Split long segment: {chunk_end - start_time:.2f}s")

                start_time = chunk_end

    print(f"✅ Finalized {len(final_segments)} segments after max-length splitting.")
    return final_segments, final_metadata


def extract_audio_segment(audio_tensor, sample_rate, start_time, end_time):
    """
    Extracts a segment from the given audio tensor using start and end times (in seconds).
    Returns a tuple (audio_data as numpy array, sample_rate).
    """
    start_idx = int(start_time * sample_rate)
    end_idx = int(end_time * sample_rate)
    if audio_tensor.dim() > 1:
        segment = audio_tensor[:, start_idx:end_idx]
        segment = segment.mean(dim=0)  # Convert to mono
    else:
        segment = audio_tensor[start_idx:end_idx]
    return segment.cpu().numpy(), sample_rate

def transcribe_with_fasterwhisper(audio_data, sample_rate, language="en"):
    """
    Transcribes audio using faster-whisper.
    Saves audio to a temporary WAV file (as 16-bit PCM), transcribes it, and returns the transcription text.
    """
    import tempfile, os, numpy as np, soundfile as sf

    # If audio_data is empty, return an empty transcription.
    if audio_data.size == 0:
        return ""
    
    # Squeeze to remove extra dimensions.
    audio_data = np.squeeze(audio_data)
    
    # If the audio is 2D and in shape (channels, samples), transpose to (samples, channels)
    if audio_data.ndim == 2:
        # We assume the number of channels is the first dimension; adjust if necessary.
        audio_data = audio_data.T

    # Ensure the data is float32 in the range [-1, 1]
    if np.issubdtype(audio_data.dtype, np.floating):
        audio_data = np.clip(audio_data, -1, 1).astype(np.float32)
    else:
        audio_data = audio_data.astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_filename = tmp.name
    try:
        sf.write(tmp_filename, audio_data, sample_rate, format="WAV", subtype="PCM_16")
        segments, info = whisper_model.transcribe(tmp_filename, beam_size=5, language=language)
        transcription = " ".join([seg.text for seg in segments])
    finally:
        os.remove(tmp_filename)
    
    return transcription

def audio_to_html(audio_data, sample_rate):
    """
    Converts a numpy audio array into an HTML <audio> element with a base64-encoded WAV.
    Checks for empty segments and ensures data is in float32 within [-1, 1].
    """
    import tempfile, os, base64, numpy as np
    import soundfile as sf

    # Remove extra dimensions.
    audio_data = np.squeeze(audio_data)
    
    # Check if the segment is empty.
    if audio_data.size == 0 or (audio_data.ndim == 1 and audio_data.shape[0] == 0) or \
       (audio_data.ndim == 2 and audio_data.shape[1] == 0):
        return "<div>No audio available</div>"
    
    # If 2D (channels, samples), transpose to (samples, channels).
    if audio_data.ndim == 2:
        audio_data = audio_data.T

    # Ensure data is float32 in range [-1, 1].
    if np.issubdtype(audio_data.dtype, np.floating):
        audio_data = np.clip(audio_data, -1, 1).astype(np.float32)
    else:
        audio_data = audio_data.astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_filename = tmp.name
    try:
        # Write as a 16-bit PCM WAV file.
        sf.write(temp_filename, audio_data, sample_rate, format="WAV", subtype="PCM_16")
        with open(temp_filename, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print("Error writing audio:", e)
        return "<div>Error generating audio</div>"
    finally:
        os.remove(temp_filename)
    
    return f'<audio controls src="data:audio/wav;base64,{b64}"></audio>'

import json
import os

import json
import os

def save_embeddings_to_jsonl(jsonl_path, embeddings_tensor, metadata):
    """
    Save embeddings + metadata to a JSONL file on disk.

    Args:
        jsonl_path (str): where to write the file
        embeddings_tensor (torch.Tensor): shape [N, D]
        metadata (List[dict]): one dict per segment
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
    Load stored embeddings from JSONL and retrieve top-k most similar to the query.
    Args:
        query_embedding (np.array or torch.Tensor): shape [D]
        jsonl_path (str): path to JSONL file
        top_k (int): number of top results to return
    Returns:
        List[dict]: top-k entries with metadata + segment_id
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

    all_embeddings_tensor = torch.cat(embeddings, dim=0)  # [N, D]
    similarities = torch.mm(query_tensor, all_embeddings_tensor.T).squeeze(0)  # [N]
    top_indices = torch.topk(similarities, k=min(top_k, len(entries))).indices.tolist()

    return [entries[i] for i in top_indices]

def diarize_segment(audio_data, sample_rate):
    """
    Uses NVIDIA NeMo's ClusteringDiarizer with system VAD to perform speaker diarization,
    with parameters adjusted for studio-quality audio.
    
    Parameters:
      audio_data (np.array): Audio data (shape: [channels, samples] or [samples])
      sample_rate (int): Sample rate of the audio.
      
    Returns:
      List[Dict]: Each dict has keys 'start', 'end', and 'speaker'.
    """
    import tempfile, os, json, shutil, requests
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ClusteringDiarizer
    import soundfile as sf

    # Ensure audio is mono: if multi-channel, average across channels.
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=0)  # now shape is (samples,)

    # Write the audio data to a temporary WAV file.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        sf.write(tmp_audio.name, audio_data, sample_rate, format="WAV")
        audio_filepath = tmp_audio.name

    # Create a temporary manifest file (JSONL) required by NeMo.
    manifest = {
        "audio_filepath": audio_filepath,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": None
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_manifest:
        tmp_manifest.write(json.dumps(manifest) + "\n")
        manifest_filepath = tmp_manifest.name

    # Create a temporary directory for NeMo outputs.
    out_dir = tempfile.mkdtemp()

    # Download the NeMo diarization config if it doesn't exist locally.
    config_path = "diar_infer_telephonic.yaml"
    config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
    if not os.path.exists(config_path):
        r = requests.get(config_url)
        with open(config_path, "w") as f:
            f.write(r.text)

    # Load and update the NeMo config.
    config = OmegaConf.load(config_path)
    config.diarizer.manifest_filepath = manifest_filepath
    config.diarizer.out_dir = out_dir

    # Use system VAD.
    config.diarizer.oracle_vad = False  
    config.diarizer.vad.model_path = "vad_multilingual_marblenet"
    # Adjust VAD parameters for studio-quality audio:
    config.diarizer.vad.parameters.onset = 0.5
    config.diarizer.vad.parameters.offset = 0.4
    config.diarizer.vad.parameters.pad_offset = -0.02

    # Adjust clustering parameters if available.
    # (Here we set a clustering threshold; note that the exact name may vary by config version.)
    config.diarizer.clustering.parameters.threshold = 0.75
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Disable multiprocessing to avoid pickling issues on Windows.
    config.num_workers = 0

    # Initialize and run the clustering diarizer.
    diarizer = ClusteringDiarizer(cfg=config)
    diarizer.diarize()

    # Locate the RTTM output file.
    pred_rttm_dir = os.path.join(out_dir, "pred_rttms")
    rttm_file = None
    if os.path.exists(pred_rttm_dir):
        for f in os.listdir(pred_rttm_dir):
            if f.endswith(".rttm"):
                rttm_file = os.path.join(pred_rttm_dir, f)
                break
    if rttm_file is None:
        os.remove(audio_filepath)
        os.remove(manifest_filepath)
        shutil.rmtree(out_dir)
        raise RuntimeError("RTTM file not found after diarization.")

    # Parse the RTTM file.
    segments = []
    with open(rttm_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker_id = parts[7]
            segments.append({
                "start": start,
                "end": end,
                "speaker": speaker_id
            })

    # Clean up temporary files.
    os.remove(audio_filepath)
    os.remove(manifest_filepath)
    shutil.rmtree(out_dir)

    return segments
