import gradio as gr
import torch, torchaudio, numpy as np, time, json
from contextlib import contextmanager
from openai import OpenAI
from helpers import (
    get_audio_embedding,
    segment_audio_silero,
    transcribe_with_fasterwhisper,
    audio_to_html,
    save_embeddings_to_jsonl,
    diarize_segment,  # NEW: Import the diarization function
    assign_speakers_to_vad_segments,
)
import tempfile
import soundfile as sf
import os

# Initialize OpenAI and device
openai = OpenAI(api_key="")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global state for a single podcast file
single_file_state = {
    "embeddings_tensor": None,
    "metadata": None,
    "audio_tensor": None,
    "sample_rate": None
}

# Global speaker mapping: default labels to your custom names.
# Note: extracted labels might include "male", "music", "noEnergy", etc.
custom_speaker_mapping = {
    "speaker_0": "Garnt",
    "speaker_1": "Joey",
    "speaker_2": "Connor"
}

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name} took {end - start:.3f} seconds")

def update_all_metadata_with_mapping():
    if single_file_state["metadata"] is None:
        return
    for meta in single_file_state["metadata"]:
        original = meta.get("speaker_label", meta.get("speaker"))
        meta["speaker"] = custom_speaker_mapping.get(original, original)

def load_saved_embeddings():
    if not os.path.exists("data/segments.jsonl"):
        return "❌ No saved embeddings found."

    try:
        entries = []
        embeddings = []
        with open("data/segments.jsonl", "r") as f:
            for line in f:
                entry = json.loads(line)
                emb = torch.tensor(entry["embedding"]).float()
                embeddings.append(emb)
                entries.append(entry)

        embeddings_tensor = torch.stack(embeddings).float().to(device)
        metadata = [entry["metadata"] for entry in entries]

        # Try to load corresponding audio segments
        audio_segments = torch.load("data/audio_segments.pt") if os.path.exists("data/audio_segments.pt") else [None] * len(metadata)

        single_file_state["embeddings_tensor"] = embeddings_tensor
        single_file_state["metadata"] = metadata
        single_file_state["audio_tensor"] = audio_segments
        single_file_state["sample_rate"] = 16000  # or also save/load this

        return f"✅ Loaded {len(metadata)} segments and audio from saved embeddings."
    except Exception as e:
        return f"❌ Error loading embeddings: {str(e)}"


def get_extracted_speaker_mapping():
    """
    Returns a string that shows the mapping of all unique extracted speaker labels
    to their current custom names.
    """
    if single_file_state["metadata"] is None:
        return "No speakers extracted yet."
    mapping = {}
    for meta in single_file_state["metadata"]:
        label = meta.get("speaker_label", "unknown")
        if label not in mapping:
            mapping[label] = custom_speaker_mapping.get(label, label)
    mapping_str = "Extracted Speaker Mapping:\n"
    for key, val in mapping.items():
        mapping_str += f"{key}: {val}\n"
    return mapping_str

def index_podcast(podcast_file):
    if podcast_file is None:
        return "Please upload a podcast audio file."

    print("🔍 Loading podcast...")
    audio_tensor, sample_rate = torchaudio.load(podcast_file)

    # For testing: Trim the podcast to the first 20 minutes (1200 seconds)
    # max_duration = 5 * 60  # 20 minutes in seconds
    # max_samples = max_duration * sample_rate
    # if audio_tensor.shape[1] > max_samples:
    #     print("✂️  Trimming audio to first 20 minutes for testing.")
    #     audio_tensor = audio_tensor[:, :max_samples]

    # Resample if necessary
    if sample_rate != 16000:
        print(f"🔄 Resampling from {sample_rate} to 16kHz")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_tensor = resampler(audio_tensor)
        sample_rate = 16000

    print(f"📏 Audio tensor shape: {audio_tensor.shape}, Sample rate: {sample_rate}")

    # ----- NEW: Diarize the FULL audio BEFORE segmentation -----
    hf_token = "YOUR_HF_TOKEN"  # Replace with your actual Hugging Face token
    print("🔍 Running diarization on full audio...")
    audio_np = audio_tensor.cpu().numpy()
    diarization_results = diarize_segment(audio_np, sample_rate)
    print(f"🔍 Diarization found {len(diarization_results)} speaker turns.")

    # Extract segments based on diarization results
    speaker_segments = []
    speaker_metadata = []
    # Step 1: Get Silero segments
    vad_segments, vad_metadata = segment_audio_silero(audio_tensor, sample_rate)

    # Step 2: Assign speakers using NeMo diarization results
    matched_segments, matched_metadata = assign_speakers_to_vad_segments(
        vad_segments, vad_metadata, diarization_results, custom_speaker_mapping
    )

    # Optionally merge consecutive sub-segments with same speaker
    merged_segments, merged_metadata = merge_consecutive_speaker_segments(
        matched_segments, matched_metadata,
        gap_threshold=1.0  # or however big a gap you allow
    )

    speaker_segments = merged_segments
    speaker_metadata = merged_metadata

    print(f"✅ Indexed {len(speaker_segments)} speaker segments from diarization.")

    # Compute embeddings for each speaker segment
    embeddings = []
    stored_segments = []  # Store processed segments
    for i, seg in enumerate(speaker_segments):
        if seg.numel() == 0:
            print(f"⚠️ Skipping empty segment {i+1}")
            continue

        print(f"🎙 Processing speaker segment {i+1} of {len(speaker_segments)}")
        emb = get_audio_embedding(seg, sample_rate)
        if np.any(np.isnan(emb)):
            print(f"⚠️ NaN detected in embedding for segment {i+1}, skipping.")
            continue

        embeddings.append(emb)
        stored_segments.append(seg)

    if not embeddings:
        return "❌ No valid embeddings computed. Possible empty or corrupted audio."

    try:
        embeddings_tensor = torch.from_numpy(np.stack(embeddings)).float().to(device)
    except ValueError as e:
        print(f"❌ Error stacking embeddings: {e}")
        return "❌ Failed to process embeddings."

    # Store speaker-segment info
    single_file_state["embeddings_tensor"] = embeddings_tensor
    single_file_state["metadata"] = speaker_metadata
    single_file_state["audio_tensor"] = stored_segments
    single_file_state["sample_rate"] = sample_rate

    # Save JSONL to disk
    save_embeddings_to_jsonl("data/segments.jsonl", embeddings_tensor, speaker_metadata)
    # Save audio segments to a binary file
    torch.save(stored_segments, "data/audio_segments.pt")
    save_full_state("data/full_state.pt")

    return f"✅ Podcast indexed into {len(stored_segments)} speaker segments."

def save_full_state(path="data/full_state.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        state = {
            "embeddings_tensor": single_file_state["embeddings_tensor"].cpu(),
            "metadata": single_file_state["metadata"],
            "audio_tensor": [seg.cpu() for seg in single_file_state["audio_tensor"]],
            "sample_rate": single_file_state["sample_rate"],
            "speaker_mapping": custom_speaker_mapping,
        }
        torch.save(state, path)
        return "✅ Full state saved successfully!"
    except Exception as e:
        return f"❌ Error saving state: {str(e)}"


def load_full_state(path="data/full_state.pt"):
    if not os.path.exists(path):
        return "❌ No saved state found."

    try:
        state = torch.load(path, map_location=device)
        single_file_state["embeddings_tensor"] = state["embeddings_tensor"].to(device)
        single_file_state["metadata"] = state["metadata"]
        single_file_state["audio_tensor"] = [seg.to(device) for seg in state["audio_tensor"]]
        single_file_state["sample_rate"] = state.get("sample_rate", 16000)

        global custom_speaker_mapping
        custom_speaker_mapping = state.get("speaker_mapping", custom_speaker_mapping)

        # 🔥 Apply mapping to all metadata after loading
        update_all_metadata_with_mapping()

        # ✅ Now that it's updated, save the corrected JSONL
        if single_file_state["embeddings_tensor"] is not None and single_file_state["metadata"] is not None:
            save_embeddings_to_jsonl("data/segments.jsonl", single_file_state["embeddings_tensor"], single_file_state["metadata"])
            print("📝 Resaved segments.jsonl with updated speaker labels.")

        return f"✅ Loaded full state with {len(single_file_state['metadata'])} segments."
    except Exception as e:
        return f"❌ Error loading state: {str(e)}"


def query_rag(query_audio_file, lambda_param=0.5, length_alpha=0.1):
    """
    1) Loads and embeds the query audio.
    2) Reads all stored segments from data/segments.jsonl, normalizes their embeddings.
    3) Computes similarity to the query, applies a length-based boost, and picks top-k.
    4) Applies MMR re-ranking on those top-k to get a diverse & relevant final set.
    5) Transcribes and returns the top segments + an LLM answer.

    Args:
        query_audio_file (str): Path to the user's query audio file.
        lambda_param (float): MMR trade-off (0 = more diverse, 1 = more relevant).
        length_alpha (float): Strength of the length boost. If 0.0, no boost.
    """
    import math
    import json
    import torch
    import numpy as np
    from torch.nn import functional as F

    if query_audio_file is None:
        return "Provide a query audio clip.", "", []

    # Make sure we have segments to search
    if not os.path.exists("data/segments.jsonl"):
        return "Index a podcast first.", "", []

    # 1) Load & resample query audio
    query_tensor, query_sr = torchaudio.load(query_audio_file)
    if query_sr != 16000:
        resampler = torchaudio.transforms.Resample(query_sr, 16000)
        query_tensor = resampler(query_tensor)
        query_sr = 16000

    # Convert to mono
    if query_tensor.dim() > 1:
        query_tensor = query_tensor.mean(dim=0)

    # Transcribe the user's query so we know what they said
    query_audio_np = query_tensor.cpu().numpy()
    query_transcription = transcribe_with_fasterwhisper(query_audio_np, query_sr)

    # Clip query to 10s max (optional – you can remove if you want)
    query_tensor = query_tensor[: query_sr * 10]
    if query_tensor.shape[0] < query_sr * 4:
        print("⚠️ Query too short (<4s). Retrieval quality may suffer.")

    # 2) Compute the query embedding
    query_embedding = get_audio_embedding(query_tensor, query_sr)  # shape [dim]
    query_tensor = torch.tensor(query_embedding).float()
    query_tensor = F.normalize(query_tensor, dim=0).unsqueeze(0)  # shape [1, dim]

    # 3) Load & normalize all indexed embeddings
    entries, embeddings = [], []
    with open("data/segments.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            emb = torch.tensor(entry["embedding"]).float()
            emb = F.normalize(emb, dim=0).unsqueeze(0)  # shape [1, dim]
            embeddings.append(emb)
            entries.append(entry)

    all_embeddings_tensor = torch.cat(embeddings, dim=0)  # shape [N, dim]

    # Compute raw similarity: query ⋅ candidate
    similarities = torch.mm(query_tensor, all_embeddings_tensor.T).squeeze(0)  # shape [N]

    # === LENGTH BOOST ===
    # We'll multiply each similarity by (1 + length_alpha * log1p(duration))
    # so that longer segments get a slight advantage.
    sim_np = similarities.detach().cpu().numpy()  # to NumPy
    boosted_scores = sim_np.copy()

    for i, entry in enumerate(entries):
        meta = entry["metadata"]
        duration = meta["end_time"] - meta["start_time"]  # how long this segment is
        length_boost = 1.0 + length_alpha * math.log1p(duration)  # log(1+duration)
        boosted_scores[i] *= length_boost

    # Convert back to Torch
    boosted_scores_torch = torch.from_numpy(boosted_scores).to(similarities.device)

    # We'll pick a pool of top-50 from boosted scores to feed into MMR
    top_k_pool = 50
    top_indices = torch.topk(boosted_scores_torch, k=min(top_k_pool, len(entries))).indices.tolist()

    # 4) MMR re-ranking (using the boosted similarity for "query relevance")
    def mmr_rerank(query_scores, candidate_embeddings, candidate_indices, top_k=10, lambda_param=0.5):
        """
        MMR that uses 'query_scores' for relevance and cos-sim among candidates for diversity.
        query_scores: 1D array or tensor with a relevance score per candidate
        """
        selected = []
        candidate_pool = candidate_indices.copy()

        while len(selected) < top_k and candidate_pool:
            mmr_vals = []
            for idx in candidate_pool:
                sim_q = query_scores[idx]  # boosted similarity
                # find max sim to any already selected candidate
                if selected:
                    sim_sel = max(
                        F.cosine_similarity(
                            candidate_embeddings[idx].unsqueeze(0),
                            candidate_embeddings[s].unsqueeze(0),
                            dim=1
                        ).item()
                        for s in selected
                    )
                else:
                    sim_sel = 0.0

                # MMR formula
                mmr_score = lambda_param * sim_q - (1 - lambda_param) * sim_sel
                mmr_vals.append((idx, mmr_score))

            # pick best
            mmr_vals.sort(key=lambda x: x[1], reverse=True)
            best_idx = mmr_vals[0][0]
            selected.append(best_idx)
            candidate_pool.remove(best_idx)

        return selected

    reranked_indices = mmr_rerank(
        boosted_scores_torch,  # query_scores
        all_embeddings_tensor, # candidate embeddings
        top_indices,
        top_k=10,  # final results
        lambda_param=lambda_param
    )

    top_results = [entries[i] for i in reranked_indices]

    # 5) Transcribe top segments and build final answer
    retrieved_transcriptions = []
    retrieved_audio_html = []
    retrieved_docs = []

    for result in top_results:
        meta = result["metadata"]
        idx = result["segment_id"]

        snippet = single_file_state["audio_tensor"][idx]
        if snippet is None or snippet.numel() == 0:
            print(f"❌ Empty audio segment at index {idx}. Skipping.")
            continue

        snippet_np = snippet.cpu().numpy()
        if snippet_np.ndim == 1:
            snippet_np = np.expand_dims(snippet_np, axis=0)
        if snippet_np.dtype != np.float32:
            snippet_np = snippet_np.astype(np.float32)
        if np.any(np.isnan(snippet_np)):
            print(f"❌ NaN audio at index {idx}. Skipping.")
            continue

        seg_transcription = transcribe_with_fasterwhisper(snippet_np, single_file_state["sample_rate"])
        speaker = meta.get("speaker", "unknown")

        retrieved_transcriptions.append(f"{speaker} said: {seg_transcription}")
        retrieved_audio_html.append(
            f"<div><b>{speaker}:</b> {audio_to_html(snippet_np, single_file_state['sample_rate'])}</div><br>"
        )
        retrieved_docs.append((speaker, seg_transcription))

    print(f"🗣️ Query transcription: {query_transcription}")

    # Format prompt for OpenAI
    prompt = (
        f"User query transcription: {query_transcription}\n\n"
        f"Retrieved podcast segments transcriptions:\n"
    )
    for i, trans in enumerate(retrieved_transcriptions):
        prompt += f"Segment {i+1}: {trans}\n"
    prompt += "\nBased on the above, generate a concise and accurate answer to the user's query."

    print(prompt)

    # Call OpenAI
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions about Garnt, Joey, and Connor "
                    "based on the speech segments provided. Only use the segments' content to answer."
                )
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=300,
    )
    answer = response.choices[0].message.content

    # Return final answer + retrieved audio in HTML + an optional doc list
    return answer, retrieved_audio_html, [
        (entry["metadata"].get("speaker", "unknown"), trans)
        for entry, trans in zip(top_results, [t.split(": ", 1)[-1] for t in retrieved_transcriptions])
    ]



def apply_speaker_mapping(mapping_json):
    global custom_speaker_mapping

    try:
        mapping = json.loads(mapping_json)
        if not isinstance(mapping, dict):
            return "❌ Mapping must be a JSON object."

        custom_speaker_mapping = mapping
        update_all_metadata_with_mapping()

        # 🔥 After updating metadata, re-save the updated mapping into the .jsonl
        if single_file_state["embeddings_tensor"] is not None and single_file_state["metadata"] is not None:
            save_embeddings_to_jsonl("data/segments.jsonl", single_file_state["embeddings_tensor"], single_file_state["metadata"])
            print("📝 Resaved segments.jsonl with updated speaker labels.")

        return "✅ Speaker mapping updated successfully!"
    except Exception as e:
        return f"❌ Error parsing JSON: {str(e)}"

def merge_consecutive_speaker_segments(segments, metadata, gap_threshold=1.0):
    """
    Merges consecutive segments if:
      - They have the same speaker
      - The gap between them <= gap_threshold
    Returns (merged_segments, merged_metadata)
    """
    # Sort everything by start_time just in case
    # (If they're already in chronological order, you can skip sorting.)
    combined = sorted(zip(segments, metadata), key=lambda x: x[1]["start_time"])

    merged_segments = []
    merged_metadata = []

    current_seg = None
    current_meta = None

    for seg, meta in combined:
        if current_seg is None:
            # This is the first segment
            current_seg = seg
            current_meta = dict(meta)  # copy
        else:
            # Check speaker match + time gap
            same_speaker = (meta["speaker"] == current_meta["speaker"])
            gap = meta["start_time"] - current_meta["end_time"]

            if same_speaker and gap <= gap_threshold:
                # Merge the audio by concatenation along time axis (dim=1)
                current_seg = torch.cat([current_seg, seg], dim=1)
                # Update the end time
                current_meta["end_time"] = meta["end_time"]
            else:
                # Different speaker or bigger gap -> finalize the old segment
                merged_segments.append(current_seg)
                merged_metadata.append(current_meta)

                # Start a new current segment
                current_seg = seg
                current_meta = dict(meta)

    # Append the last leftover
    if current_seg is not None:
        merged_segments.append(current_seg)
        merged_metadata.append(current_meta)

    return merged_segments, merged_metadata

def merge_segments_for_speaker(speaker_segments, gap_threshold=1.0):
    merged = []
    current_meta, current_seg = None, None
    for meta, seg in speaker_segments:  # KEEP ORIGINAL ORDER, DO NOT SORT
        if current_meta is None:
            current_meta = meta.copy()
            current_seg = seg
        else:
            gap = meta["start_time"] - current_meta["end_time"]
            # Only merge if the segments are **back-to-back in diarization output**
            # and close enough in time
            if gap <= gap_threshold:
                current_meta["end_time"] = meta["end_time"]
                current_seg = torch.cat([current_seg, seg], dim=1)
            else:
                merged.append((current_meta, current_seg))
                current_meta = meta.copy()
                current_seg = seg
    if current_meta is not None:
        merged.append((current_meta, current_seg))
    return merged

def get_speaker_samples_html():
    """
    Returns an HTML snippet showing one or more merged samples per unique speaker.
    Only merged segments longer than a minimum duration (e.g. 2 seconds) are shown.
    """
    import numpy as np
    if single_file_state["metadata"] is None or single_file_state["audio_tensor"] is None:
        return "Index a podcast first to see speaker samples."
    
    speaker_groups = {}
    for meta, seg in zip(single_file_state["metadata"], single_file_state["audio_tensor"]):
        speaker = meta.get("speaker", "unknown")
        if speaker not in speaker_groups:
            speaker_groups[speaker] = []
        speaker_groups[speaker].append((meta, seg))
    
    html = "<h3>Speaker Samples (Merged)</h3>"
    min_duration = 2.0  # seconds
    max_segments = 3    # max samples per speaker

    for speaker, segments in speaker_groups.items():
        merged_segments = merge_segments_for_speaker(segments, gap_threshold=1.0)
        html += f"<h4>{speaker}</h4>"

        if not merged_segments:
            html += "<div>No segments found.</div>"
            continue

        shown = 0
        for i, (meta, seg) in enumerate(merged_segments):
            if shown >= max_segments:
                break

            duration = meta["end_time"] - meta["start_time"]
            if duration < min_duration:
                continue

            try:
                seg_np = seg.cpu().numpy()

                if seg_np is None or seg_np.size == 0:
                    continue
                if np.any(np.isnan(seg_np)):
                    continue
                if seg_np.dtype != np.float32:
                    seg_np = seg_np.astype(np.float32)

                sample_html = audio_to_html(seg_np, single_file_state["sample_rate"])
                html += f"<div><b>Segment {shown + 1} ({duration:.1f} sec):</b><br>{sample_html}</div><br>"
                shown += 1

            except Exception as e:
                print(f"⚠️ Failed to render audio for speaker '{speaker}', segment {i+1}: {e}")
                continue

    return html


with gr.Blocks() as demo:
    gr.Markdown("# Podcast RAG System")

    # 🧠 Just placeholders, don't render yet
    extracted_mapping_text = None
    speaker_samples_html = None

    with gr.Tab("Index Podcast"):
        file_input = gr.Audio(type="filepath")
        upload_btn = gr.Button("Index Uploaded File")
        index_status = gr.Textbox()
        upload_btn.click(index_podcast, file_input, index_status)

        load_btn = gr.Button("🔁 Load Saved State")
        load_status = gr.Textbox()
        load_btn.click(
            fn=load_full_state,
            inputs=None,
            outputs=load_status,
        ).then(
            fn=get_extracted_speaker_mapping,
            inputs=None,
            outputs=extracted_mapping_text,
        ).then(
            fn=get_speaker_samples_html,
            inputs=None,
            outputs=speaker_samples_html,
        )

        save_btn = gr.Button("💾 Save Current State")
        save_status = gr.Textbox()
        save_btn.click(save_full_state, None, save_status)


    with gr.Tab("2️⃣ Assign Speakers"):
        gr.Markdown("### 🔊 Identify Speakers")

        mapping_json = gr.Textbox(value=json.dumps(custom_speaker_mapping), label="Speaker Mapping (JSON)")
        mapping_status = gr.Textbox(label="Mapping Update Status")
        update_mapping_btn = gr.Button("Update Speaker Mapping")
        update_mapping_btn.click(apply_speaker_mapping, mapping_json, mapping_status)

        refresh_samples_btn = gr.Button("Refresh Speaker Samples")
        speaker_samples_html = gr.HTML(label="Speaker Samples")  # ✅ defined here
        refresh_samples_btn.click(get_speaker_samples_html, None, speaker_samples_html)

        refresh_extracted_mapping_btn = gr.Button("Refresh Extracted Speaker Mapping")
        extracted_mapping_text = gr.Textbox(label="Extracted Speaker Mapping", interactive=False)  # ✅ defined here
        refresh_extracted_mapping_btn.click(get_extracted_speaker_mapping, None, extracted_mapping_text)


    with gr.Tab("Query Podcast"):
        query_input = gr.Audio(type="filepath", label="Spoken Query")
        lambda_slider = gr.Slider(
            minimum=0.0, 
            maximum=1.0, 
            step=0.05, 
            value=0.5, 
            label="MMR Lambda (0 = diverse, 1 = relevant)"
        )
        query_btn = gr.Button("Retrieve Answer")
        answer_output = gr.Textbox(label="LLM Answer")
        audio_output = gr.HTML(label="Retrieved Segments (Audio)")

        query_btn.click(query_rag, [query_input, lambda_slider], [answer_output, audio_output])

if __name__ == "__main__":
    demo.launch()
