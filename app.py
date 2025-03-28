import gradio as gr
import torch, torchaudio, numpy as np, time, json
from contextlib import contextmanager
from openai import OpenAI
from helpers import (
    segment_audio_silero,
    transcribe_with_fasterwhisper,
    audio_to_html,
    save_embeddings_to_jsonl,
    diarize_segment,
)
from embeddings import get_audio_embedding, get_clap_embedding, save_embeddings_to_jsonl, query_from_jsonl
from speakers import assign_speaker_activity_to_vad_chunks, transcribe_with_fasterwhisper_and_assign_speaker_labels
from rerankers import rerank_with_mmr, rerank_with_rrf
import os
from collections import defaultdict
import faiss

use_clap = True
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

per_podcast_speaker_mapping = {}

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
    #print(f"{name} took {end - start:.3f} seconds")

def update_all_metadata_with_mapping():
    if single_file_state["metadata"] is None:
        return
    for meta in single_file_state["metadata"]:
        title = meta.get("podcast_title")
        raw = meta.get("speaker_label", meta.get("speaker"))
        mapped_name = per_podcast_speaker_mapping.get(title, {}).get(raw, raw)
        meta["speaker"] = mapped_name

def generate_mapping_ui_for_all_podcasts():
    import json
    from collections import defaultdict

    if not single_file_state.get("metadata"):
        return "<div>📭 No metadata loaded.</div>", []
    boxes=[]
    MAX_SEGMENTS_PER_PODCAST = 3
    MAX_AUDIO_SECONDS = 60 

    metadata = single_file_state["metadata"]
    audio_segments = single_file_state["audio_tensor"]
    sample_rate = single_file_state["sample_rate"]

    # Group segments by podcast title
    title_to_segments = defaultdict(list)
    for i, meta in enumerate(metadata):
        title = meta.get("podcast_title", f"Untitled_{i}")
        title_to_segments[title].append((meta, audio_segments[i]))

    full_html = ""
    added_main_podcast = False


    for title, segments in title_to_segments.items():
        # 👇 One base podcast only (no "ft.")
        if "ft." not in title.lower():
            if added_main_podcast:
                continue  # already added base show
            added_main_podcast = True
        else:
            # 👇 Only include podcasts with "ft." in title
            if "ft." not in title.lower():
                continue

        segments = segments[:MAX_SEGMENTS_PER_PODCAST]

        # Speaker mapping preview
        current_mapping = {}
        for meta, _ in segments:
            raw = meta.get("speaker_label")
            mapped = meta.get("speaker", raw)
            current_mapping[raw] = mapped

        mapping_json = json.dumps(current_mapping, indent=2)
        boxes.append((title, mapping_json))

        full_html += f"<h3>🎙️ {title}</h3><div style='border:1px solid #aaa;padding:8px;margin-bottom:10px;'>"
        full_html += f"<b>Speaker Mapping:</b><pre>{mapping_json}</pre>"

        # Segment previews
        for i, (meta, snippet) in enumerate(segments):
            transcript = meta.get("transcript_with_speaker_labels", "No transcript.")
            try:
                snippet_np = snippet.cpu().numpy()
                if MAX_AUDIO_SECONDS:
                    snippet_np = snippet_np[:, :sample_rate * MAX_AUDIO_SECONDS]
                audio_html = audio_to_html(snippet_np, sample_rate)
            except Exception as e:
                audio_html = f"<div>Error loading audio: {e}</div>"

            full_html += f"<h5>Segment {i+1}</h5><pre>{transcript}</pre>{audio_html}<br><br>"

        full_html += "</div>"

    return full_html, boxes

def collect_and_apply_mappings(mapping_state, *textbox_values):
    combined = {}
    for (title, _), textbox in zip(mapping_state, textbox_values):
        try:
            combined[title] = json.loads(textbox)
        except Exception:
            return f"❌ Invalid JSON for episode: {title}"
    return apply_per_podcast_speaker_mapping(json.dumps(combined))


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
    for key, val in sorted(mapping.items(), key=lambda x: int(x[0].split('_')[1]) if len(x[0].split('_')) > 1 and x[0].split('_')[1].isdigit() else 0):
        mapping_str += f"{key}: {val}\n"
    return mapping_str

def save_full_state(path="data/full_state.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        # Save the PyTorch state
        state = {
            "embeddings_tensor": single_file_state["embeddings_tensor"].cpu(),
            "metadata": single_file_state["metadata"],
            "audio_tensor": [seg.cpu() for seg in single_file_state["audio_tensor"]],
            "sample_rate": single_file_state["sample_rate"],
        }
        torch.save(state, path)

        # Save speaker mapping to JSON file
        with open("data/speaker_mappings.json", "w", encoding="utf-8") as f:
            json.dump(per_podcast_speaker_mapping, f, indent=2)

        return "✅ Full state saved successfully!"
    except Exception as e:
        return f"❌ Error saving state: {str(e)}"

def load_full_state(path="data/full_state.pt"):
    if not os.path.exists(path):
        return "❌ No saved state found."

    try:
        print("📦 Attempting to load state from:", path)

        # ✅ Load first, so 'state' exists before using it
        state = torch.load(path, map_location=device)

        # Restore speaker mapping from external file
        global per_podcast_speaker_mapping
        if os.path.exists("data/speaker_mappings.json"):
            with open("data/speaker_mappings.json", "r", encoding="utf-8") as f:
                per_podcast_speaker_mapping = json.load(f)
            print("✅ Loaded speaker_mappings.json")
        else:
            per_podcast_speaker_mapping = {}
            print("⚠️ speaker_mappings.json not found — using empty mapping.")

        _ = apply_per_podcast_speaker_mapping(json.dumps(per_podcast_speaker_mapping))

        single_file_state["embeddings_tensor"] = state["embeddings_tensor"].to(device)
        single_file_state["metadata"] = state["metadata"]

        # Load precomputed normalized embeddings (for faster similarity search)
        emb_tensor = state["embeddings_tensor"].to(device)
        single_file_state["normalized_embeddings"] = torch.nn.functional.normalize(emb_tensor, dim=1)

        # Load JSONL once into memory
        jsonl_path = "data/segments.jsonl"
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r") as f:
                single_file_state["jsonl_entries"] = [json.loads(line) for line in f]
            print(f"📄 Loaded {len(single_file_state['jsonl_entries'])} entries from JSONL")

        single_file_state["audio_tensor"] = [seg.to(device) for seg in state["audio_tensor"]]
        single_file_state["sample_rate"] = state.get("sample_rate", 16000)

        global custom_speaker_mapping
        custom_speaker_mapping = state.get("speaker_mapping", custom_speaker_mapping)

        # 🔥 Apply mapping to all metadata after loading
        update_all_metadata_with_mapping()

        # ✅ Now that it's updated, save the corrected JSONL
        if single_file_state["embeddings_tensor"] is not None and single_file_state["metadata"] is not None:
            save_embeddings_to_jsonl("data/segments.jsonl", single_file_state["embeddings_tensor"], single_file_state["metadata"])
            print("💾 Resaved segments.jsonl with updated speaker labels.")

        print(f"✅ Loaded state with {len(single_file_state['metadata'])} segments.")
        return f"✅ Loaded full state with {len(single_file_state['metadata'])} segments."
    except Exception as e:
        print(f"❌ Exception in load_full_state: {e}")
        return f"❌ Error loading state: {str(e)}"


def apply_per_podcast_speaker_mapping(mapping_json_by_title):
    """
    Accepts a dict of { title: { speaker_0: Name, ... } }
    and updates metadata and global map accordingly.
    """
    global per_podcast_speaker_mapping

    try:
        mappings = json.loads(mapping_json_by_title)
        if not isinstance(mappings, dict):
            return "❌ Expected a dict of title -> speaker mapping objects."

        per_podcast_speaker_mapping = mappings

        # Apply new mapping to all segments
        for meta in single_file_state.get("metadata", []):
            title = meta.get("podcast_title", "Unknown")
            label = meta.get("speaker_label")
            if title in mappings and label in mappings[title]:
                meta["speaker"] = mappings[title][label]
            else:
                meta["speaker"] = label  # fallback to raw

        # Save new version to .jsonl
        save_embeddings_to_jsonl(
            "data/segments.jsonl",
            single_file_state["embeddings_tensor"],
            single_file_state["metadata"],
        )

        return "✅ Per-episode speaker mappings applied."
    except Exception as e:
        return f"❌ Error applying mappings: {str(e)}"


def index_podcast_folder(podcasts_folder="podcasts"):
    all_segments = []
    all_metadata = []

    # Iterate over all MP3 files in the podcasts folder
    for file in sorted(os.listdir(podcasts_folder)):
        if file.endswith(".mp3"):
            file_path = os.path.join(podcasts_folder, file)
            base_name = file.rsplit('.', 1)[0]
            json_path = os.path.join(podcasts_folder, f"{base_name}.json")
            metadata_info = {}
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata_info = json.load(f)

            #print(f"\n🔍 Indexing {file_path}...")

            # Load + resample audio
            audio_tensor, sample_rate = torchaudio.load(file_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                audio_tensor = resampler(audio_tensor)
                sample_rate = 16000

            # --- Limit to the first 5 minutes (300 seconds) ---
            max_duration_seconds = 60  # 5 minutes
            max_samples = int(max_duration_seconds * sample_rate)
            if audio_tensor.shape[1] > max_samples:
                audio_tensor = audio_tensor[:, :max_samples]
            # ---------------------------------------------------

            audio_np = audio_tensor.cpu().numpy()

            # Diarization
            diarization_results = diarize_segment(audio_np, sample_rate)
            #print(f"🧠 Diarization produced {len(diarization_results)} segments.")

            # VAD segmentation
            vad_segments, vad_metadata = segment_audio_silero(audio_tensor, sample_rate)
            #print(f"🔊 VAD produced {len(vad_segments)} speech chunks.")

            # ---- Attach raw speaker activity to each VAD segment ----
            # Pass an empty dict for the custom mapping to use raw labels.
            merged_segments, merged_metadata = assign_speaker_activity_to_vad_chunks(
                vad_segments, vad_metadata, diarization_results, {}
            )
            #print(f"🎤 Assigned speaker activity to {len(merged_segments)} VAD segments.")

            # Attach podcast metadata to each segment
            for meta in merged_metadata:
                meta["podcast_title"] = metadata_info.get("title", base_name)
                meta["upload_date"] = metadata_info.get("upload_date", "N/A")
                meta["uploader"] = metadata_info.get("uploader", "N/A")
                meta["webpage_url"] = metadata_info.get("webpage_url", "N/A")
                if not isinstance(meta["webpage_url"], str) or meta["webpage_url"].startswith("<audio"):
                    meta["webpage_url"] = "N/A"

            # Use these segments from here on
            all_segments.extend(merged_segments)
            all_metadata.extend(merged_metadata)

    #print(f"\n✅ Total segments collected from folder: {len(all_segments)}")
    avg_duration = np.mean([m["end_time"] - m["start_time"] for m in all_metadata])
    #print(f"📊 Avg duration of segments: {avg_duration:.2f}s")

    # Compute embeddings and transcriptions
    embeddings = []
    stored_segments = []
    for i, seg in enumerate(all_segments):
        if seg.numel() == 0:
            #print(f"⚠️ Skipping empty segment {i}")
            continue
        emb = get_clap_embedding(seg, 16000)
        if np.any(np.isnan(emb)):
            #print(f"⚠️ NaN in embedding for segment {i}")
            continue
        embeddings.append(emb)
        stored_segments.append(seg)
        # Transcribe the segment using the new helper; this assigns raw speaker labels.
        transcript = transcribe_with_fasterwhisper_and_assign_speaker_labels(seg, 16000, all_metadata[i], language="en")
        # Save the transcript in metadata under the key "transcript_with_speaker_labels"
        all_metadata[i]["transcript_with_speaker_labels"] = transcript

    if not embeddings:
        return "❌ No valid embeddings computed from the podcasts folder."

    embeddings_tensor = torch.from_numpy(np.stack(embeddings)).float().to(device)

    # Save to global state
    single_file_state["embeddings_tensor"] = embeddings_tensor
    single_file_state["metadata"] = all_metadata
    single_file_state["audio_tensor"] = stored_segments
    single_file_state["sample_rate"] = 16000

    #print(f"💾 Saving {len(all_metadata)} metadata entries and {len(stored_segments)} audio segments.")
    durations = [m["end_time"] - m["start_time"] for m in all_metadata]
    #print(f"⏱️ Final avg duration: {np.mean(durations):.2f}s")

    # Save
    save_embeddings_to_jsonl("data/segments.jsonl", embeddings_tensor, all_metadata)
    print("Embeddings saved")
    torch.save(stored_segments, "data/audio_segments.pt")
    save_full_state("data/full_state.pt")
    print("Full state saved")
    
    return f"✅ Indexed podcasts folder: {len(stored_segments)} segments from all podcasts."


def query_rag(query_audio_file, lambda_param=0.5, length_alpha=0.05):
    """
    1) Loads and embeds the query audio.
    2) Reads all stored segments from data/segments.jsonl, normalizes their embeddings.
    3) Computes similarity to the query, applies a length-based boost, and picks top candidates.
    4) Uses precomputed transcriptions (with raw speaker labels) from indexing to build the final answer.
    
    Args:
        query_audio_file (str): Path to the user's query audio file.
        lambda_param (float): MMR trade-after.
        length_alpha (float): Strength of the length boost.
    """
    import math
    import json
    import torch
    import numpy as np
    from torch.nn import functional as F

    if query_audio_file is None:
        return "Provide a query audio clip.", "", []

    if not os.path.exists("data/segments.jsonl"):
        return "Index a podcast first.", "", []

    # 1) Load & resample query audio
    query_tensor, query_sr = torchaudio.load(query_audio_file)
    if query_sr != 16000:
        resampler = torchaudio.transforms.Resample(query_sr, 16000)
        query_tensor = resampler(query_tensor)
        query_sr = 16000
    if query_tensor.dim() > 1:
        query_tensor = query_tensor.mean(dim=0)

    # Transcribe the user's query (for context)
    query_audio_np = query_tensor.cpu().numpy()
    query_transcription = transcribe_with_fasterwhisper(query_audio_np, query_sr)

    # 2) Compute the query embedding
    query_embedding = get_clap_embedding(query_tensor, query_sr)
    query_tensor = torch.tensor(query_embedding).float()
    query_tensor = F.normalize(query_tensor, dim=0).unsqueeze(0)

    # 3) Load & normalize all indexed embeddings
    entries = single_file_state.get("jsonl_entries", [])
    all_embeddings_tensor = single_file_state.get("normalized_embeddings", None)
    if all_embeddings_tensor is not None:
        all_embeddings_tensor = all_embeddings_tensor.to(query_tensor.device)

    if not entries or all_embeddings_tensor is None:
        return "Embeddings or metadata not loaded. Please index or load a saved state first.", "", []

    # Compute similarity
    query_tensor = query_tensor.to(all_embeddings_tensor.device)

    # Compute similarity
    query_np = query_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(query_np.reshape(1, -1))  # normalize query for cosine

    # Prepare FAISS index (if not already built and cached)
    if "faiss_index" not in single_file_state:
        all_embeddings_np = all_embeddings_tensor.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(all_embeddings_np)  # normalize embeddings for cosine
        index = faiss.IndexFlatIP(all_embeddings_np.shape[1])
        index.add(all_embeddings_np)
        single_file_state["faiss_index"] = index
    else:
        index = single_file_state["faiss_index"]

    # Run FAISS search
    top_k_pool = 50
    D, I = index.search(query_np.reshape(1, -1), k=min(top_k_pool, len(entries)))
    top_indices = I[0].tolist()
    top_results = [entries[i] for i in top_indices][:10]

    
    # Optional reranking: Uncomment ONE of the following
    # 🔁 RRF
    # top_results = rerank_with_rrf(top_results, [boosted_scores[i] for i in top_indices])[:5]

    # 🔁 MMR
    # candidate_embs = all_embeddings_tensor[top_indices]
    # top_k_mmr_indices = rerank_with_mmr(query_tensor, candidate_embs, top_k=len(top_indices), lambda_param=lambda_param)
    # top_results = [top_results[i] for i in top_k_mmr_indices][:5]

    # Get the title of the main matching segment
    titles_in_results = set(
        r["metadata"].get("podcast_title", "Unknown") for r in top_results
    )

    # def format_speaker_mapping_block(titles):
    #     lines = ["Speaker Mapping by Episode:"]
    #     for title in titles:
    #         speaker_map = per_podcast_speaker_mapping.get(title, {})
    #         if not speaker_map:
    #             continue
    #         lines.append(f"- {title}:")
    #         for speaker_id, name in speaker_map.items():
    #             lines.append(f"  {speaker_id} = {name}")
    #     return "\n".join(lines)
    # speaker_mapping_block = format_speaker_mapping_block(titles_in_results)

    # 4) Build final answer with context segments
    prompt = f"User: {query_transcription}\n"
    prompt += "Transcription:\n"

    added_ids = set()
    segment_contexts = []
    audio_segments_to_display = []

    for rank, result in enumerate(top_results):
        idx = result["segment_id"]

        for offset in [-1, 0, 1]:
            neighbor_idx = idx + offset
            if neighbor_idx < 0 or neighbor_idx >= len(single_file_state["metadata"]):
                continue
            if neighbor_idx in added_ids:
                continue

            added_ids.add(neighbor_idx)
            meta = single_file_state["metadata"][neighbor_idx]
            seg_transcription = meta.get("transcript_with_speaker_labels", "No transcription available.")
            start = meta["start_time"]
            end = meta["end_time"]
            duration = end - start
            title = meta.get("podcast_title", "Unknown")
            speaker = meta.get("speaker", "unknown")

            role = "relevant match" if neighbor_idx == idx else "context"
            seg_num = len(segment_contexts) + 1

            prompt += (
                f"Segment {seg_num} ({role}):\n"
                f"[{start:.2f}s – {end:.2f}s] (duration: {duration:.1f}s)\n"
                f"{seg_transcription}\n"
                f"in the episode: {title}\n\n"
            )

            segment_contexts.append((speaker, seg_transcription))

            snippet = single_file_state["audio_tensor"][neighbor_idx]
            snippet_np = snippet.cpu().numpy()
            audio_html = (
                f"<div><b>{title}:</b><br>"
                f"<b>Segment {seg_num} ({role})</b><br>"
                f"{audio_to_html(snippet_np, single_file_state['sample_rate'])}<br></div><br>"
            )
            audio_segments_to_display.append(audio_html)

    mapping_block = "Speaker Mappings:\n" + "\n".join(
        f"{title}: {json.dumps(mapping, indent=2)}" 
        for title, mapping in per_podcast_speaker_mapping.items()
    )

    # 🔥 Final prompt to LLM
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    f"""You are a helpful assistant that uses only the provided transcription to answer the user's query. 
                    Please cite the *Segment number* (e.g., "Segment 2"F) and answer in your own words. 
                    Do not use timestamps. 
                    Speaker labels like "speaker_0" appear in the transcript. Here are the mappings of speaker labels to real speaker names:
                    {mapping_block}

                    If the user asks about a specific speaker, answer only from that speaker's content. 
                    If it’s a general or indirect query, include perspectives from multiple speakers if relevant.

                    Use only the most relevant segments (marked as 'relevant match') when forming your main answer."""
                                    )
            },
            {"role": "user", "content": prompt + "\nAssistant:\n"}
        ],
        temperature=0.7,
        max_tokens=500,
    )
    answer = response.choices[0].message.content

    return answer, "".join(audio_segments_to_display), segment_contexts

def get_example_segments_and_timelines_html():
    """
    Returns an HTML snippet showing 2 example segments per unique podcast title,
    along with their speaker timelines and audio players.
    """
    import numpy as np
    from collections import defaultdict

    metadata = single_file_state.get("metadata")
    audio_segments = single_file_state.get("audio_tensor")
    sample_rate = single_file_state.get("sample_rate")

    if not metadata or not audio_segments:
        return "<div>No segments available. Please index a podcast first.</div>"

    html = "<h3>Example Segments and Speaker Timelines</h3>"

    try:
        # Group by Podcast Title
        title_to_segments = defaultdict(list)
        for i, meta in enumerate(metadata):
            title = meta.get("Podcast Title", f"Untitled_{i}")
            title_to_segments[title].append((meta, audio_segments[i]))

        if not title_to_segments:
            html += "<div><strong>Debug:</strong> No podcast titles found in metadata.</div>"
            return html

        max_podcasts = 31
        max_segments_per_podcast = 3
        segment_counter = 1

        # Only grab up to 12 unique podcasts
        for title, segments in list(title_to_segments.items())[:max_podcasts]:
            html += f"<h4>Podcast: {title} ({len(segments)} segments)</h4>"

            for meta, snippet in segments[:max_segments_per_podcast]:
                transcript = meta.get("transcript_with_speaker_labels", "No transcription available.")
                speaker_activity = meta.get("speaker_activity", {})

                # Speaker timeline
                timeline_html = "<ul>"
                for speaker, intervals in speaker_activity.items():
                    intervals_str = ", ".join([f"{interval['start']:.1f}s–{interval['end']:.1f}s" for interval in intervals])
                    timeline_html += f"<li>{speaker}: {intervals_str}</li>"
                timeline_html += "</ul>"

                try:
                    snippet_np = snippet.cpu().numpy()
                    audio_html = audio_to_html(snippet_np, sample_rate)
                except Exception as e:
                    audio_html = f"<div>Error loading audio: {e}</div>"

                start_time = meta.get("start_time", 0)
                end_time = meta.get("end_time", 0)
                duration = end_time - start_time

                html += (
                    f"<div style='border:1px solid #ccc; margin-bottom:10px; padding:5px;'>"
                    f"<h5>Segment {segment_counter} (Duration: {duration:.1f}s)</h5>"
                    f"<pre>{transcript}</pre>"
                    f"<strong>Speaker Timeline:</strong> {timeline_html}"
                    f"<div><strong>Audio:</strong><br>{audio_html}</div>"
                    f"</div>"
                )
                segment_counter += 1

        if segment_counter == 1:
            html += "<div><strong>Debug:</strong> Grouping worked but no segments were added to the output.</div>"

    except Exception as e:
        html += f"<div><strong>Error:</strong> {str(e)}</div>"

    return html

def get_per_podcast_speaker_mapping():
    """
    Builds a dict: { podcast_title: { speaker_label: speaker_name } }
    based on your custom_speaker_mapping and existing metadata.
    """
    mapping = {}

    for meta in single_file_state.get("metadata", []):
        title = meta.get("podcast_title", "Unknown Podcast")
        raw_label = meta.get("speaker_label", None)
        mapped_name = meta.get("speaker", raw_label)

        if title not in mapping:
            mapping[title] = {}
        
        if raw_label and raw_label not in mapping[title]:
            mapping[title][raw_label] = mapped_name

    return mapping

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Podcast RAG System")

    with gr.Tab("Save and Load"):
        load_btn = gr.Button("🔁 Load Saved State")
        load_status = gr.Textbox()
        extracted_mapping_text = gr.Textbox(label="Extracted Mapping", visible=False)
        speaker_samples_html = gr.HTML(label="Speaker Samples", visible=False)

        load_btn.click(fn=load_full_state, inputs=None, outputs=load_status
        ).then(
            fn=get_extracted_speaker_mapping,
            inputs=None,
            outputs=extracted_mapping_text
        ).then(
            fn=get_example_segments_and_timelines_html,
            inputs=None,
            outputs=speaker_samples_html
        )

        save_btn = gr.Button("💾 Save Current State")
        save_status = gr.Textbox()
        save_btn.click(save_full_state, None, save_status)

    with gr.Tab("Index Podcast Folder"):
        gr.Markdown("### 🔍 Index all podcasts from the 'podcasts' folder")
        index_folder_btn = gr.Button("Index Podcast Folder")
        index_folder_status = gr.Textbox(label="Index Folder Status")
        index_folder_btn.click(index_podcast_folder, None, index_folder_status)
    
    with gr.Tab("🗣 Speaker Samples"):
        speaker_samples_html = gr.HTML()
        refresh_btn = gr.Button("🔄 Refresh Samples")

        # Call the display function on click
        refresh_btn.click(fn=get_example_segments_and_timelines_html, inputs=None, outputs=speaker_samples_html)

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

        segment_contexts_state = gr.State()
        query_btn.click(query_rag, [query_input, lambda_slider], [answer_output, audio_output, segment_contexts_state])

# Launch the Gradio interface
demo.launch()

if __name__ == "__main__":
    demo.launch(debug=False)
