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
from embeddings import get_audio_embedding, log_debug, get_clap_embedding, save_embeddings_to_jsonl, query_from_jsonl
from speakers import assign_speaker_activity_to_vad_chunks, transcribe_with_fasterwhisper_and_assign_speaker_labels
from rerankers import reciprocal_rank_fusion, rerank_segments
import os
from collections import defaultdict
import faiss
import csv
from pathlib import Path
import time
torchaudio.set_audio_backend("ffmpeg")

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
    print(f"{name} took {end - start:.3f} seconds")

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

    boxes = []
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
        is_guest = "ft." in title.lower()

        if is_guest:
            section_label = "🎤 Guest Podcast"
        else:
            if added_main_podcast:
                continue  # already added one base podcast
            added_main_podcast = True
            section_label = "🎧 Base Podcast"

        # Only show one segment per podcast title
        meta, snippet = segments[0]

        # Speaker mapping
        raw = meta.get("speaker_label")
        mapped = meta.get("speaker", raw)
        current_mapping = {raw: mapped}
        mapping_json = json.dumps(current_mapping, indent=2)
        boxes.append((title, mapping_json))

        # HTML
        full_html += f"<h3>{section_label}: {title}</h3>"
        full_html += "<div style='border:1px solid #aaa;padding:8px;margin-bottom:10px;'>"
        full_html += f"<b>Speaker Mapping:</b><pre>{mapping_json}</pre>"

        transcript = meta.get("transcript_with_speaker_labels", "No transcript.")
        try:
            snippet_np = snippet.cpu().numpy()
            snippet_np = snippet_np[:, :sample_rate * MAX_AUDIO_SECONDS]
            audio_html = audio_to_html(snippet_np, sample_rate)
        except Exception as e:
            audio_html = f"<div>Error loading audio: {e}</div>"

        full_html += f"<h5>Segment Preview</h5><pre>{transcript}</pre>{audio_html}<br><br>"
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

def fix_missing_transcripts():
    print("🔍 Checking for missing transcripts after loading state...")

    num_fixed = 0

    for idx, meta in enumerate(single_file_state["metadata"]):
        transcript = meta.get("transcript_with_speaker_labels", "").strip()

        if not transcript:
            print(f"⚡ Transcribing missing transcript for segment {idx}...")
            try:
                snippet = single_file_state["audio_tensor"][idx]
                snippet_np = snippet.cpu().numpy()

                # Use your speaker-aware transcription
                transcript = transcribe_with_fasterwhisper_and_assign_speaker_labels(
                    snippet_np,
                    single_file_state["sample_rate"],
                    meta
                )

                if not transcript.strip():
                    transcript = "[Empty after transcription]"

                meta["transcript_with_speaker_labels"] = transcript
                num_fixed += 1

            except Exception as e:
                print(f"❌ Failed to transcribe segment {idx}: {e}")
                meta["transcript_with_speaker_labels"] = "[Error transcribing]"

    print(f"✅ Finished fixing transcripts. {num_fixed} segments updated.")
    return num_fixed

def load_full_state(path="data/full_state.pt"):
    if not os.path.exists(path):
        return "❌ No saved state found."

    try:
        print("📦 Attempting to load state from:", path)

        # ✅ Load saved state
        state = torch.load(path, map_location=device)

        # Restore speaker mapping
        global per_podcast_speaker_mapping
        if os.path.exists("data/speaker_mappings.json"):
            with open("data/speaker_mappings.json", "r", encoding="utf-8") as f:
                per_podcast_speaker_mapping = json.load(f)
            print("✅ Loaded speaker_mappings.json")
        else:
            per_podcast_speaker_mapping = {}
            print("⚠️ speaker_mappings.json not found — using empty mapping.")

        _ = apply_per_podcast_speaker_mapping(json.dumps(per_podcast_speaker_mapping))

        # Restore main state
        single_file_state["embeddings_tensor"] = state["embeddings_tensor"].to(device)
        single_file_state["metadata"] = state["metadata"]

        emb_tensor = state["embeddings_tensor"].to(device)
        normalized_embeddings = torch.nn.functional.normalize(emb_tensor, dim=1)
        single_file_state["normalized_embeddings"] = normalized_embeddings

        if os.path.exists("data/segments.jsonl"):
            with open("data/segments.jsonl", "r") as f:
                single_file_state["jsonl_entries"] = [json.loads(line) for line in f]
            print(f"📄 Loaded {len(single_file_state['jsonl_entries'])} entries from JSONL")

        single_file_state["audio_tensor"] = [seg.to(device) for seg in state["audio_tensor"]]
        single_file_state["sample_rate"] = state.get("sample_rate", 16000)

        global custom_speaker_mapping
        custom_speaker_mapping = state.get("speaker_mapping", custom_speaker_mapping)

        # 🔥 Apply mapping to metadata
        update_all_metadata_with_mapping()

        # 🔥🔥 Build FAISS index once here
        print("⚙️ Building FAISS index...")
        embeddings_np = normalized_embeddings.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(embeddings_np)

        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatIP(embeddings_np.shape[1])
            index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            index = faiss.IndexFlatIP(embeddings_np.shape[1])

        index.add(embeddings_np)
        single_file_state["faiss_index"] = index
        print(f"✅ FAISS index built with {embeddings_np.shape[0]} vectors.")

        print(f"✅ Loaded state with {len(single_file_state['metadata'])} segments.")
        
        num_fixed = fix_missing_transcripts()

        if num_fixed > 0:
            print("💾 Auto-saving updated full_state.pt...")
            save_full_state("data/full_state_autosave.pt")


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

import subprocess 

def load_mp3_ffmpeg(filepath, target_sr=16000):
    cmd = [
        "ffmpeg",
        "-i", filepath,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    audio_data = process.stdout.read()
    audio_np = np.frombuffer(audio_data, dtype=np.float32)
    audio_tensor = torch.tensor(audio_np).unsqueeze(0)  # shape: [1, samples]
    print(f"🧪 Loaded {filepath} — shape: {audio_tensor.shape}, max: {audio_tensor.max().item()}, min: {audio_tensor.min().item()}")
    return audio_tensor, target_sr

def index_podcast_folder(podcasts_folder="podcasts"):
    all_segments = []
    all_metadata = []

    # Iterate over all MP3 files in the podcasts folder
    for file in sorted(os.listdir(podcasts_folder)):
        if file.endswith((".mp3", ".wav", ".webm")):
            file_path = os.path.join(podcasts_folder, file)
            base_name = file.rsplit('.', 1)[0]
            json_path = os.path.join(podcasts_folder, f"{base_name}.json")
            metadata_info = {}
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata_info = json.load(f)

            print(f"\n🔍 Indexing {file_path}...")

            # Load + resample audio
            audio_tensor, sample_rate = load_mp3_ffmpeg(file_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                audio_tensor = resampler(audio_tensor)
                sample_rate = 16000

            # --- Limit to the first 5 minutes (300 seconds) (Optional for debugging) ---
            # max_duration_seconds = 60  # 5 minutes
            # max_samples = int(max_duration_seconds * sample_rate)
            # if audio_tensor.shape[1] > max_samples:
            #     audio_tensor = audio_tensor[:, :max_samples]

            audio_np = audio_tensor.cpu().numpy()

            # Diarization
            diarization_results = diarize_segment(audio_np, sample_rate)
            print(f"🧠 Diarization produced {len(diarization_results)} segments.")

            # VAD segmentation
            vad_segments, vad_metadata = segment_audio_silero(audio_tensor, sample_rate)
            print(f"🔊 VAD produced {len(vad_segments)} speech chunks.")

            # ---- Attach raw speaker activity to each VAD segment ----
            # Pass an empty dict for the custom mapping to use raw labels.
            merged_segments, merged_metadata = assign_speaker_activity_to_vad_chunks(
                vad_segments, vad_metadata, diarization_results, {}
            )
            print(f"🎤 Assigned speaker activity to {len(merged_segments)} VAD segments.")

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

    print(f"\n✅ Total segments collected from folder: {len(all_segments)}")
    avg_duration = np.mean([m["end_time"] - m["start_time"] for m in all_metadata])
    print(f"📊 Avg duration of segments: {avg_duration:.2f}s")

    # Compute embeddings and transcriptions
    embeddings = []
    stored_segments = []
    valid_metadata = []

    for i, seg in enumerate(all_segments):
        if seg is None or seg.numel() == 0:
            print(f"⚠️ Skipping empty segment {i}")
            continue
        log_debug(f"📦 Segment {i} stats — shape: {seg.shape}, min: {seg.min().item():.4f}, max: {seg.max().item():.4f}, mean: {seg.mean().item():.4f}")
        try:
            emb = get_clap_embedding(seg, 16000)

            if emb is None or np.any(np.isnan(emb)) or emb.shape != (512,):
                print(f"⚠️ Invalid embedding for segment {i}, skipping")
                continue

            embeddings.append(emb)
            stored_segments.append(seg)
            valid_metadata.append(all_metadata[i])

        except Exception as e:
            print(f"❌ Error at segment {i}: {e}")
            continue

    if not embeddings:
        raise RuntimeError("❌ No valid embeddings were created after processing all segments. Cannot continue.")

    # Now embeddings are guaranteed valid
    embeddings_tensor = torch.from_numpy(np.stack(embeddings)).float().to(device)

    # Store
    single_file_state["embeddings_tensor"] = embeddings_tensor
    single_file_state["metadata"] = valid_metadata
    single_file_state["audio_tensor"] = stored_segments
    single_file_state["sample_rate"] = 16000


    print(f"💾 Saving {len(all_metadata)} metadata entries and {len(stored_segments)} audio segments.")
    durations = [m["end_time"] - m["start_time"] for m in all_metadata]
    print(f"⏱️ Final avg duration: {np.mean(durations):.2f}s")
    print(f"📝 Transcribing {len(merged_segments)} segments with speaker labels...")

    for seg, meta in zip(merged_segments, merged_metadata):
        try:
            seg_np = seg.cpu().numpy()
            transcript = transcribe_with_fasterwhisper_and_assign_speaker_labels(
                seg_np,  # audio segment
                sample_rate,  # 16000 Hz
                meta  # contains "start_time" and "speaker_activity"
            )
            meta["transcript_with_speaker_labels"] = transcript.strip()
        except Exception as e:
            print(f"❌ Error transcribing a segment: {e}")
            meta["transcript_with_speaker_labels"] = "[Error during transcription]"

    # Save
    save_embeddings_to_jsonl("data/segments.jsonl", embeddings_tensor, all_metadata)
    print("Embeddings saved")
    torch.save(stored_segments, "data/audio_segments.pt")
    save_full_state("data/full_state.pt")
    print("Full state saved")
    
    return f"✅ Indexed podcasts folder: {len(stored_segments)} segments from all podcasts."


def run_evaluation():

    queries = [
    "What are some other anime ‘genre mashups’ that completely surprised you?",
    "Garnt, what’s the silliest sibling argument you remember, and how did you end up working it out?",
    "Joey, how do you even plan a trip to a place with so little tourism info available, like Greenland or Madagascar?",
    "Connor, what would you tell your younger self about ‘fitting in’ given your experiences with coin-flip vodka games and peer pressure at Uni?",
    "What’s your new minimum standard for accommodations on tour after the horrifying LA shower and cobweb-filled top floor?",
    "Do you think you’ll ever reach Connor’s 40-second gold-split shower record, or is that physically impossible for you?",
    "How do musicals and stage plays influence the anime community compared to traditional marketing?",
    "Did your school sports positions translate into how you approach teamwork as adults?",
    "How do you balance traveling for exotic adventures and sticking to comfort zones like needing a hot shower each morning?",
    "What’s the first rule or guideline for replicating the ‘Seven Wonders of Anime’ discussion?",
    "What titles would make your personal ‘Seven Wonders of Anime’ and why?",
    "What ultimately pushes ‘Gundam’ or ‘Evangelion’ over the other for the mecha spot?",
    "Garnt, do you still think ‘One Piece’ can surpass Goku’s legacy once it ends?",
    "Did ‘Haruhi Suzumiya’ genuinely shape modern anime adaptations more than ‘Death Note’?",
    "What makes ‘High School of the Dead’ stand out from other ecchi/harem shows?",
    "Between ‘Astro Boy,’ ‘Sailor Moon,’ and other classics, which one belongs on the anime Mount Rushmore?",
    "Connor, have your feelings on ‘Akira’ changed—do you now see it as a top-tier anime movie?",
    "Was there an underrated anime you fought to include in the Wonders debate that got shut down?",
    "What’s one anti-bucket list activity you each tried once and vowed never to do again?",
    "How did ‘Ring of Fire’ and dirty pints shape your drinking habits today?",
    "What’s your worst blackout story and how did it feel waking up not knowing where you were?",
    "Joey, do you regret the kamikaze tequila shot with snorting salt and lemon in your eye, or would you do it again?",
    "Garnt, is it true you once drank washing-up liquid during a game? How did that happen?",
    "What liquor can each of you no longer stomach because of a bad hangover?",
    "Connor, did the coin-flipping vodka game night make you hate doing mass shots?",
    "What’s your take on the ‘multiday hangover,’ and how do you survive them now that you're older?",
    "What travel conditions now make or break a trip for you?",
    "If you could pick one unconventional travel destination, where would you go and why?",
    "Joey, is Machu Picchu still on your bucket list or has another place replaced it?",
    "Which is worse: cold showers every day or no access to warm water at all?",
    "Garnt, how did living as a monk in Thailand shape your view on daily comforts like beds and hot showers?",
    "Can the one-minute shower routine ever replace your normal shower, or is that impossible?",
    "Why are you all so different about morning vs. night showers? Does it cause trip friction?",
    "What was your worst shared bathroom experience—especially the LA top-floor horror?",
    "Connor, did that LA shower change how you view traveling for shows?",
    "Joey and Connor, what’s your best tip to get Garnt to show up on time despite his long morning showers?",
    "Garnt, what’s your Windows XP boot-up morning routine and why must it include a hot shower?",
    "Do you miss sibling fights or appreciate your calmer adult relationships now?",
    "What’s your funniest childhood sibling fight story—did any get out of hand?",
    "Joey, how did growing up with one sister compare to Connor’s chaos of multiple brothers?",
    "If you had to recommend a single anime to a newbie (not Death Note or AOT), what would it be?",
    "What’s your advice for surviving college drinking culture?",
    "Has your perspective on clubbing changed after some awful nights?",
    "Would you actually consider freezing for a once-in-a-lifetime trip to Greenland or Antarctica?",
    "Did hating rugby or football as kids affect your views on fitness as adults?",
    "Connor, do you ever miss being a goalie or is that firmly on your anti-bucket list?",
    "If forced to retry a winter sport, which would you choose—snowboarding, skiing, etc.?",
    "Looking back, were those wild party nights worth the stories or just dumb?",
    "What anime best captures the experience of being a broke, binge-drinking college student?",
    "If you made a Trash Taste Bingo Card, what squares would be absolutely required?"
    ]
    AGENT_NAME = "voxrag"
    QUERY_AUDIO_DIR = Path("eval/queries_audio")
    DOC_FILE = "eval/documents_vanilla.csv"
    ANSWER_FILE = "eval/answers_vanilla.csv"
    TIMING_FILE = "eval/timing_vanilla.csv"

    try:
        with open(DOC_FILE, "w", newline='', encoding="utf-8") as doc_f, \
             open(ANSWER_FILE, "w", newline='', encoding="utf-8") as ans_f, \
             open(TIMING_FILE, "w", newline='', encoding="utf-8") as timing_f:

            doc_writer = csv.writer(doc_f)
            doc_writer.writerow(["qid", "did", "document"])

            ans_writer = csv.writer(ans_f)
            ans_writer.writerow(["qid", "agent", "answer"])

            timing_writer = csv.writer(timing_f)
            timing_writer.writerow(["qid", "elapsed_seconds"])

            for i, query in enumerate(queries):
                audio_path = QUERY_AUDIO_DIR / f"audios--{i+1:02d}.wav"

                if not audio_path.exists():
                    print(f"⚠️ Skipping Q{i}: {audio_path.name} does not exist.")
                    ans_writer.writerow([i, AGENT_NAME, "ERROR: Missing audio file"])
                    timing_writer.writerow([i, "ERROR: Missing audio file"])
                    continue

                print(f"🔍 Processing Q{i}: {query}")
                print(f"🎧 Using audio: {audio_path}")

                try:
                    start_time = time.time()

                    result = query_rag(str(audio_path), single_file_state)

                    end_time = time.time()
                    elapsed = end_time - start_time

                    if not isinstance(result, tuple) or len(result) != 4:
                        raise ValueError("query_rag() must return a 4-tuple: (answer, html, all_context, retrieved_only)")

                    answer, html, all_context, retrieved_only = result

                    if not retrieved_only or not isinstance(retrieved_only, list):
                        raise ValueError("Retrieved segments missing or not a list")

                    for j, (did, speaker, text) in enumerate(retrieved_only):
                        clean_text = text.strip().replace("\n", " ")
                        doc_writer.writerow([i, did, f"{speaker} said: {clean_text}"])

                    ans_writer.writerow([i, AGENT_NAME, answer.strip()])
                    timing_writer.writerow([i, f"{elapsed:.4f}"])

                    print(f"✅ Q{i} completed in {elapsed:.2f} seconds.")

                except Exception as e:
                    print(f"❌ Error on Q{i}: {e}")
                    ans_writer.writerow([i, AGENT_NAME, "ERROR"])
                    timing_writer.writerow([i, "ERROR"])

        return "✅ Evaluation completed. Results saved to documents, answers, and timing CSVs."

    except Exception as e:
        return f"❌ Evaluation failed: {e}"
    
def query_rag(query_audio_file, lambda_param=0.5, length_alpha=0.05):
    import os
    import torch
    import numpy as np
    import time
    from torch.nn import functional as F

    if query_audio_file is None:
        print("❌ No query_audio_file provided.")
        return "Provide a query audio clip.", "", []

    if not os.path.exists(query_audio_file):
        print(f"❌ Audio file not found: {query_audio_file}")
        return "Audio file not found.", "", []

    total_start = time.perf_counter()
    print(f"🎧 Loading audio: {query_audio_file}")

    load_start = time.perf_counter()
    query_tensor, query_sr = load_mp3_ffmpeg(query_audio_file)
    print(f"🔵 Audio load took {time.perf_counter() - load_start:.2f}s")

    if query_tensor is None or query_tensor.numel() == 0:
        print("❌ Failed to load or empty audio.")
        return "Failed to load audio file.", "", []

    if query_sr != 16000:
        print("🔁 Resampling to 16kHz")
        resampler = torchaudio.transforms.Resample(query_sr, 16000)
        query_tensor = resampler(query_tensor)

    if query_tensor.dim() > 1:
        query_tensor = query_tensor.mean(dim=0)

    transcribe_start = time.perf_counter()
    query_audio_np = query_tensor.cpu().numpy()
    query_transcription = transcribe_with_fasterwhisper(query_audio_np, 16000)
    print(f"🟣 Transcription took {time.perf_counter() - transcribe_start:.2f}s")

    embed_start = time.perf_counter()
    query_embedding = get_clap_embedding(query_tensor, 16000)
    if query_embedding is None:
        print("❌ Failed to get query embedding.")
        return "Failed to create embedding.", "", []

    query_tensor = torch.tensor(query_embedding).float()
    query_tensor = F.normalize(query_tensor, dim=0).unsqueeze(0)
    print(f"🟢 Embedding + normalize took {time.perf_counter() - embed_start:.2f}s")

    entries = single_file_state.get("jsonl_entries", [])
    all_embeddings_tensor = single_file_state.get("normalized_embeddings", None)

    if not entries or all_embeddings_tensor is None:
        print("❌ Embeddings or entries missing from state.")
        return "Embeddings or metadata not loaded.", "", []

    all_embeddings_tensor = all_embeddings_tensor.to(query_tensor.device)
    query_tensor = query_tensor.to(all_embeddings_tensor.device)

    query_np = query_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(query_np.reshape(1, -1))

    faiss_start = time.perf_counter()

    index = single_file_state.get("faiss_index")
    if index is None:
        print("❌ FAISS index missing — please load or reindex.")
        return "FAISS index missing.", "", []

    print(f"🛠 FAISS setup took {time.perf_counter() - faiss_start:.2f}s")

    search_start = time.perf_counter()
    top_k_pool = 100
    D, I = index.search(query_np.reshape(1, -1), k=min(top_k_pool, len(entries)))
    print(f"🔍 FAISS search took {time.perf_counter() - search_start:.2f}s")

    top_indices = I[0].tolist()
    valid_indices = [i for i in top_indices if 0 <= i < len(entries)]
    top_results = [entries[i] for i in valid_indices][:10]

    # Build prompt
    prompt_start = time.perf_counter()

    prompt = f"User: {query_transcription}\nTranscription:\n"
    added_ids = set()
    segment_contexts = []
    audio_segments_to_display = []

    for result in top_results:
        idx = result["segment_id"]
        for neighbor_idx in [idx - 1, idx, idx + 1]:
            print(f"Checking segment {neighbor_idx}...")

            if not (0 <= neighbor_idx < len(single_file_state["metadata"])):
                print(f" - Skipped: out of bounds")
                continue
            if neighbor_idx in added_ids:
                print(f" - Skipped: already added")
                continue

            meta = single_file_state["metadata"][neighbor_idx]
            transcript = meta.get("transcript_with_speaker_labels")

            if not transcript or transcript.strip() == "":
                print(f" - Skipped: empty transcript")
                continue

            print(f" - ✅ Adding segment {neighbor_idx} with transcript len={len(transcript)}")


            speaker = meta.get("speaker", "unknown")
            title = meta.get("podcast_title", "Unknown")
            start = meta.get("start_time", 0)
            end = meta.get("end_time", 0)
            duration = end - start
            role = "relevant match" if neighbor_idx == idx else "context"
            seg_num = len(segment_contexts) + 1

            prompt += (
                f"Segment {seg_num} ({role}):\n"
                f"[{start:.2f}s – {end:.2f}s] (duration: {duration:.1f}s)\n"
                f"{transcript}\n"
                f"in the episode: {title}\n\n"
            )

            segment_contexts.append((speaker, transcript))
            added_ids.add(neighbor_idx)

            snippet = single_file_state["audio_tensor"][neighbor_idx]
            snippet_np = snippet.cpu().numpy()
            audio_html = (
                f"<div><b>{title}:</b><br>"
                f"<b>Segment {seg_num} ({role})</b><br>"
                f"{audio_to_html(snippet_np, single_file_state['sample_rate'])}<br></div><br>"
            )
            audio_segments_to_display.append(audio_html)

    print(f"📝 Prompt build took {time.perf_counter() - prompt_start:.2f}s")

    # LLM Call
    openai_start = time.perf_counter()

    # Build a dynamic mapping block
    mapping_block = {}

    for result in top_results:
        title = result.get("podcast_title", "Unknown Podcast")
        if title not in mapping_block:
            # Pull mapping from your global per_podcast_speaker_mapping
            if title in per_podcast_speaker_mapping:
                mapping_block[title] = per_podcast_speaker_mapping[title]
            else:
                mapping_block = {
                    "The 7 Anime That Every Fan NEEDS To Watch Trash Taste #172": {
                        "speaker_0": "Joey",
                        "speaker_1": "Connor",
                        "speaker_3": "Garnt"
                    }
                }

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"""You are a helpful assistant that uses only the provided transcription to answer the user's query. 
                    Please cite the *Segment number* (e.g., "Segment 2") and answer in your own words. 
                    Do not use timestamps. Speaker labels like "speaker_0" appear in the transcript. 
                    If the context is there, please use the names of the people talking instead of speaker_0, etc.

                    Here are the mappings of speaker labels to real speaker names:
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
    print(f"💬 OpenAI call took {time.perf_counter() - openai_start:.2f}s")

    answer = response.choices[0].message.content
    print(f"🧠 Answer generated. Length: {len(answer)} chars")

    retrieved_segments = []
    for r in top_results:
        segment_id = r["segment_id"]
        if 0 <= segment_id < len(single_file_state["metadata"]):
            meta = single_file_state["metadata"][segment_id]
            speaker = meta.get("speaker", "unknown")
            transcript = meta.get("transcript_with_speaker_labels")

    total_time = time.perf_counter() - total_start
    print(f"✅ Total query_rag() time: {total_time:.2f}s")

    # print("\n=== AUDIO SEGMENTS TO DISPLAY ===")
    # if not audio_segments_to_display:
    #     print("⚠️ No audio segments generated!")
    # else:
    #     for idx, html in enumerate(audio_segments_to_display):
    #         short_preview = html.replace("\n", "").replace(" ", "")[:100]  # squish and truncate
    #         print(f"[{idx}] {short_preview}...")
    #     print(f"✅ Total segments: {len(audio_segments_to_display)}")

    return answer, "".join(audio_segments_to_display), segment_contexts, retrieved_segments

def query_rag_for_ui(query_audio, lambda_val):
    answer, audio_html, segment_contexts, retrieved_segments = query_rag(query_audio, lambda_val)
    return answer, audio_html, segment_contexts  # only return 3 for Gradio

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
            title = meta.get("podcast_title", f"Untitled_{i}")
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

    with gr.Tab("📊 Run Evaluation"):
        eval_btn = gr.Button("Run Evaluation on All Queries")
        eval_status = gr.Textbox(label="Evaluation Status")
        eval_btn.click(fn=run_evaluation, inputs=None, outputs=eval_status)

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
        query_btn.click(query_rag_for_ui, [query_input, lambda_slider], [answer_output, audio_output, segment_contexts_state])

import sys

if __name__ == "__main__":
    share_flag = "--share" in sys.argv
    demo.launch(server_port=7861, share=share_flag)
