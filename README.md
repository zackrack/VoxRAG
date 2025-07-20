
# VoxRAG: Transcription-Free Retrieval-Augmented Generation for Spoken Question Answering

---

## 🔍 Overview


**Please cite our paper** if you use this codebase or reference VoxRAG in your research:

> *Zackary Rackauckas, Julia Hirschberg. “VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering.” arXiv:2505.17326v1 [cs.IR], May 2025.*

**VoxRAG** is a modular speech-to-speech retrieval-augmented generation (RAG) pipeline that bypasses automatic speech recognition (ASR) to retrieve and reason over semantically relevant podcast audio using **audio embeddings** alone. It performs end-to-end semantic search and QA directly on spoken audio.

This repository includes:
- Silence-aware segmentation
- Speaker diarization
- CLAP audio embedding generation
- L2-normalized FAISS similarity search
- Optional reranking with a cross-encoder
- GPT-4o-based answer generation

## 🛠️ Installation

```bash
git clone https://github.com/your-username/VoxRAG.git
cd VoxRAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Make sure ffmpeg is installed on your system.
---

## ▶️ Running VoxRAG

### To run the interface locally:

```bash
python app.py
```

### To deploy publicly using Gradio:

```bash
python app.py --share
```

## 📁 Project File Structure (Data & Audio Folders)

```
VoxRAG/
├── app.py                        # Main entry point with Gradio + FastAPI server
├── data/                         # Saved models, metadata, and embeddings
│   ├── full_state.pt             # Serialized PyTorch state: embeddings, metadata, audio
│   ├── full_state_autosave.pt    # Autosave backup of the same
│   ├── audio_segments.pt         # Serialized list of audio segment tensors
│   ├── speaker_mappings.json     # Custom per-podcast speaker names
│   ├── segments.jsonl            # Final per-segment metadata for retrieval
│
├── podcasts/                     # Folder containing raw podcast audio
│   ├── episode1.wav              # Any supported format: .wav, .mp3, .webm
│   ├── episode1.json             # (Optional) Metadata sidecar for the episode
│   └── ...
│
├── eval/                         # Evaluation scripts and output
│   ├── queries_audio/            # 50 spoken query .wav files
│   ├── documents_vanilla.csv     # Retrieved text segments per query
│   ├── answers_vanilla.csv       # Answers from GPT-4o
│   ├── timing_vanilla.csv        # Timing for each query evaluation
│
├── embeddings.py                 # CLAP / WavLM embedding functions
├── rerankers.py                  # Optional reranker (e.g. cross-encoder)
├── helpers.py                    # Audio preprocessing, segmentation, transcription
├── speakers.py                   # Diarization and speaker label assignment
├── requirements.txt
├── README.md
```

### 💡 How They're Used

- `podcasts/` is scanned by `index_podcast_folder()`. It expects `.wav`, `.mp3`, or `.webm` podcast files.  
  You can also place optional `.json` metadata next to audio files with the same base name.
  
- `data/` is automatically created to store:
  - All indexed embeddings and audio segments (`full_state.pt`)
  - Segment-level metadata for retrieval (`segments.jsonl`)
  - Final transcript+speaker mappings (`speaker_mappings.json`)

- `eval/` is used from the "Run Evaluation" tab in Gradio. It assumes:
  - Spoken query audio lives in `eval/queries_audio/`
  - Outputs are saved to CSVs (retrieved docs, generated answers, timing).
  - You can find eval scripts in the RAGElo repository here: https://github.com/zetaalphavector/RAGElo

---

## 🧠 Motivation

Retrieval-Augmented Generation (RAG) is typically text-centric. VoxRAG proposes a **fully speech-native** alternative, keeping both queries and documents in the acoustic domain through the retrieval stage. This avoids the pitfalls of early transcription errors, which are especially common in noisy or informal podcast content.

---

## 📦 Features

- **Transcription-Free Retrieval:** Query and document segments are embedded via `CLAP` and compared using cosine similarity in FAISS.
- **Segment Pipeline:**
  - Silence-aware segmentation (via Silero VAD)
  - Speaker diarization (NeMo ClusteringDiarizer)
  - Optional transcription for GPT prompt construction
- **Modular Retrieval:**
  - FAISS cosine search
  - Optional MiniLM-based reranking
- **LLM Answer Generation:**
  - GPT-4o with segment-aware prompting
  - Gradio interface with transcript and audio playback

---

## 🗃️ Dataset

- **Corpus:** 20 episodes from the *Trash Taste* podcast
- **Eval Episode:** 1 representative 2-hour episode segmented into 202 chunks
- **Query Set:**
  - 11 organic questions
  - 50 diverse synthetic spoken queries recorded in studio
- **Audio Segments:** 16 kHz mono WAV files processed offline

---

## 📊 Performance

| Metric            | Score |
|-------------------|-------|
| **Recall@10 (SR)**| 0.60  |
| **nDCG@10 (SR)**  | 0.27  |
| **Relevance**     | 0.84  |
| **Accuracy**      | 0.58  |
| **Completeness**  | 0.56  |
| **Precision**     | 0.46  |

- CLAP embeddings support **coarse semantic alignment**.
- Precision and factual granularity remain key limitations.

---
