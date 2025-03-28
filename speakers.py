from faster_whisper import WhisperModel

def get_speaker_activity_in_window(diarization_segments, window_start, window_end):
    """
    Returns speaker activity within a time window as:
    {
        "speaker_0": [{"start": 100.1, "end": 102.8}, ...],
        "speaker_1": [...],
    }
    """
    speakers_in_window = {}

    for seg in diarization_segments:
        start = seg["start"]
        end = seg["end"]
        speaker = seg["speaker"]

        if end < window_start or start > window_end:
            continue

        clipped_start = max(start, window_start)
        clipped_end = min(end, window_end)
        duration = clipped_end - clipped_start
        if duration <= 0:
            continue

        if speaker not in speakers_in_window:
            speakers_in_window[speaker] = []

        speakers_in_window[speaker].append({
            "start": round(clipped_start, 3),
            "end": round(clipped_end, 3)
        })

    return speakers_in_window

def assign_speaker_activity_to_vad_chunk(vad_segment, vad_meta, diarization_segments, custom_speaker_mapping):
    """
    Attaches speaker activity information to a single VAD chunk.
    
    Args:
        vad_segment (torch.Tensor): The original VAD audio segment.
        vad_meta (dict): Metadata for the VAD chunk. Must include "start_time" and "end_time".
        diarization_segments (list of dict): Diarization results with keys "start", "end", "speaker".
        custom_speaker_mapping (dict): Mapping to convert diarization speaker labels to your desired labels.
        
    Returns:
        Tuple: (vad_segment, updated_vad_meta) where updated_vad_meta has a new key "speaker_activity"
               containing a dict with each speaker and their active time intervals within the chunk.
    """
    activity_raw = get_speaker_activity_in_window(
        diarization_segments,
        vad_meta["start_time"],
        vad_meta["end_time"]
    )
    
    # Map diarization speaker labels using your custom mapping
    vad_meta["speaker_activity"] = {
        custom_speaker_mapping.get(speaker, speaker): times
        for speaker, times in activity_raw.items()
    }
    
    return vad_segment, vad_meta

def assign_speaker_activity_to_vad_chunks(vad_segments, vad_metadata, diarization_segments, custom_speaker_mapping):
    """
    Processes a list of VAD chunks, attaching speaker activity info to each chunk's metadata.
    
    Args:
        vad_segments (List[torch.Tensor]): List of VAD audio segments.
        vad_metadata (List[dict]): List of metadata dicts (each with "start_time" and "end_time").
        diarization_segments (list of dict): Diarization results.
        custom_speaker_mapping (dict): Mapping from diarization labels to custom labels.
    
    Returns:
        Tuple: (updated_vad_segments, updated_vad_metadata)
    """
    updated_vad_segments = []
    updated_vad_metadata = []
    
    for seg, meta in zip(vad_segments, vad_metadata):
        updated_seg, updated_meta = assign_speaker_activity_to_vad_chunk(
            seg, meta, diarization_segments, custom_speaker_mapping
        )
        updated_vad_segments.append(updated_seg)
        updated_vad_metadata.append(updated_meta)
    
    return updated_vad_segments, updated_vad_metadata

whisper_model = WhisperModel("tiny", device="cuda", compute_type="float16")

def transcribe_with_fasterwhisper_and_assign_speaker_labels(audio_data, sample_rate, vad_meta, language="en"):
    """
    Transcribes audio using faster-whisper, including timestamps for each segment,
    and assigns a raw speaker label (e.g., "speaker_0", "speaker_1", etc.) to each segment
    based on the overlap with the speaker activity from vad_meta.
    
    Args:
        audio_data (torch.Tensor or np.array): The audio data for the VAD chunk.
        sample_rate (int): The sample rate.
        vad_meta (dict): Metadata for the VAD chunk. Must include:
                         - "start_time": float, absolute start time of this chunk.
                         - "speaker_activity": dict mapping raw speaker labels (e.g. "speaker_0")
                           to a list of dicts with "start" and "end" (absolute times).
        language (str): Language for transcription.
    
    Returns:
        transcript (str): Lines formatted as:
            [abs_start s - abs_end s] speaker_X: transcription text
    """
    import tempfile, os, numpy as np, soundfile as sf

    # Convert torch.Tensor to numpy if needed.
    if hasattr(audio_data, "cpu"):
        audio_data = audio_data.cpu().numpy()
    
    if audio_data.size == 0:
        return ""
    
    # Squeeze extra dimensions & ensure correct shape.
    audio_data = np.squeeze(audio_data)
    if audio_data.ndim == 2:
        audio_data = audio_data.T

    if np.issubdtype(audio_data.dtype, np.floating):
        audio_data = np.clip(audio_data, -1, 1).astype(np.float32)
    else:
        audio_data = audio_data.astype(np.float32)
    
    # Write audio to a temporary WAV file.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_filename = tmp.name
    try:
        sf.write(tmp_filename, audio_data, sample_rate, format="WAV", subtype="PCM_16")
        # Transcribe using faster-whisper.
        segments, info = whisper_model.transcribe(tmp_filename, beam_size=5, language=language)
    finally:
        os.remove(tmp_filename)
    
    transcript_lines = []
    chunk_offset = vad_meta["start_time"]  # absolute start time of this VAD chunk
    # Use the raw speaker activity (with labels like "speaker_0", etc.)
    speaker_activity = vad_meta.get("speaker_activity", {})
    # Convert absolute intervals into relative intervals (relative to the VAD chunk)
    rel_speaker_activity = {}
    for speaker, intervals in speaker_activity.items():
        rel_intervals = []
        for interval in intervals:
            rel_intervals.append({
                "start": interval["start"] - chunk_offset,
                "end": interval["end"] - chunk_offset
            })
        rel_speaker_activity[speaker] = rel_intervals

    def interval_overlap(a, b, c, d):
        """Returns the overlap length between intervals [a, b] and [c, d]."""
        return max(0, min(b, d) - max(a, c))
    
    # For each transcription segment from Whisper, find the speaker with maximum overlap.
    for seg in segments:
        seg_start = seg.start  # relative start time within the chunk
        seg_end = seg.end      # relative end time
        abs_start = seg_start + chunk_offset
        abs_end = seg_end + chunk_offset
        
        best_speaker = "unknown"
        best_overlap = 0.0
        for speaker, intervals in rel_speaker_activity.items():
            total_overlap = 0.0
            for interval in intervals:
                total_overlap += interval_overlap(seg_start, seg_end, interval["start"], interval["end"])
            if total_overlap > best_overlap:
                best_overlap = total_overlap
                best_speaker = speaker
        
        line = f"[{abs_start:.1f}s - {abs_end:.1f}s] {best_speaker}: {seg.text.strip()}"
        transcript_lines.append(line)
    
    transcript = "\n".join(transcript_lines)
    return transcript

def format_speaker_timeline(speaker_activity):
    """
    Formats the speaker activity dictionary into a string timeline.
    
    Args:
        speaker_activity (dict): Dictionary mapping speaker labels to list of time intervals.
    
    Returns:
        timeline_str (str): A formatted string showing each speaker's active time intervals.
    """
    timeline_str = "Speaker Timeline:\n"
    for speaker, intervals in speaker_activity.items():
        intervals_str = ", ".join([f"{seg['start']}s–{seg['end']}s" for seg in intervals])
        timeline_str += f"{speaker}: {intervals_str}\n"
    return timeline_str
