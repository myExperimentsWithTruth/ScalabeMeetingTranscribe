#!/usr/bin/env python3
"""
Fix failed transcripts by re-transcribing problematic chunks and patching final files.
"""

import os
import re
from audio_transcriber import (
    ChunkInfo, TranscriptResult, TranscribeConfig, 
    transcribe_chunk, save_transcript, format_timestamp,
    DEFAULT_API_KEY
)

# Configuration
CHUNKS_DIR = "chunks"
TRANSCRIPTS_DIR = "transcripts"
FINAL_DIR = "final_transcripts"
CHUNK_DURATION = 600  # 10 minutes
SPEED_FACTOR = 2.0

# Marker text that indicates a failed transcription
FAILED_MARKERS = [
    "As there is no audio file attached",
    "sample transcription",
    "I cannot transcribe a specific recording"
]


def find_failed_transcripts():
    """Find all transcript files that contain failure markers."""
    failed = []
    
    for filename in os.listdir(TRANSCRIPTS_DIR):
        if not filename.endswith('.txt'):
            continue
            
        filepath = os.path.join(TRANSCRIPTS_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for failure markers
        for marker in FAILED_MARKERS:
            if marker.lower() in content.lower():
                failed.append(filename)
                break
    
    return failed


def get_chunk_info_from_filename(transcript_filename: str) -> ChunkInfo:
    """Extract chunk info from transcript filename."""
    # transcript filename: D1-S1_chunk_006.txt
    # chunk filename: D1-S1_chunk_006.mp3
    base = transcript_filename.replace('.txt', '')
    chunk_path = os.path.join(CHUNKS_DIR, f"{base}.mp3")
    
    # Extract chunk number
    match = re.search(r'_chunk_(\d+)', base)
    chunk_num = int(match.group(1)) if match else 0
    
    # Extract source file name (e.g., D1-S1)
    source_base = re.sub(r'_chunk_\d+', '', base)
    source_file = f"{source_base}.mp3"
    
    # Calculate original start time
    original_start = chunk_num * CHUNK_DURATION
    
    return ChunkInfo(
        path=chunk_path,
        source_file=source_file,
        chunk_number=chunk_num,
        original_start_sec=original_start,
        duration_sec=CHUNK_DURATION
    )


def patch_final_transcript(source_base: str, chunk_num: int, new_text: str):
    """Patch the final transcript file with the new chunk text."""
    final_path = os.path.join(FINAL_DIR, f"{source_base}_full.txt")
    
    if not os.path.exists(final_path):
        print(f"Warning: Final transcript not found: {final_path}")
        return False
    
    with open(final_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the chunk boundary markers
    # Format: --- Chunk N [MM:SS] ---
    chunk_pattern = rf'(--- Chunk {chunk_num} \[[\d:]+\] ---\n)(.*?)((?=\n--- Chunk \d+)|$)'
    
    def replace_chunk(match):
        boundary = match.group(1)
        # Keep the boundary marker, replace the content
        return f"{boundary}{new_text.strip()}\n"
    
    new_content = re.sub(chunk_pattern, replace_chunk, content, flags=re.DOTALL)
    
    if new_content == content:
        print(f"Warning: Could not find chunk {chunk_num} boundary in {final_path}")
        return False
    
    with open(final_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Patched: {final_path}")
    return True


def main():
    print("Scanning for failed transcripts...")
    failed = find_failed_transcripts()
    
    if not failed:
        print("No failed transcripts found!")
        return
    
    print(f"\nFound {len(failed)} failed transcript(s):")
    for f in failed:
        print(f"  - {f}")
    
    # Configure transcription
    config = TranscribeConfig(
        api_key=DEFAULT_API_KEY,
        model="models/gemini-3-pro-preview",
        max_workers=1,
        max_retries=3,
        speed_factor=SPEED_FACTOR
    )
    
    print(f"\nRe-transcribing with model: {config.model}")
    print("-" * 50)
    
    for transcript_file in failed:
        print(f"\nProcessing: {transcript_file}")
        
        # Get chunk info
        chunk_info = get_chunk_info_from_filename(transcript_file)
        
        # Verify chunk file exists
        if not os.path.exists(chunk_info.path):
            print(f"  Error: Chunk file not found: {chunk_info.path}")
            continue
        
        print(f"  Chunk: {chunk_info.path}")
        print(f"  Original start: {format_timestamp(chunk_info.original_start_sec)}")
        
        # Re-transcribe
        print("  Transcribing...")
        result = transcribe_chunk(chunk_info, config)
        
        if result.success:
            # Save the new transcript
            saved_path = save_transcript(result, TRANSCRIPTS_DIR)
            print(f"  Saved: {saved_path}")
            
            # Patch the final file
            source_base = re.sub(r'_chunk_\d+', '', transcript_file.replace('.txt', ''))
            patch_final_transcript(source_base, chunk_info.chunk_number, result.text)
        else:
            print(f"  Failed: {result.error}")
    
    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
