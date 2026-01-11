"""
Scalable Meeting Transcriber

A production-ready pipeline for transcribing long audio recordings into 
timestamped text transcripts using Google's Gemini API.

Pipeline: Chunking → Transcription → Consolidation

Features:
- Smart chunking with configurable duration
- Speed optimization (2x default) to reduce API costs
- Automatic timestamp correction to original audio timing
- Parallel processing with rate limiting
- Fault tolerance with retry logic
- Resume support for interrupted jobs

Usage:
    python audio_transcriber.py meeting.mp3
    python audio_transcriber.py *.mp3 --chunk-duration 300 --speed 1.5
    python audio_transcriber.py recording.mp3 --api-key YOUR_KEY --workers 5

Environment Variables:
    GEMINI_API_KEY: Your Google Gemini API key

Author: https://github.com/myExperimentsWithTruth
License: MIT
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChunkConfig:
    """Configuration for audio chunking.
    
    Attributes:
        duration_sec: Duration of each chunk in seconds (default 600 = 10 minutes)
        speed_factor: Playback speed multiplier (default 2.0)
        min_chunk_sec: Minimum chunk duration to create (default 10 seconds)
        output_dir: Directory to store chunk files
    """
    duration_sec: int = 600
    speed_factor: float = 2.0
    min_chunk_sec: int = 10
    output_dir: str = "chunks"


@dataclass
class TranscribeConfig:
    """Configuration for transcription.
    
    Attributes:
        api_key: Gemini API key
        model: Gemini model identifier
        max_workers: Number of parallel transcription workers
        max_retries: Maximum retry attempts for failed API calls
        speed_factor: Speed factor applied to audio chunks (for timestamp correction)
    """
    api_key: str = ""
    model: str = "models/gemini-3-pro-preview"
    max_workers: int = 3
    max_retries: int = 3
    speed_factor: float = 2.0


@dataclass
class PipelineConfig:
    """Main configuration combining all settings.
    
    Attributes:
        chunk: Chunking configuration
        transcribe: Transcription configuration
        transcripts_dir: Directory to store transcript files
    """
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    transcripts_dir: str = "transcripts"


@dataclass
class ChunkInfo:
    """Information about a created audio chunk.
    
    Attributes:
        path: Path to the chunk file
        source_file: Original MP3 filename
        chunk_number: 0-indexed chunk number
        original_start_sec: Start time in original audio (before speed change)
        duration_sec: Duration in original audio (before speed change)
    """
    path: str
    source_file: str
    chunk_number: int
    original_start_sec: float
    duration_sec: float


@dataclass
class TranscriptResult:
    """Result of transcribing a chunk.
    
    Attributes:
        chunk_info: Information about the source chunk
        text: Transcribed text with corrected timestamps
        success: Whether transcription succeeded
        error: Error message if failed
    """
    chunk_info: ChunkInfo
    text: str
    success: bool
    error: Optional[str] = None


# Default API key - MUST be provided via --api-key flag or GEMINI_API_KEY environment variable
DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def get_ffmpeg_path() -> str:
    """Get the path to ffmpeg executable from imageio-ffmpeg."""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def get_audio_duration(mp3_path: str) -> float:
    """Get the duration of an MP3 file in seconds using ffmpeg.
    
    Args:
        mp3_path: Path to the MP3 file
        
    Returns:
        Duration in seconds
        
    Raises:
        FileNotFoundError: If the MP3 file doesn't exist
        RuntimeError: If ffmpeg fails to parse the file
    """
    import subprocess
    import re
    
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"MP3 file not found: {mp3_path}")
    
    ffmpeg = get_ffmpeg_path()
    
    # Run ffmpeg to get file info (outputs to stderr)
    result = subprocess.run(
        [ffmpeg, "-i", mp3_path, "-f", "null", "-"],
        capture_output=True,
        text=True
    )
    
    # Parse duration from stderr output
    # Format: Duration: HH:MM:SS.ms
    duration_pattern = r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)"
    match = re.search(duration_pattern, result.stderr)
    
    if not match:
        raise RuntimeError(f"Could not parse duration from ffmpeg output for: {mp3_path}")
    
    hours, minutes, seconds, centiseconds = match.groups()
    total_seconds = (
        int(hours) * 3600 +
        int(minutes) * 60 +
        int(seconds) +
        int(centiseconds) / 100
    )
    
    return total_seconds


def get_chunk_filename(source_file: str, chunk_number: int) -> str:
    """Generate chunk filename following naming convention.
    
    Args:
        source_file: Original MP3 filename (with or without path)
        chunk_number: 0-indexed chunk number
        
    Returns:
        Chunk filename in format {original_filename}_chunk_{number:03d}.mp3
    """
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    return f"{base_name}_chunk_{chunk_number:03d}.mp3"


def create_chunk(
    mp3_path: str,
    chunk_number: int,
    start_sec: float,
    duration_sec: float,
    config: ChunkConfig
) -> ChunkInfo:
    """Create a single audio chunk from an MP3 file.
    
    Args:
        mp3_path: Path to the source MP3 file
        chunk_number: 0-indexed chunk number
        start_sec: Start time in the original audio (seconds)
        duration_sec: Duration to extract from original audio (seconds)
        config: Chunking configuration
        
    Returns:
        ChunkInfo with details about the created chunk
        
    Raises:
        RuntimeError: If ffmpeg fails to create the chunk
    """
    import subprocess
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Generate output filename
    chunk_filename = get_chunk_filename(mp3_path, chunk_number)
    output_path = os.path.join(config.output_dir, chunk_filename)
    
    ffmpeg = get_ffmpeg_path()
    
    # Build ffmpeg command with atempo filter for speed adjustment
    # atempo filter only accepts values between 0.5 and 2.0
    # For higher speeds, chain multiple atempo filters
    speed = config.speed_factor
    atempo_filters = []
    while speed > 2.0:
        atempo_filters.append("atempo=2.0")
        speed /= 2.0
    if speed > 0.5:
        atempo_filters.append(f"atempo={speed}")
    
    filter_str = ",".join(atempo_filters) if atempo_filters else "atempo=1.0"
    
    # Calculate output duration: input_duration / speed_factor
    output_duration = duration_sec / config.speed_factor
    
    # Place -ss BEFORE -i for fast seeking that works correctly with atempo filter
    # -t after -ss (before -i) specifies output duration
    cmd = [
        ffmpeg,
        "-y",  # Overwrite output file
        "-ss", str(start_sec),  # Seek BEFORE -i for correct behavior with filters
        "-i", mp3_path,
        "-t", str(output_duration),  # Output duration
        "-af", filter_str,
        "-acodec", "libmp3lame",
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to create chunk: {result.stderr}")
    
    return ChunkInfo(
        path=output_path,
        source_file=mp3_path,
        chunk_number=chunk_number,
        original_start_sec=start_sec,
        duration_sec=duration_sec
    )


def calculate_chunk_boundaries(
    total_duration: float,
    chunk_duration: int,
    min_chunk_duration: int
) -> List[tuple]:
    """Calculate chunk start times and durations.
    
    Args:
        total_duration: Total audio duration in seconds
        chunk_duration: Target chunk duration in seconds
        min_chunk_duration: Minimum chunk duration (skip if smaller)
        
    Returns:
        List of (start_sec, duration_sec) tuples for each chunk
    """
    boundaries = []
    start = 0.0
    
    while start < total_duration:
        remaining = total_duration - start
        duration = min(chunk_duration, remaining)
        
        # Skip if chunk would be too short
        if duration < min_chunk_duration:
            break
            
        boundaries.append((start, duration))
        start += chunk_duration
    
    return boundaries


def create_all_chunks(mp3_paths: List[str], config: ChunkConfig) -> List[ChunkInfo]:
    """Create chunks for multiple MP3 files in parallel.
    
    Args:
        mp3_paths: List of paths to MP3 files
        config: Chunking configuration
        
    Returns:
        List of ChunkInfo for all created chunks
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    all_chunks = []
    tasks = []
    
    # Prepare all chunking tasks
    for mp3_path in mp3_paths:
        try:
            duration = get_audio_duration(mp3_path)
            boundaries = calculate_chunk_boundaries(
                duration,
                config.duration_sec,
                config.min_chunk_sec
            )
            
            for chunk_num, (start, dur) in enumerate(boundaries):
                tasks.append((mp3_path, chunk_num, start, dur))
                
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Skipping {mp3_path}: {e}")
            continue
    
    # Process chunks in parallel
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(create_chunk, mp3_path, chunk_num, start, dur, config): 
            (mp3_path, chunk_num)
            for mp3_path, chunk_num, start, dur in tasks
        }
        
        for future in as_completed(futures):
            mp3_path, chunk_num = futures[future]
            try:
                chunk_info = future.result()
                all_chunks.append(chunk_info)
                print(f"Created: {chunk_info.path}")
            except Exception as e:
                print(f"Error creating chunk {chunk_num} for {mp3_path}: {e}")
    
    # Sort by source file and chunk number for consistent ordering
    all_chunks.sort(key=lambda c: (c.source_file, c.chunk_number))
    
    return all_chunks


# =============================================================================
# Timestamp Utilities
# =============================================================================

def correct_timestamp(
    timestamp_sec: float,
    speed_factor: float,
    chunk_offset_sec: float
) -> float:
    """Correct a timestamp from sped-up audio to original audio timing.
    
    When audio is sped up by a factor S, timestamps in the transcription
    correspond to the sped-up version. To map back to original audio:
    1. Multiply by speed_factor (since sped-up audio plays faster)
    2. Add the chunk's offset in the original audio
    
    Formula: corrected = (timestamp_sec * speed_factor) + chunk_offset_sec
    
    Args:
        timestamp_sec: Timestamp from the transcription (in sped-up audio)
        speed_factor: Speed multiplier applied to the audio (e.g., 2.0 for 2x)
        chunk_offset_sec: Start time of this chunk in the original audio
        
    Returns:
        Corrected timestamp in seconds, mapped to original audio timing
        
    Example:
        If audio was sped up 2x and chunk starts at 600s in original:
        - A timestamp of 30s in transcription → (30 * 2) + 600 = 660s in original
    """
    return (timestamp_sec * speed_factor) + chunk_offset_sec


def format_timestamp(seconds: float) -> str:
    """Format a timestamp in seconds to human-readable format.
    
    Uses [MM:SS] format for times under 1 hour, [HH:MM:SS] for times >= 1 hour.
    
    Args:
        seconds: Time value in seconds (can be float)
        
    Returns:
        Formatted timestamp string with brackets
        - [MM:SS] for times < 3600 seconds (1 hour)
        - [HH:MM:SS] for times >= 3600 seconds
        
    Examples:
        format_timestamp(65.5) → "[01:05]"
        format_timestamp(3661) → "[01:01:01]"
        format_timestamp(0) → "[00:00]"
    """
    # Convert to integer seconds (truncate fractional part)
    total_seconds = int(seconds)
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if total_seconds < 3600:
        # Under 1 hour: [MM:SS]
        return f"[{minutes:02d}:{secs:02d}]"
    else:
        # 1 hour or more: [HH:MM:SS]
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def create_default_config(api_key: Optional[str] = None) -> PipelineConfig:
    """Create a default pipeline configuration.
    
    Args:
        api_key: Optional API key override. Uses DEFAULT_API_KEY if not provided.
    
    Returns:
        PipelineConfig with default settings
    """
    return PipelineConfig(
        chunk=ChunkConfig(),
        transcribe=TranscribeConfig(api_key=api_key or DEFAULT_API_KEY),
    )


# =============================================================================
# Transcriber Component
# =============================================================================

def transcribe_chunk(chunk_info: ChunkInfo, config: TranscribeConfig) -> TranscriptResult:
    """Transcribe a single audio chunk using Gemini API.
    
    Uploads the audio file to Gemini, sends a prompt for Hinglish transcription,
    and returns the transcribed text with timestamps corrected to original audio timing.
    
    Implements retry logic with exponential backoff for transient API errors
    (429 rate limit, 500 server errors).
    
    Args:
        chunk_info: Information about the chunk to transcribe
        config: Transcription configuration with API key and model
        
    Returns:
        TranscriptResult with transcribed text and success status
        
    Requirements: 2.1, 2.2, 2.4
    """
    import google.generativeai as genai
    import re
    import time
    
    # Configure the Gemini API
    genai.configure(api_key=config.api_key)
    
    audio_file = None
    last_error = None
    
    for attempt in range(config.max_retries):
        try:
            # Upload the audio file (only on first attempt or if not uploaded)
            if audio_file is None:
                audio_file = genai.upload_file(chunk_info.path)
            
            # Create the model
            model = genai.GenerativeModel(config.model)
            
            # Prompt for Hinglish transcription
            prompt = """Transcribe this audio file. The audio contains Hindi-English (Hinglish) speech.

Instructions:
1. Transcribe exactly what is spoken - do NOT translate
2. Keep Hindi words in Hindi (Devanagari or romanized as spoken)
3. Keep English words in English
4. Include timestamps in [MM:SS] format at natural breaks (every 30-60 seconds or at topic changes)
5. Preserve the original language mix as spoken

Output format:
[MM:SS] Transcribed text here...
[MM:SS] More transcribed text...
"""
            
            # Generate transcription
            response = model.generate_content([prompt, audio_file])
            
            # Get the transcribed text
            transcribed_text = response.text
            
            # Replace all timestamps with corrected versions
            def correct_timestamp_in_text(match):
                timestamp_str = match.group(1)
                parts = timestamp_str.split(':')
                
                if len(parts) == 2:
                    # [MM:SS] format
                    minutes, seconds = int(parts[0]), int(parts[1])
                    timestamp_sec = minutes * 60 + seconds
                elif len(parts) == 3:
                    # [HH:MM:SS] format
                    hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
                    timestamp_sec = hours * 3600 + minutes * 60 + seconds
                else:
                    return match.group(0)  # Return unchanged if format is unexpected
                
                # Correct the timestamp using speed factor from config
                corrected_sec = correct_timestamp(
                    timestamp_sec,
                    config.speed_factor,
                    chunk_info.original_start_sec
                )
                
                return format_timestamp(corrected_sec)
            
            corrected_text = re.sub(
                r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]',
                correct_timestamp_in_text,
                transcribed_text
            )
            
            # Clean up the uploaded file
            try:
                audio_file.delete()
            except Exception:
                pass  # Ignore cleanup errors
            
            return TranscriptResult(
                chunk_info=chunk_info,
                text=corrected_text,
                success=True,
                error=None
            )
            
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Check if this is a retryable error (429 rate limit or 500 server error)
            is_retryable = (
                '429' in str(e) or
                '500' in str(e) or
                'rate' in error_str or
                'quota' in error_str or
                'resource_exhausted' in error_str or
                'internal' in error_str or
                'unavailable' in error_str
            )
            
            if is_retryable and attempt < config.max_retries - 1:
                # Exponential backoff: 2^attempt seconds (1, 2, 4, ...)
                delay = 2 ** attempt
                print(f"Retry {attempt + 1}/{config.max_retries} for {chunk_info.path} after {delay}s: {e}")
                time.sleep(delay)
                continue
            else:
                # Non-retryable error or max retries reached
                break
    
    # Clean up uploaded file on failure
    if audio_file is not None:
        try:
            audio_file.delete()
        except Exception:
            pass
    
    return TranscriptResult(
        chunk_info=chunk_info,
        text="",
        success=False,
        error=str(last_error) if last_error else "Unknown error"
    )


def save_transcript(result: TranscriptResult, output_dir: str) -> Optional[str]:
    """Save a transcript result to a text file.
    
    Args:
        result: TranscriptResult to save
        output_dir: Directory to save transcript files
        
    Returns:
        Path to saved file, or None if save failed
        
    Requirements: 2.5
    """
    if not result.success:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename from chunk path
    chunk_basename = os.path.splitext(os.path.basename(result.chunk_info.path))[0]
    output_path = os.path.join(output_dir, f"{chunk_basename}.txt")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result.text)
        return output_path
    except Exception as e:
        print(f"Error saving transcript to {output_path}: {e}")
        return None


def transcribe_all(
    chunks: List[ChunkInfo],
    config: TranscribeConfig,
    output_dir: str = "transcripts"
) -> List[TranscriptResult]:
    """Transcribe multiple audio chunks in parallel with rate limiting.
    
    Uses ThreadPoolExecutor with configurable number of workers to process
    chunks in parallel while respecting API rate limits.
    
    Args:
        chunks: List of ChunkInfo objects to transcribe
        config: Transcription configuration
        output_dir: Directory to save individual transcript files
        
    Returns:
        List of TranscriptResult for all chunks (both successful and failed)
        
    Requirements: 2.3, 2.5
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = []
    
    # Process chunks in parallel with limited workers
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # Submit all transcription tasks
        futures = {
            executor.submit(transcribe_chunk, chunk, config): chunk
            for chunk in chunks
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                if result.success:
                    # Save individual transcript
                    saved_path = save_transcript(result, output_dir)
                    if saved_path:
                        print(f"Transcribed: {chunk.path} → {saved_path}")
                    else:
                        print(f"Transcribed: {chunk.path} (save failed)")
                else:
                    print(f"Failed: {chunk.path} - {result.error}")
                    
            except Exception as e:
                # Handle unexpected errors
                error_result = TranscriptResult(
                    chunk_info=chunk,
                    text="",
                    success=False,
                    error=str(e)
                )
                results.append(error_result)
                print(f"Error: {chunk.path} - {e}")
    
    # Sort results by source file and chunk number for consistent ordering
    results.sort(key=lambda r: (r.chunk_info.source_file, r.chunk_info.chunk_number))
    
    return results


# =============================================================================
# Consolidator Component
# =============================================================================

def consolidate_transcripts(
    transcripts: List[TranscriptResult],
    output_dir: str = "."
) -> dict:
    """Consolidate chunk transcripts into complete files per source audio.
    
    Merges all chunk transcripts from the same source file into a single
    consolidated transcript, sorted by original start time. Adds chunk
    boundary markers showing the original timestamp where each chunk begins.
    
    Args:
        transcripts: List of TranscriptResult objects to consolidate
        output_dir: Directory to save consolidated transcript files
        
    Returns:
        Dictionary mapping source filenames to output file paths
        
    Requirements: 4.1, 4.2, 4.3
    """
    from collections import defaultdict
    
    # Group transcripts by source file
    by_source: dict = defaultdict(list)
    for result in transcripts:
        if result.success:
            by_source[result.chunk_info.source_file].append(result)
    
    output_paths = {}
    
    for source_file, results in by_source.items():
        # Sort by original start time (Requirements 4.1)
        results.sort(key=lambda r: r.chunk_info.original_start_sec)
        
        # Build consolidated content with chunk boundary markers
        consolidated_parts = []
        
        for result in results:
            chunk_info = result.chunk_info
            
            # Create chunk boundary marker (Requirements 4.2)
            # Shows the original timestamp where this chunk starts
            start_timestamp = format_timestamp(chunk_info.original_start_sec)
            boundary_marker = f"\n--- Chunk {chunk_info.chunk_number} {start_timestamp} ---\n"
            
            consolidated_parts.append(boundary_marker)
            consolidated_parts.append(result.text.strip())
            consolidated_parts.append("\n")
        
        # Generate output filename: {original_filename}_full.txt (Requirements 4.3)
        source_basename = os.path.splitext(os.path.basename(source_file))[0]
        output_filename = f"{source_basename}_full.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Write consolidated transcript
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("".join(consolidated_parts).strip())
                f.write("\n")  # Ensure file ends with newline
            
            output_paths[source_file] = output_path
            print(f"Consolidated: {source_file} → {output_path}")
            
        except Exception as e:
            print(f"Error saving consolidated transcript for {source_file}: {e}")
    
    return output_paths


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args(args: Optional[List[str]] = None):
    """Parse command-line arguments.
    
    Args:
        args: Optional list of arguments (uses sys.argv if None)
        
    Returns:
        Parsed arguments namespace
        
    Requirements: 5.1, 5.2, 5.3, 5.4
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Audio Transcription Pipeline - Convert MP3 files to timestamped transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio1.mp3 audio2.mp3
  %(prog)s *.mp3 --chunk-duration 300 --speed 1.5
  %(prog)s recording.mp3 --api-key YOUR_KEY --workers 5
        """
    )
    
    # Input files (positional, required)
    parser.add_argument(
        "input_files",
        nargs="+",
        metavar="FILE",
        help="MP3 file(s) to transcribe"
    )
    
    # Chunk configuration (Requirements 5.1)
    parser.add_argument(
        "--chunk-duration", "-c",
        type=int,
        default=600,
        metavar="SECONDS",
        help="Duration of each chunk in seconds (default: 600 = 10 minutes)"
    )
    
    # Speed factor (Requirements 5.2)
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=2.0,
        metavar="FACTOR",
        help="Speed factor for audio processing (default: 2.0)"
    )
    
    # API key (Requirements 5.3)
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        metavar="KEY",
        help="Gemini API key (default: uses built-in key)"
    )
    
    # Parallel workers (Requirements 5.4)
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=3,
        metavar="N",
        help="Number of parallel transcription workers (default: 3)"
    )
    
    # Output directories
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="chunks",
        metavar="DIR",
        help="Directory for chunk files (default: chunks)"
    )
    
    parser.add_argument(
        "--transcripts-dir",
        type=str,
        default="transcripts",
        metavar="DIR",
        help="Directory for transcript files (default: transcripts)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=".",
        metavar="DIR",
        help="Directory for consolidated transcripts (default: current directory)"
    )
    
    # Additional options
    parser.add_argument(
        "--min-chunk",
        type=int,
        default=10,
        metavar="SECONDS",
        help="Minimum chunk duration in seconds (default: 10)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/gemini-3-pro-preview",
        metavar="MODEL",
        help="Gemini model to use (default: models/gemini-3-pro-preview)"
    )
    
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Skip chunking step (use existing chunks)"
    )
    
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Skip transcription step (consolidate existing transcripts)"
    )
    
    return parser.parse_args(args)


def run_pipeline(args) -> dict:
    """Run the full transcription pipeline.
    
    Executes the pipeline: Chunker → Transcriber → Consolidator
    with progress reporting and error handling.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with pipeline results:
        - chunks: List of ChunkInfo created
        - transcripts: List of TranscriptResult
        - consolidated: Dict mapping source files to output paths
        - errors: List of error messages
    """
    import glob
    
    results = {
        "chunks": [],
        "transcripts": [],
        "consolidated": {},
        "errors": []
    }
    
    # Expand glob patterns in input files
    input_files = []
    for pattern in args.input_files:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(expanded)
        else:
            # If no glob match, treat as literal filename
            input_files.append(pattern)
    
    # Validate input files exist
    valid_files = []
    for f in input_files:
        if os.path.exists(f):
            valid_files.append(f)
        else:
            error_msg = f"File not found: {f}"
            print(f"Warning: {error_msg}")
            results["errors"].append(error_msg)
    
    if not valid_files:
        print("Error: No valid input files found")
        return results
    
    print(f"\n{'='*60}")
    print("Audio Transcription Pipeline")
    print(f"{'='*60}")
    print(f"Input files: {len(valid_files)}")
    print(f"Chunk duration: {args.chunk_duration}s")
    print(f"Speed factor: {args.speed}x")
    print(f"Workers: {args.workers}")
    print(f"{'='*60}\n")
    
    # Build configuration
    chunk_config = ChunkConfig(
        duration_sec=args.chunk_duration,
        speed_factor=args.speed,
        min_chunk_sec=args.min_chunk,
        output_dir=args.chunks_dir
    )
    
    transcribe_config = TranscribeConfig(
        api_key=args.api_key or DEFAULT_API_KEY,
        model=args.model,
        max_workers=args.workers,
        max_retries=3,
        speed_factor=args.speed
    )
    
    # ==========================================================================
    # Step 1: Chunking
    # ==========================================================================
    if not args.skip_chunking:
        print(f"[Step 1/3] Chunking {len(valid_files)} file(s)...")
        print("-" * 40)
        
        try:
            chunks = create_all_chunks(valid_files, chunk_config)
            results["chunks"] = chunks
            print(f"\nCreated {len(chunks)} chunk(s)")
        except Exception as e:
            error_msg = f"Chunking failed: {e}"
            print(f"Error: {error_msg}")
            results["errors"].append(error_msg)
            # Save partial results and continue if possible
            if not results["chunks"]:
                print("No chunks created. Aborting pipeline.")
                return results
    else:
        print("[Step 1/3] Skipping chunking (using existing chunks)...")
        # Load existing chunks from chunks directory
        chunks = load_existing_chunks(valid_files, chunk_config)
        results["chunks"] = chunks
        print(f"Found {len(chunks)} existing chunk(s)")
    
    if not results["chunks"]:
        print("No chunks to process. Aborting pipeline.")
        return results
    
    print()
    
    # ==========================================================================
    # Step 2: Transcription
    # ==========================================================================
    if not args.skip_transcription:
        print(f"[Step 2/3] Transcribing {len(results['chunks'])} chunk(s)...")
        print("-" * 40)
        
        try:
            transcripts = transcribe_all(
                results["chunks"],
                transcribe_config,
                args.transcripts_dir
            )
            results["transcripts"] = transcripts
            
            # Count successes and failures
            successes = sum(1 for t in transcripts if t.success)
            failures = len(transcripts) - successes
            print(f"\nTranscribed: {successes} succeeded, {failures} failed")
            
            # Record errors
            for t in transcripts:
                if not t.success:
                    results["errors"].append(f"Transcription failed for {t.chunk_info.path}: {t.error}")
                    
        except Exception as e:
            error_msg = f"Transcription failed: {e}"
            print(f"Error: {error_msg}")
            results["errors"].append(error_msg)
    else:
        print("[Step 2/3] Skipping transcription (using existing transcripts)...")
        # Load existing transcripts
        transcripts = load_existing_transcripts(results["chunks"], args.transcripts_dir)
        results["transcripts"] = transcripts
        print(f"Found {len(transcripts)} existing transcript(s)")
    
    if not results["transcripts"]:
        print("No transcripts to consolidate. Aborting pipeline.")
        return results
    
    print()
    
    # ==========================================================================
    # Step 3: Consolidation
    # ==========================================================================
    print("[Step 3/3] Consolidating transcripts...")
    print("-" * 40)
    
    try:
        consolidated = consolidate_transcripts(
            results["transcripts"],
            args.output_dir
        )
        results["consolidated"] = consolidated
        print(f"\nConsolidated {len(consolidated)} file(s)")
    except Exception as e:
        error_msg = f"Consolidation failed: {e}"
        print(f"Error: {error_msg}")
        results["errors"].append(error_msg)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print()
    print(f"{'='*60}")
    print("Pipeline Complete")
    print(f"{'='*60}")
    print(f"Chunks created: {len(results['chunks'])}")
    print(f"Transcripts: {sum(1 for t in results['transcripts'] if t.success)}/{len(results['transcripts'])}")
    print(f"Consolidated files: {len(results['consolidated'])}")
    
    if results["errors"]:
        print(f"\nWarnings/Errors: {len(results['errors'])}")
        for error in results["errors"][:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(results["errors"]) > 5:
            print(f"  ... and {len(results['errors']) - 5} more")
    
    print(f"{'='*60}\n")
    
    return results


def load_existing_chunks(source_files: List[str], config: ChunkConfig) -> List[ChunkInfo]:
    """Load existing chunk files for given source files.
    
    Args:
        source_files: List of original MP3 file paths
        config: Chunk configuration with output directory
        
    Returns:
        List of ChunkInfo for existing chunks
    """
    import glob
    import re
    
    chunks = []
    
    for source_file in source_files:
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        pattern = os.path.join(config.output_dir, f"{base_name}_chunk_*.mp3")
        
        for chunk_path in sorted(glob.glob(pattern)):
            # Extract chunk number from filename
            match = re.search(r'_chunk_(\d+)\.mp3$', chunk_path)
            if match:
                chunk_num = int(match.group(1))
                
                # Calculate original start time based on chunk number
                original_start = chunk_num * config.duration_sec
                
                chunks.append(ChunkInfo(
                    path=chunk_path,
                    source_file=source_file,
                    chunk_number=chunk_num,
                    original_start_sec=original_start,
                    duration_sec=config.duration_sec
                ))
    
    return chunks


def load_existing_transcripts(chunks: List[ChunkInfo], transcripts_dir: str) -> List[TranscriptResult]:
    """Load existing transcript files for given chunks.
    
    Args:
        chunks: List of ChunkInfo to find transcripts for
        transcripts_dir: Directory containing transcript files
        
    Returns:
        List of TranscriptResult for existing transcripts
    """
    results = []
    
    for chunk in chunks:
        chunk_basename = os.path.splitext(os.path.basename(chunk.path))[0]
        transcript_path = os.path.join(transcripts_dir, f"{chunk_basename}.txt")
        
        if os.path.exists(transcript_path):
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                results.append(TranscriptResult(
                    chunk_info=chunk,
                    text=text,
                    success=True,
                    error=None
                ))
            except Exception as e:
                results.append(TranscriptResult(
                    chunk_info=chunk,
                    text="",
                    success=False,
                    error=str(e)
                ))
    
    return results


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        args: Optional list of arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success, 1 for errors)
    """
    parsed_args = parse_args(args)
    results = run_pipeline(parsed_args)
    
    # Return non-zero exit code if there were errors
    if results["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
