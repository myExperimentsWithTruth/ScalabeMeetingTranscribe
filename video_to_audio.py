#!/usr/bin/env python3
"""
Video to Audio Converter

A simple utility to extract audio from video files (e.g., Teams meeting recordings).
Uses ffmpeg bundled via imageio-ffmpeg.

Usage:
    python video_to_audio.py video.mp4
    python video_to_audio.py video.mp4 --output audio.mp3
    python video_to_audio.py *.mp4 --output-dir audio_files/
"""

import os
import sys
import argparse
import subprocess
from typing import List, Optional


def get_ffmpeg_path() -> str:
    """Get the path to ffmpeg executable from imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        # Fall back to system ffmpeg
        return "ffmpeg"


def convert_video_to_audio(
    video_path: str,
    output_path: Optional[str] = None,
    audio_quality: int = 4
) -> str:
    """Convert a video file to MP3 audio.
    
    Args:
        video_path: Path to the input video file
        output_path: Optional output path. If None, uses same name with .mp3 extension
        audio_quality: MP3 quality (0-9, lower is better, default 4 = ~165 kbps)
        
    Returns:
        Path to the created audio file
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If ffmpeg conversion fails
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(video_path)[0]
        output_path = f"{base_name}.mp3"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    ffmpeg = get_ffmpeg_path()
    
    # Build ffmpeg command
    # -vn: no video
    # -acodec libmp3lame: use MP3 encoder
    # -q:a: audio quality (VBR)
    cmd = [
        ffmpeg,
        "-y",  # Overwrite output
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "libmp3lame",
        "-q:a", str(audio_quality),
        output_path
    ]
    
    print(f"Converting: {video_path} → {output_path}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
    
    # Get file sizes for reporting
    video_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
    audio_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    print(f"  Video: {video_size:.1f} MB → Audio: {audio_size:.1f} MB ({audio_size/video_size*100:.1f}%)")
    
    return output_path


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert video files to MP3 audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s meeting.mp4
  %(prog)s meeting.mp4 --output audio.mp3
  %(prog)s *.mp4 --output-dir audio_files/
        """
    )
    
    parser.add_argument(
        "input_files",
        nargs="+",
        metavar="FILE",
        help="Video file(s) to convert"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (only for single input file)"
    )
    
    parser.add_argument(
        "--output-dir", "-d",
        type=str,
        default=None,
        help="Output directory for converted files"
    )
    
    parser.add_argument(
        "--quality", "-q",
        type=int,
        default=4,
        choices=range(0, 10),
        metavar="0-9",
        help="Audio quality (0=best, 9=worst, default: 4)"
    )
    
    parsed = parser.parse_args(args)
    
    # Validate arguments
    if parsed.output and len(parsed.input_files) > 1:
        print("Error: --output can only be used with a single input file")
        return 1
    
    # Process files
    import glob
    
    input_files = []
    for pattern in parsed.input_files:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(expanded)
        else:
            input_files.append(pattern)
    
    success_count = 0
    error_count = 0
    
    for video_path in input_files:
        try:
            # Determine output path
            if parsed.output:
                output_path = parsed.output
            elif parsed.output_dir:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(parsed.output_dir, f"{base_name}.mp3")
            else:
                output_path = None
            
            convert_video_to_audio(video_path, output_path, parsed.quality)
            success_count += 1
            
        except Exception as e:
            print(f"Error converting {video_path}: {e}")
            error_count += 1
    
    print(f"\nCompleted: {success_count} succeeded, {error_count} failed")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
