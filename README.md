# Scalable Meeting Transcriber

A production-ready pipeline for transcribing long audio recordings (like Teams meetings) into timestamped text transcripts using Google's Gemini API. Designed to handle multi-hour recordings efficiently through intelligent chunking, speed optimization, and parallel processing.

## Problem Statement

Transcribing long meeting recordings (2-4+ hours) presents several challenges:
- **API Limits**: Most transcription APIs have file size or duration limits
- **Cost Optimization**: Processing hours of audio can be expensive
- **Timestamp Accuracy**: Timestamps must map to the original recording, not processed chunks
- **Language Support**: Business meetings often mix languages (e.g., Hindi-English "Hinglish")

## Our Approach

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Video/Audio    │────▶│    Chunker      │────▶│   Transcriber   │────▶│  Consolidator   │
│  (MP4/MP3)      │     │  (10min chunks) │     │  (Gemini API)   │     │  (Full text)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                              │                        │
                              ▼                        ▼
                        Speed up 2x              Parallel processing
                        (41% cost savings)       with rate limiting
```

### Key Features

- **Smart Chunking**: Splits audio into configurable chunks (default: 10 minutes)
- **Speed Optimization**: Speeds up audio (default: 2x) to reduce API costs by ~41%
- **Timestamp Correction**: Automatically corrects timestamps to match original audio timing
- **Parallel Processing**: Processes multiple chunks simultaneously with rate limiting
- **Fault Tolerance**: Retry logic with exponential backoff for API failures
- **Resume Support**: Can skip chunking/transcription steps to resume interrupted jobs
- **Multi-language**: Optimized for Hindi-English (Hinglish) but works with any language

---

## Cost Optimization: Speed Factor Analysis

### How Gemini Audio Pricing Works

Gemini tokenizes audio at a **fixed rate of 32 tokens per second** of audio duration ([source](https://ai.google.dev/gemini-api/docs/audio)). This means:
- 1 minute of audio = 1,920 tokens
- 1 hour of audio = 115,200 tokens
- **The file duration determines cost, not the content**

By speeding up audio, we reduce the file duration and thus the input tokens sent to the API.

### Speed Factor Testing

We evaluated multiple speed factors to find the optimal balance between cost savings and transcription quality:

| Speed | Audio Duration | Input Tokens | Quality Impact | Recommendation |
|-------|---------------|--------------|----------------|----------------|
| 1x (original) | 100% | 100% | Baseline | No savings |
| **2x** | 50% | 50% | Excellent - Gemini handles well | **Recommended** |
| 3x | 33% | 33% | Some words compressed | Quality degradation |
| 4x | 25% | 25% | Significant quality loss | Not recommended |

### Why We Chose 2x Speed

1. **Quality Preservation**: At 2x speed, Gemini's audio understanding remains excellent. Speech is still clear and distinguishable.

2. **Proven Results**: We tested with 10.7 hours of real meeting recordings (Hindi-English mix) and achieved 100% transcription success rate.

3. **Diminishing Returns**: While 3x/4x would save more on input costs, the quality degradation leads to:
   - Missing words and phrases
   - Incorrect transcriptions requiring manual correction
   - Potential need for re-processing (negating savings)

4. **ffmpeg Compatibility**: The `atempo` filter works optimally in the 0.5-2.0 range. Higher speeds require filter chaining, adding complexity.

### Actual Cost Savings (Validated)

For our 10.7-hour transcription job using Gemini 3 Pro Preview:

| Component | Without 2x Speed | With 2x Speed | Savings |
|-----------|-----------------|---------------|---------|
| Audio file duration | 640 min | 320 min | 50% |
| Input tokens | 1,228,800 | 614,400 | 614,400 (50%) |
| Input cost (@$2/1M) | $2.46 | $1.23 | **$1.23** |
| Output tokens* | ~48,000 | ~48,000 | $0 |
| Output cost (@$12/1M) | $0.58 | $0.58 | $0 |
| **Total Cost** | **$3.03** | **$1.80** | **$1.23 (41%)** |

*Output tokens (transcription text) are not affected by input speed - the same content produces similar output regardless of playback speed.

### Cost Projection at Scale

| Audio Duration | Without 2x | With 2x | Annual Savings (if weekly) |
|---------------|-----------|---------|---------------------------|
| 10 hours | $3.03 | $1.80 | $64/year |
| 50 hours | $15.15 | $9.00 | $320/year |
| 100 hours | $30.30 | $18.00 | $640/year |

---

## Prerequisites

- Python 3.8+
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- ffmpeg (bundled via `imageio-ffmpeg`)

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/myExperimentsWithTruth/ScalabeMeetingTranscriber.git
cd ScalabeMeetingTranscriber

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Your API Key

```bash
# Option 1: Environment variable (recommended)
export GEMINI_API_KEY="your-api-key-here"

# Option 2: Pass via command line
python audio_transcriber.py audio.mp3 --api-key "your-api-key-here"
```

### 3. Convert Video to Audio (if needed)

If you have video files (e.g., Teams recordings), first extract the audio:

```bash
# Using ffmpeg directly
ffmpeg -i meeting.mp4 -vn -acodec libmp3lame -q:a 4 meeting.mp3

# Or use the provided helper script
python video_to_audio.py meeting.mp4
```

### 4. Run Transcription

```bash
# Basic usage
python audio_transcriber.py meeting.mp3

# Multiple files
python audio_transcriber.py day1_session1.mp3 day1_session2.mp3

# With custom settings
python audio_transcriber.py meeting.mp3 \
    --chunk-duration 300 \
    --speed 1.5 \
    --workers 5 \
    --output-dir transcripts/
```

## Usage Guide

### Command Line Options

```
usage: audio_transcriber.py [-h] [--chunk-duration SECONDS] [--speed FACTOR]
                            [--api-key KEY] [--workers N] [--chunks-dir DIR]
                            [--transcripts-dir DIR] [--output-dir DIR]
                            [--min-chunk SECONDS] [--model MODEL]
                            [--skip-chunking] [--skip-transcription]
                            FILE [FILE ...]

Arguments:
  FILE                    MP3 file(s) to transcribe

Options:
  --chunk-duration, -c    Duration of each chunk in seconds (default: 600)
  --speed, -s             Speed factor for audio (default: 2.0)
  --api-key, -k           Gemini API key (or set GEMINI_API_KEY env var)
  --workers, -w           Parallel transcription workers (default: 3)
  --chunks-dir            Directory for chunk files (default: chunks)
  --transcripts-dir       Directory for transcript files (default: transcripts)
  --output-dir, -o        Directory for final transcripts (default: .)
  --min-chunk             Minimum chunk duration in seconds (default: 10)
  --model                 Gemini model (default: models/gemini-3-pro-preview)
  --skip-chunking         Skip chunking, use existing chunks
  --skip-transcription    Skip transcription, consolidate existing
```

### Workflow Examples

#### Full Pipeline (Recommended)
```bash
python audio_transcriber.py meeting.mp3 --output-dir final_transcripts/
```

#### Two-Step Process (For Large Files)
```bash
# Step 1: Create chunks only (no API calls - validate before spending)
python audio_transcriber.py meeting.mp3 --skip-transcription

# Verify chunks look correct
ls -la chunks/

# Step 2: Transcribe and consolidate
python audio_transcriber.py meeting.mp3 --skip-chunking --output-dir final_transcripts/
```

#### Resume After Interruption
```bash
# If transcription was interrupted, existing transcripts are preserved
python audio_transcriber.py meeting.mp3 --skip-chunking --output-dir final_transcripts/
```

## Output Structure

```
project/
├── chunks/                          # Intermediate chunk files
│   ├── meeting_chunk_000.mp3
│   ├── meeting_chunk_001.mp3
│   └── ...
├── transcripts/                     # Individual chunk transcripts
│   ├── meeting_chunk_000.txt
│   ├── meeting_chunk_001.txt
│   └── ...
└── final_transcripts/               # Consolidated output
    └── meeting_full.txt
```

### Output Format

The final transcript includes chunk markers with timestamps:

```
--- Chunk 0 [00:00] ---
[00:00] Speaker 1: Welcome everyone to today's meeting...
[00:45] Speaker 2: Thanks for joining. Let's start with...
[01:30] Speaker 1: Great point. Moving on to the next topic...

--- Chunk 1 [10:00] ---
[10:00] Speaker 2: As we discussed earlier...
[10:45] Speaker 1: I agree with that assessment...
```

## How It Works

### 1. Chunking Phase
- Splits audio into 10-minute segments (configurable)
- Applies speed factor (2x default) using ffmpeg's atempo filter
- Processes multiple files in parallel
- Skips chunks shorter than minimum duration (10s default)

### 2. Transcription Phase
- Uploads each chunk to Gemini API
- Uses optimized prompt for mixed-language transcription
- Corrects timestamps: `corrected = (chunk_timestamp × speed_factor) + chunk_offset`
- Implements retry with exponential backoff for rate limits
- Saves individual chunk transcripts for fault tolerance

### 3. Consolidation Phase
- Merges all chunk transcripts in chronological order
- Adds chunk boundary markers with original timestamps
- Produces single consolidated file per source audio

## Troubleshooting

### "API key not found"
```bash
export GEMINI_API_KEY="your-key"
# or
python audio_transcriber.py audio.mp3 --api-key "your-key"
```

### "Rate limit exceeded"
- Reduce `--workers` to 2 or 1
- The script automatically retries with exponential backoff

### "ffmpeg not found"
```bash
pip install imageio-ffmpeg
```

### Interrupted transcription
```bash
# Resume from where you left off
python audio_transcriber.py audio.mp3 --skip-chunking
```

## Performance Results

Tested with real meeting recordings:

| Audio Duration | Chunks | Transcription Time | Output Size | Success Rate |
|---------------|--------|-------------------|-------------|--------------|
| 1.2 hours     | 7      | ~3 minutes        | 66 KB       | 100% |
| 1.8 hours     | 11     | ~5 minutes        | 130 KB      | 100% |
| 3.7 hours     | 22     | ~10 minutes       | 198 KB      | 100% |
| 4.0 hours     | 24     | ~12 minutes       | 217 KB      | 100% |
| **Total: 10.7 hours** | **64** | **~30 minutes** | **611 KB** | **100%** |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Development Approach

This project was developed using [Kiro](https://kiro.dev), an AI-powered IDE.

### Why Spec-Driven Development?

Initially, we attempted "vibe coding" - describing the problem and letting the AI generate code iteratively. Despite this being a seemingly simple problem (chunk audio, transcribe, merge), vibe coding failed to capture:

- **Timestamp correction complexity**: The math for correcting timestamps across speed-adjusted, chunked audio requires careful consideration of offsets and multipliers
- **Edge cases**: Minimum chunk duration, parallel processing race conditions, API retry logic
- **Cost implications**: Understanding that Gemini charges by audio duration (32 tokens/second) was critical for the speed optimization decision
- **Integration points**: Wiring chunker → transcriber → consolidator with proper error handling and resume capability

After scrapping the vibe-coded attempts, we switched to **spec-driven development**:

1. **Requirements Document**: Formal acceptance criteria using EARS patterns
2. **Design Document**: Architecture, data models, correctness properties
3. **Implementation Tasks**: Incremental, testable steps with clear dependencies

This structured approach caught issues early (like the timestamp correction formula) and produced production-quality code on the first implementation pass.

### Lesson Learned

Even for "simple" problems, spec-driven development pays off when there are:
- Multiple interacting components
- Mathematical transformations (timestamps, costs)
- External API dependencies with cost implications
- Need for fault tolerance and resume capability

## Acknowledgments

- [Kiro](https://kiro.dev) for spec-driven AI development
- Google Gemini API for powerful transcription capabilities
- ffmpeg for audio processing
- imageio-ffmpeg for bundled ffmpeg distribution
