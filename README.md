# Say What - a YouTube video-to-text transcriber
 Provide either a YouTube URL, or a file with URLs, or point Say What to a channel to have it retrieve all uploaded video URLs and then run OpenAI's SOTA Whisper model to transcribe the video into text.
# Say What

A powerful and robust YouTube transcription tool using OpenAI's Whisper speech recognition model. Say What makes it easy to convert speech to text from individual videos, batches of videos, or entire YouTube channels.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **High-Quality Transcription**: Uses OpenAI's state-of-the-art Whisper speech recognition model
- **Hardware Optimized**: Detects GPU/CPU capabilities and optimizes processing accordingly
- **Progress Tracking**: Real-time progress bar with accurate time estimation
- **Channel Processing**: Extract and transcribe all videos from a YouTube channel
- **Batch Processing**: Process multiple videos from a URL list
- **Playlist Handling**: Automatically extracts video IDs from playlist URLs
- **Skip Existing**: Avoids reprocessing videos that have already been transcribed
- **Detailed Error Reporting**: Comprehensive logs for failed transcriptions with troubleshooting tips
- **Multiple Model Sizes**: Choose from tiny, base, small, medium, or large models based on accuracy needs

## Installation

### Prerequisites

- Python 3.6+
- ffmpeg

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/say-what.git
cd say-what
```

2. Install required packages:
```bash
pip install openai-whisper yt-dlp torch psutil
```

3. Make sure ffmpeg is installed on your system:
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or install with Chocolatey: `choco install ffmpeg`

## Usage

### Transcribe a Single Video

```bash
python say-what.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Transcribe Multiple Videos from a File

Create a text file with one URL per line (lines starting with # are ignored):

```bash
python say-what.py --file urls.txt
```

### Process an Entire YouTube Channel

Extract and transcribe all videos from a YouTube channel:

```bash
python say-what.py --channel "https://www.youtube.com/@ChannelName"
```

Only extract URLs without transcribing (useful for large channels):

```bash
python say-what.py --channel "https://www.youtube.com/@ChannelName" --extract-only
```

Limit the number of videos to process from a channel:

```bash
python say-what.py --channel "https://www.youtube.com/@ChannelName" --max-videos 100
```

### Additional Options

```bash
  --url URL, -u URL     YouTube video URL
  --file FILE, -f FILE  File containing YouTube URLs (one per line)
  --channel CHANNEL, -c CHANNEL
                        YouTube channel URL to extract and transcribe all videos
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory for transcript files (default: transcripts)
  --model MODEL, -m MODEL
                        Whisper model size (tiny, base, small, medium, large)
  --keep-audio          Keep the downloaded audio files
  --max-videos MAX_VIDEOS
                        Maximum number of videos to extract from channel (newest first)
  --extract-only        Only extract URLs from channel without transcribing
```

## Model Sizes and Performance

| Model  | Size   | Required VRAM | Speed (CPU) | Speed (GPU) | Accuracy |
|--------|--------|---------------|-------------|-------------|----------|
| tiny   | 39 MB  | <1 GB         | ~100x       | ~120x       | Low      |
| base   | 74 MB  | <1 GB         | ~80x        | ~120x       | Basic    |
| small  | 244 MB | ~1 GB         | ~60x        | ~120x       | Good     |
| medium | 769 MB | ~2.5 GB       | ~30x        | ~90x        | Very Good|
*Speed is expressed as a multiple of real-time (higher is better). For example, 30x means processing a 30-minute video in 1 minute.*

The default model is `large` for maximum accuracy, but you can choose a smaller model for faster processing or if you have limited system resources.

## Say What in Action

Say What provides detailed progress feedback during transcription:

```
Processing: Psychology of Propaganda, Leadership, and Creativity
=================================================================
Using CPU on Intel Core i7 @ 3.2GHz (8 cores) for inference
Audio duration: 1:30:00 (5400.00 seconds)
Estimated processing speed: 18.5x real-time
Estimated completion time: 0:04:52

[████████████░░░░░░] 60% | ⠼ | Elapsed: 0:02:55 | ETA: 0:01:57
```

## Examples

### Extract URLs from a Channel

```bash
# Extract all videos from a channel to a file
python channel_extractor.py "https://www.youtube.com/@samvaknin"

# Extract only the 100 most recent videos
python channel_extractor.py "https://www.youtube.com/@samvaknin" --max 100
```

### Transcribe Videos with Different Models

```bash
# Fast transcription with small model
python say-what.py --file urls.txt --model small

# Highest quality transcription
python say-what.py --file urls.txt --model large
```

## Output Files

- **Transcripts**: Saved in the output directory (default: "transcripts") with filenames based on video titles
- **URL Lists**: When extracting from channels, saved as `[channel_name]_video_urls.txt`
- **Error Reports**: Failed URLs are logged in `failed_urls.txt` and a detailed `error_report.txt`

## Troubleshooting

1. **Model Download Issues**: If the script fails to download the Whisper model, try running with a smaller model first or check your internet connection.

2. **Audio Download Errors**: Make sure yt-dlp is up-to-date:
   ```bash
   pip install -U yt-dlp
   ```

3. **Memory Issues**: If you encounter memory errors, try a smaller model size:
   ```bash
   python robust-youtube-transcriber.py --url "YOUR_URL" --model small
   ```

4. **CPU/GPU Detection**: If you have issues with hardware detection, the script will fall back to conservative estimates. You can manually install dependencies for better detection:
   ```bash
   pip install torch psutil
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - For the incredible speech recognition model
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - For reliable YouTube video downloading