## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - For the incredible speech recognition model
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - For reliable YouTube video downloading# YouTube Transcriber

A robust tool to transcribe YouTube videos using OpenAI's Whisper speech recognition model. This script handles both individual videos and batch processing, with support for extracting all videos from a YouTube channel.

## Features

- **High-quality transcription** using OpenAI's Whisper model
- **Automatic filename generation** based on video titles
- **Batch processing** of multiple YouTube URLs
- **YouTube channel support** - extract all videos from a channel
- **Robust error handling** with detailed error reports
- **Resume capability** - skips already transcribed videos
- **Playlist URL handling** - automatically extracts video IDs from playlist URLs
- **Keeps track of failures** with detailed error diagnostics

## Installation

1. First, ensure you have Python 3.6+ installed.
2. Install the required dependencies:

```bash
pip install openai-whisper yt-dlp
```

3. Make sure you have ffmpeg installed on your system:
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or install with Chocolatey: `choco install ffmpeg`

4. Download both Python scripts from this repository:
   - `robust-youtube-transcriber.py` - Main transcription script
   - `channel_url_extractor.py` - Utility for extracting channel videos

## Usage

### Transcribe a Single Video

```bash
python robust-youtube-transcriber.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Transcribe Multiple Videos from a File

1. Create a text file with one URL per line (lines starting with # are ignored)
2. Run:

```bash
python robust-youtube-transcriber.py --file urls.txt
```

### Process an Entire YouTube Channel

Extract and transcribe all videos from a YouTube channel:

```bash
python robust-youtube-transcriber.py --channel "https://www.youtube.com/@ChannelName"
```

Only extract URLs without transcribing (useful for large channels):

```bash
python robust-youtube-transcriber.py --channel "https://www.youtube.com/@ChannelName" --extract-only
```

Limit the number of videos to process from a channel (newest first):

```bash
python robust-youtube-transcriber.py --channel "https://www.youtube.com/@ChannelName" --max-videos 100
```

### Additional Options

- `--output-dir` / `-o`: Specify output directory (default: "transcripts")
- `--model` / `-m`: Choose Whisper model size (tiny, base, small, medium, large)
- `--keep-audio`: Keep the downloaded audio files
- `--max-videos`: Maximum number of videos to extract from a channel (newest first)
- `--extract-only`: Only extract URLs from a channel without transcribing

## Model Sizes

The script uses OpenAI's Whisper model for transcription. Available model sizes are:

- `tiny`: Fastest, lowest accuracy (~1GB RAM)
- `base`: Fast with better accuracy (~1GB RAM)
- `small`: Good balance of speed and accuracy (~2GB RAM)
- `medium`: High accuracy, slower (~5GB RAM)
- `large`: Highest accuracy, slowest (~10GB RAM)

The default is `large` for maximum transcription quality.

## Output Files

- Transcribed text files are saved in the output directory (default: "transcripts")
- Each transcript file is named based on the video title
- For channel extraction, URLs are saved to `[channel_name]_video_urls.txt`
- Failed URLs are logged in `failed_urls.txt` with error messages
- A detailed error report is generated in `error_report.txt`

## Error Handling

If some videos fail to process, the script will:

1. Continue processing the remaining videos
2. Create a `failed_urls.txt` file with basic error information
3. Create a detailed `error_report.txt` with possible causes and troubleshooting tips

## Examples

### Extract URLs from a Channel

```bash
# Extract all videos from a channel to a file
python channel_url_extractor.py "https://www.youtube.com/@samvaknin"

# Extract only the 100 most recent videos
python channel_url_extractor.py "https://www.youtube.com/@samvaknin" --max 100
```

### Transcribe Videos

```bash
# Transcribe with the large model and save to a custom directory
python robust-youtube-transcriber.py --file urls.txt --output-dir my_transcripts --model large

# Extract and transcribe all videos from a channel
python robust-youtube-transcriber.py --channel "https://www.youtube.com/@samvaknin" --output-dir samvaknin_transcripts
```

## Troubleshooting

1. **Model Download Issues**: If the script fails to download the Whisper model, try running with a smaller model first.
2. **Audio Download Errors**: Make sure yt-dlp is up-to-date: `pip install -U yt-dlp`
3. **Memory Issues**: If you encounter memory errors, try a smaller model size.

## License

This project is open source and available for personal and educational use.