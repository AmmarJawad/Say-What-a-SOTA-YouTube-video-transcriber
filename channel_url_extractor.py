#!/usr/bin/env python3
"""
YouTube Channel Video URL Extractor
This script extracts all video URLs from a YouTube channel and saves them to a text file.
It can be used standalone or integrated with the YouTube transcription script.
"""

import os
import re
import argparse
import subprocess
import time
import sys

def extract_channel_name(url):
    """Extract the channel name from a YouTube channel URL"""
    # Try to match @username format
    match = re.search(r'youtube\.com/@([^/]+)', url)
    if match:
        return match.group(1)
    
    # Try to match channel ID format
    match = re.search(r'youtube\.com/channel/([^/]+)', url)
    if match:
        return match.group(1)
    
    # Try to match custom URL format
    match = re.search(r'youtube\.com/c/([^/]+)', url)
    if match:
        return match.group(1)
    
    # Try to match user format
    match = re.search(r'youtube\.com/user/([^/]+)', url)
    if match:
        return match.group(1)
    
    # If no pattern matches, use the whole URL as a basis
    return url.split('/')[-1]

def get_channel_videos(channel_url, output_file, max_videos=None, verbose=True):
    """
    Extract all video URLs from a YouTube channel and save to a file
    
    Parameters:
    - channel_url: URL of the YouTube channel
    - output_file: Path to save the list of video URLs
    - max_videos: Maximum number of videos to extract (None for all)
    - verbose: Whether to print progress information
    
    Returns:
    - Number of videos extracted
    """
    # Check if yt-dlp is installed
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: yt-dlp is not installed. Please install it with 'pip install yt-dlp'")
        return 0
    
    # Build command to extract video URLs
    cmd = [
        "yt-dlp",
        "--flat-playlist",  # Don't download videos, just extract metadata
        "--get-id",         # Get only the video IDs
        "--no-warnings",    # Suppress warnings
    ]
    
    # Add limit if specified
    if max_videos:
        cmd.extend(["-I", f"1:{max_videos}"])
    
    # Add URL to playlists/videos endpoint
    if "/videos" not in channel_url:
        if channel_url.endswith("/"):
            channel_url = channel_url + "videos"
        else:
            channel_url = channel_url + "/videos"
    
    cmd.append(channel_url)
    
    if verbose:
        print(f"Extracting video URLs from channel: {channel_url}")
        print("This may take a while depending on the number of videos...")
        if max_videos:
            print(f"Limited to {max_videos} most recent videos")
    
    start_time = time.time()
    
    try:
        # Run yt-dlp to get video IDs
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        video_ids = process.stdout.strip().split('\n')
        
        # Remove empty lines
        video_ids = [vid_id for vid_id in video_ids if vid_id]
        
        # Write video URLs to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            # Add a header comment
            f.write(f"# YouTube video URLs from channel: {channel_url}\n")
            f.write(f"# Extracted on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total videos: {len(video_ids)}\n\n")
            
            # Write each video URL
            for video_id in video_ids:
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                f.write(f"{video_url}\n")
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"✓ Extracted {len(video_ids)} videos in {elapsed_time:.2f} seconds")
            print(f"✓ URLs saved to: {output_file}")
        
        return len(video_ids)
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting video URLs: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Extract all video URLs from a YouTube channel")
    parser.add_argument("channel_url", help="URL of the YouTube channel")
    parser.add_argument("--output", "-o", help="Output file (default: [channel_name]_video_urls.txt)")
    parser.add_argument("--max", "-m", type=int, help="Maximum number of videos to extract (newest first)")
    args = parser.parse_args()
    
    # Extract channel name for default output file
    channel_name = extract_channel_name(args.channel_url)
    
    # Set output file
    if args.output:
        output_file = args.output
    else:
        output_file = f"{channel_name}_video_urls.txt"
    
    # Get video URLs
    num_videos = get_channel_videos(args.channel_url, output_file, args.max)
    
    if num_videos > 0:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())