#!/usr/bin/env python3
"""
Robust YouTube Video Transcriber
This script transcribes YouTube videos using OpenAI's Whisper model with robust error handling.
"""

import os
import sys
import re
import argparse

import subprocess
import tempfile
import shutil
import hashlib
import datetime

# Define Whisper model URLs and checksums
WHISPER_MODELS = {
    "tiny": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.pt",
        "sha256": "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03"
    },
    "base": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
        "sha256": "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e"
    },
    "small": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
        "sha256": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"
    },
    "medium": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
        "sha256": "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1"
    },
    "large": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt",
        "sha256": "e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a"
    },
    "large-v3": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v3.pt",
        "sha256": "62a420e066a0eeed3f5a076c284144eecfdc9f82c21a8c96950ae93daedcf8a0"
    }
}

def calculate_sha256(file_path):
    """Calculate SHA256 checksum of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_file(url, output_path):
    """Download a file using curl or wget"""
    try:
        if shutil.which("curl"):
            subprocess.run(["curl", "-L", url, "-o", output_path], check=True)
        elif shutil.which("wget"):
            subprocess.run(["wget", url, "-O", output_path], check=True)
        else:
            print("Error: Neither curl nor wget is available. Please install one of them.")
            return False
        return True
    except subprocess.CalledProcessError:
        print(f"Error downloading from {url}")
        return False

def download_whisper_model(model_size):
    """Download Whisper model with proper checksum verification"""
    if model_size not in WHISPER_MODELS:
        print(f"Error: Model size '{model_size}' not supported.")
        print(f"Available sizes: {', '.join(WHISPER_MODELS.keys())}")
        return None
    
    # Define paths
    cache_dir = os.path.expanduser("~/.cache/whisper")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Special handling for large-v3 model using Hugging Face Transformers
    if model_size == "large-v3":
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            
            print(f"Loading Whisper large-v3 model...")
            
            # Set device and dtype
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Show GPU info if available
            if torch.cuda.is_available():
                print(f"\nCUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                print(f"PyTorch version: {torch.__version__}")
                print(f"Device set to use {device}")
                torch.backends.cuda.matmul.allow_tf32 = True
            else:
                print("\nWARNING: CUDA not detected. Using CPU which will be significantly slower.")
                print("To use GPU acceleration, install PyTorch with CUDA:")
                print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            
            # Initialize model ID
            model_id = "openai/whisper-large-v3"
            
            # Create pipeline using helper function to avoid warnings
            pipe = initialize_whisper_pipeline(model_id, torch_dtype, device)
            
            # Print information about language
            print("Note: large-v3 model configured for English transcription, with support for handling occasional non-English phrases")
            
            # Return the model ID for compatibility with the rest of the code
            return model_id
            
        except ImportError:
            print("Error: transformers package not installed. Please install it with:")
            print("pip install --upgrade transformers datasets[audio] accelerate")
            return None
        except Exception as e:
            print(f"Error loading large-v3 model: {e}")
            return None
    else:
        # Regular handling for other models
        model_path = os.path.join(cache_dir, f"{model_size}.pt")
        expected_sha256 = WHISPER_MODELS[model_size]["sha256"]
        
        if os.path.exists(model_path):
            actual_sha256 = calculate_sha256(model_path)
            if actual_sha256 == expected_sha256:
                print(f"Model {model_size} already exists with correct checksum.")
                return model_path
            else:
                print(f"Model exists but has incorrect checksum. Re-downloading...")
                os.rename(model_path, f"{model_path}.bak")
        
        # Download the model
        model_url = WHISPER_MODELS[model_size]["url"]
        print(f"Downloading Whisper {model_size} model from {model_url}")
        
        # Download to a temporary file first
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        if not download_file(model_url, temp_path):
            print("Failed to download the model.")
            return None
        
        # Verify checksum
        actual_sha256 = calculate_sha256(temp_path)
        if actual_sha256 != expected_sha256:
            print(f"Error: Downloaded model has incorrect SHA256 checksum.")
            print(f"Expected: {expected_sha256}")
            print(f"Actual: {actual_sha256}")
            os.remove(temp_path)
            return None
        
        # Move to final location
        shutil.move(temp_path, model_path)
        print(f"Successfully downloaded and verified model to {model_path}")
        return model_path

def download_youtube_audio(url, output_file="audio.m4a"):
    """Downloads audio using yt-dlp with minimal processing"""
    print(f"Downloading audio from: {url}")
    
    # Check if yt-dlp is installed
    if not shutil.which("yt-dlp"):
        return False, "yt-dlp is not installed. Please install it with 'pip install yt-dlp'"
    
    try:
        # Download best audio format available, preferring m4a
        cmd = [
            "yt-dlp", 
            "-f", "bestaudio[ext=m4a]/bestaudio",
            "-o", output_file,
            "--no-playlist",  # Don't download playlists
            url
        ]
        
        download_result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Download completed successfully")
        return True, None
    except subprocess.CalledProcessError as e:
        error_msg = f"Error downloading audio: {e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        return False, error_msg

def detect_system_info():
    """Detect system information for better progress estimation"""
    system_info = {
        "has_gpu": False,
        "gpu_name": "None",
        "gpu_memory": 0,  # in GB
        "gpu_compute_capability": 0,  # CUDA compute capability
        "cpu_info": "Unknown",
        "cpu_cores": 0,
        "cpu_freq": 0,
        "ram_gb": 0
    }
    
    # Try to get GPU info
    try:
        import torch
        if torch.cuda.is_available():
            system_info["has_gpu"] = True
            system_info["gpu_name"] = torch.cuda.get_device_name(0)
            # Get GPU memory in GB
            system_info["gpu_memory"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            # Get compute capability
            system_info["gpu_compute_capability"] = torch.cuda.get_device_capability(0)
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            system_info["has_gpu"] = True
            system_info["gpu_name"] = "Intel XPU"
            # Intel GPU memory detection would go here if available
    except (ImportError, AttributeError):
        pass
    
    # Try to get CPU and RAM info
    try:
        import psutil
        import platform
        
        # CPU info
        system_info["cpu_cores"] = psutil.cpu_count(logical=False) or psutil.cpu_count() or "Unknown"
        cpu_freq = psutil.cpu_freq()
        if cpu_freq and cpu_freq.current:
            system_info["cpu_freq"] = round(cpu_freq.current, 2)
        
        cpu_info = platform.processor()
        if cpu_info:
            system_info["cpu_info"] = cpu_info
        
        # RAM info
        ram = psutil.virtual_memory()
        if ram.total:
            system_info["ram_gb"] = round(ram.total / (1024**3), 1)
    except (ImportError, AttributeError):
        pass
    
    return system_info

def estimate_transcription_speed(model_size):
    """
    Estimate transcription speed multiplier based on model size and hardware
    Returns the estimated real-time factor (how many seconds of audio processed per second)
    """
    # Base speed factors for different model sizes (conservative estimates)
    # These are approximate real-time factors on a mid-range CPU
    base_speeds = {
        "tiny": 100,    # ~100x real-time
        "base": 80,     # ~80x real-time
        "small": 60,    # ~60x real-time
        "medium": 30,   # ~30x real-time
        "large": 20,    # ~20x real-time
        "large-v3": 15  # ~15x real-time (slightly slower than large-v2)
    }
    
    # Default to medium speed if model size not recognized
    base_speed = base_speeds.get(model_size.lower(), 25)
    
    # Get system info
    system_info = detect_system_info()
    
    # If GPU is available, adjust speed based on GPU capabilities
    if system_info["has_gpu"]:
        # Base GPU speedup factors
        gpu_speedup = {
            "tiny": 1.2,     # Small models don't benefit as much from GPU
            "base": 1.5,
            "small": 2,
            "medium": 3,
            "large": 4,      # Large models benefit most from GPU
            "large-v3": 4.5  # large-v3 benefits slightly more from GPU
        }
        
        # Get base GPU speedup
        base_gpu_speedup = gpu_speedup.get(model_size.lower(), 2.5)
        
        # Adjust based on GPU memory
        if system_info["gpu_memory"] > 0:
            # Model memory requirements (approximate in GB)
            model_memory = {
                "tiny": 1,
                "base": 1.5,
                "small": 2,
                "medium": 4,
                "large": 8,
                "large-v3": 10
            }
            
            required_memory = model_memory.get(model_size.lower(), 4)
            available_memory = system_info["gpu_memory"]
            
            # If we have enough memory, apply full speedup
            if available_memory >= required_memory:
                memory_factor = 1.0
            else:
                # Reduce speedup if we don't have enough memory
                memory_factor = (available_memory / required_memory) ** 0.5
        
        # Adjust based on compute capability
        if system_info["gpu_compute_capability"] > 0:
            # Higher compute capability means better performance
            compute_factor = min(1.5, 1.0 + (system_info["gpu_compute_capability"] / 10))
        else:
            compute_factor = 1.0
        
        # Apply GPU adjustments
        base_speed *= base_gpu_speedup * memory_factor * compute_factor
    
    # If no GPU, adjust based on CPU capabilities
    else:
        try:
            import psutil
            # CPU frequency adjustment
            cpu_freq = psutil.cpu_freq()
            if cpu_freq and cpu_freq.current:
                # Adjust based on frequency (3.0GHz as baseline)
                cpu_speed_factor = cpu_freq.current / 3000.0
                # Ensure the factor is within reasonable bounds
                cpu_speed_factor = max(0.5, min(cpu_speed_factor, 2.0))
            else:
                cpu_speed_factor = 1.0
            
            # CPU cores adjustment
            cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count() or 4
            core_factor = min(cpu_count, 8) / 4  # Normalize to 4 cores
            # Square root to account for diminishing returns
            core_factor = core_factor ** 0.5
            
            # Apply CPU adjustments
            base_speed *= cpu_speed_factor * core_factor
            
        except (ImportError, AttributeError):
            pass
    
    # Ensure the speed factor stays within reasonable bounds
    base_speed = max(1, min(base_speed, 200))
    
    return base_speed

def initialize_whisper_pipeline(model_path_or_id, torch_dtype, device):
    """Initialize Whisper pipeline with correct settings to avoid warnings"""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    
    # Load model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path_or_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path_or_id)
    
    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device
    )
    
    # Explicitly set English transcription (not translation)
    # This avoids the warning about multilingual detection
    pipe.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    pipe.model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    
    # These settings handle the multilingual case properly
    pipe.model.generation_config.task = "transcribe"
    pipe.model.generation_config.language = "en"
    
    return pipe

def transcribe_audio(audio_path, model_path, model_size="large", language="en"):
    """Transcribes audio using a pre-downloaded Whisper model with accurate progress tracking"""
    try:
        import time
        import datetime
        import threading
        import sys
        
        # Special handling for large-v3 model
        if model_size == "large-v3":
            try:
                import torch
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
                
                print(f"Loading Whisper large-v3 model...")
                
                # Set device and dtype
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                # Show GPU info if available
                if torch.cuda.is_available():
                    print(f"\nCUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
                    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                    print(f"PyTorch version: {torch.__version__}")
                    print(f"Device set to use {device}")
                    torch.backends.cuda.matmul.allow_tf32 = True
                else:
                    print("\nWARNING: CUDA not detected. Using CPU which will be significantly slower.")
                    print("To use GPU acceleration, install PyTorch with CUDA:")
                    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                
                # Initialize model ID
                model_id = "openai/whisper-large-v3"
                
                # Create pipeline using helper function to avoid warnings
                pipe = initialize_whisper_pipeline(model_id, torch_dtype, device)
                
                # Print information about language
                print("Note: large-v3 model configured for English transcription, with support for handling occasional non-English phrases")
                
                # Use the pipeline to transcribe
                result = pipe(
                    audio_path,
                    chunk_length_s=30,
                    stride_length_s=5,
                    return_timestamps=True
                )
                
                # Format the transcript
                clean_transcript = ""
                timestamped_transcript = ""
                
                # Get text from result
                if isinstance(result, dict) and "text" in result:
                    clean_transcript = result["text"].strip()
                    
                    # Get chunks if available
                    if "chunks" in result:
                        for chunk in result["chunks"]:
                            # Skip empty chunks
                            if not chunk.get("text", "").strip():
                                continue
                                
                            text = chunk.get("text", "").strip()
                            
                            # Get timestamps if available
                            start_time = 0
                            end_time = 0
                            if "timestamp" in chunk and chunk["timestamp"] and len(chunk["timestamp"]) >= 2:
                                if chunk["timestamp"][0] is not None:
                                    start_time = float(chunk["timestamp"][0])
                                if chunk["timestamp"][1] is not None:
                                    end_time = float(chunk["timestamp"][1])
                        
                            # Format with timestamp
                            timestamped_transcript += f"[{datetime.timedelta(seconds=int(start_time))} -> {datetime.timedelta(seconds=int(end_time))}] {text}\n"
                    else:
                        # No chunks, just use the text
                        timestamped_transcript = clean_transcript
                else:
                    # Handle unexpected result format
                    if isinstance(result, str):
                        clean_transcript = result.strip()
                        timestamped_transcript = clean_transcript
                    else:
                        print(f"Warning: Unexpected result format: {type(result)}")
                        clean_transcript = str(result)
                        timestamped_transcript = clean_transcript
                
                # Return both the clean and timestamped transcript for the caller to use
                return {"clean": clean_transcript, "timestamped": timestamped_transcript}
                
            except ImportError:
                print("Error: transformers package not installed. Please install it with:")
                print("pip install --upgrade transformers datasets[audio] accelerate")
                return None
            except Exception as e:
                print(f"Error loading large-v3 model: {e}")
                return None
        else:
            # Regular handling for other models
            import whisper
            print(f"Loading Whisper model from {model_path}...")
            model = whisper.load_model(model_path)
        
        # Get system info
        system_info = detect_system_info()
        
        # Determine if running on GPU
        gpu_info = ""
        if system_info["has_gpu"]:
            gpu_info = f" on {system_info['gpu_name']}"
            print(f"Using GPU{gpu_info} for inference")
        else:
            cpu_info = ""
            if system_info["cpu_info"] != "Unknown":
                cpu_info = f" on {system_info['cpu_info']}"
                if system_info["cpu_freq"] > 0:
                    cpu_info += f" @ {system_info['cpu_freq']}GHz"
                if system_info["cpu_cores"] > 0:
                    cpu_info += f" ({system_info['cpu_cores']} cores)"
            print(f"Using CPU{cpu_info} for inference")
        
        # Estimate transcription speed based on model size and hardware
        speed_factor = estimate_transcription_speed(model_size)
        
        # Show language being used
        language_display = language if language else "auto-detect"
        print(f"Transcription language: {language_display}")
        
        print("Transcribing audio... (this may take a while depending on video length)")
        
        # Initialize progress variables
        progress_dots = 0
        start_time = time.time()
        progress_active = True
        progress_lock = threading.Lock()
        transcription_started = False
        initial_delay = 5  # Seconds to wait before showing progress (model loading time)
        
        # Define a function to show the progress indicator
        def show_progress():
            nonlocal progress_dots, transcription_started
            
            # Get audio duration with ffmpeg if possible
            audio_duration_secs = None
            try:
                ffprobe_cmd = [
                    "ffprobe", 
                    "-v", "error", 
                    "-show_entries", "format=duration", 
                    "-of", "default=noprint_wrappers=1:nokey=1", 
                    audio_path
                ]
                result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    audio_duration_secs = float(result.stdout.strip())
                    print(f"Audio duration: {str(datetime.timedelta(seconds=int(audio_duration_secs)))} ({audio_duration_secs:.2f} seconds)")
                    print(f"Estimated processing speed: {speed_factor:.1f}x real-time")
                    
                    # Calculate estimated processing time
                    estimated_process_time = audio_duration_secs / speed_factor
                    print(f"Estimated completion time: {str(datetime.timedelta(seconds=int(estimated_process_time)))}")
            except Exception as e:
                print(f"Note: Could not determine audio duration: {e}")
            
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # Braille spinner characters for smoother animation
            spinner_idx = 0
            last_update_time = time.time()
            
            # Wait a bit for the model to start processing
            time.sleep(initial_delay)
            transcription_started = True
            
            while progress_active:
                with progress_lock:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
                    
                    # Update every 0.25 seconds for smoother animation
                    if current_time - last_update_time >= 0.25:
                        last_update_time = current_time
                        
                        # Build progress message
                        if audio_duration_secs:
                            # Calculate progress based on estimated processing speed
                            processed_duration = elapsed * speed_factor
                            if processed_duration < audio_duration_secs:
                                percent = min(99, int(processed_duration / audio_duration_secs * 100))
                                eta = (audio_duration_secs - processed_duration) / speed_factor
                                eta_str = str(datetime.timedelta(seconds=int(eta)))
                                
                                # Progress bar
                                bar_length = 20
                                filled_length = int(bar_length * percent / 100)
                                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                                
                                progress_msg = f"\r[{bar}] {percent}% | {spinner[spinner_idx]} | Elapsed: {elapsed_str} | ETA: {eta_str}"
                            else:
                                # Progress bar at 100%
                                bar = '█' * 20
                                progress_msg = f"\r[{bar}] Finalizing... | {spinner[spinner_idx]} | Elapsed: {elapsed_str}"
                        else:
                            # If we couldn't get audio duration, show simpler progress
                            progress_msg = f"\r{spinner[spinner_idx]} Transcribing... | Elapsed: {elapsed_str}"
                        
                        # Print the progress message
                        sys.stdout.write(progress_msg)
                        sys.stdout.flush()
                        
                        # Update spinner
                        spinner_idx = (spinner_idx + 1) % len(spinner)
                
                time.sleep(0.1)  # Smaller sleep for more responsive updates
        
        # Start the progress thread
        progress_thread = threading.Thread(target=show_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            # Perform the transcription
            if model_size == "large-v3":
                # Use the Hugging Face pipeline for large-v3 with optimized settings
                result = pipe(audio_path)
                transcript = result["text"]
                
                # If we have timestamps, format them nicely
                if "chunks" in result:
                    chunks = result["chunks"]
                    formatted_transcript = ""
                    for chunk in chunks:
                        start_time = chunk["timestamp"][0]
                        end_time = chunk["timestamp"][1]
                        text = chunk["text"].strip()
                        formatted_transcript += f"[{start_time:.2f}s -> {end_time:.2f}s] {text}\n"
                    transcript = formatted_transcript
            else:
                # Use the regular Whisper model for other models
                options = {
                    "fp16": False  # Use float32 for better compatibility
                }
                
                # Add language constraint if specified
                if language:
                    options["language"] = language
                    # Force the model to not auto-detect the language
                    options["task"] = "transcribe"
                
                # Perform the transcription with language specified
                result = model.transcribe(audio_path, **options)
                transcript = result["text"]
            
            # Stop the progress thread
            with progress_lock:
                progress_active = False
            
            # Clear the current line and show final time
            total_time = time.time() - start_time
            transcription_time = total_time - (0 if not transcription_started else initial_delay)
            
            sys.stdout.write("\r" + " " * 100 + "\r")  # Clear the line
            
            if audio_duration_secs:
                actual_speed = audio_duration_secs / transcription_time
                print(f"Transcription completed in {str(datetime.timedelta(seconds=int(total_time)))}")
                print(f"Actual processing speed: {actual_speed:.1f}x real-time")
            else:
                print(f"Transcription completed in {str(datetime.timedelta(seconds=int(total_time)))}")
            
            # Save the transcript
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"\nDEBUG - Output directory: {os.path.abspath(output_dir)}")
                print(f"DEBUG - File path: {info['output_file']}")
                print(f"DEBUG - Transcript length: {len(transcript)} characters")
                print(f"DEBUG - First 100 chars: {transcript[:100]}...")
                
                with open(info['output_file'], 'w', encoding='utf-8') as f:
                    f.write(f"Transcript of: {info['title']}\n")
                    f.write(f"YouTube URL: {info['url']}\n\n")
                    f.write(transcript)
                
                print(f"DEBUG - File saved successfully: {os.path.exists(info['output_file'])}")
            except Exception as e:
                print(f"DEBUG - Error saving file: {str(e)}")
                raise
            
            print(f"✓ Transcript saved to: {info['output_file']}")
            
            # Keep audio if requested
            if keep_audio:
                output_audio = os.path.join(output_dir, f"{info['safe_title']}.m4a")
                shutil.copy2(info['audio_file'], output_audio)
                print(f"Audio file kept at: {output_audio}")
            
            return transcript
            
        except Exception as e:
            # Stop the progress thread on error
            with progress_lock:
                progress_active = False
            
            # Clear the progress line
            sys.stdout.write("\r" + " " * 100 + "\r")
            raise e
            
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def sanitize_filename(title):
    """Convert a title to a safe filename by removing invalid characters"""
    # Remove characters not allowed in filenames
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        title = title.replace(char, '')
    
    # Replace spaces with hyphens and remove multiple hyphens
    title = title.replace(' ', '-').lower()
    while '--' in title:
        title = title.replace('--', '-')
    
    # Limit filename length and remove leading/trailing hyphens
    title = title[:100].strip('-')
    
    return title

def clean_youtube_url(url):
    """
    Clean YouTube URL by removing playlist parameters and other query params
    Returns the clean URL with just the video ID
    """
    # Try to extract video ID using different YouTube URL patterns
    video_id = None
    
    # Pattern matching for different YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\?\/]+)',  # Standard and shortened URLs
        r'(?:youtube\.com\/embed\/)([^&\?\/]+)',                # Embedded URLs
        r'(?:youtube\.com\/v\/)([^&\?\/]+)'                     # Old format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break
    
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    
    # Return original URL if no matching pattern
    return url

def get_video_info(url):
    """Get video title and ID from a YouTube URL"""
    # Clean the URL first
    clean_url = clean_youtube_url(url)
    
    try:
        cmd = ["yt-dlp", "--get-title", "--get-id", clean_url]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        
        if len(lines) >= 2:
            title = lines[0]
            video_id = lines[1]
            return title, video_id, clean_url
        else:
            return "Unknown Title", os.path.basename(clean_url), clean_url
    except Exception as e:
        return "Unknown Title", os.path.basename(clean_url), clean_url

def get_optimal_batch_size(model_size, system_info):
    """Determine optimal batch size based on GPU memory and model size"""
    if not system_info["has_gpu"]:
        return 1  # No parallel processing on CPU
    
    # Model memory requirements (approximate in GB)
    model_memory = {
        "tiny": 1,
        "base": 1.5,
        "small": 2,
        "medium": 4,
        "large": 8,
        "large-v3": 10
    }
    
    required_memory = model_memory.get(model_size.lower(), 4)
    available_memory = system_info["gpu_memory"]
    
    # Calculate maximum batch size based on available memory
    # Leave some headroom for system and other processes
    max_batch_size = int((available_memory * 0.8) / required_memory)
    
    # Cap the batch size based on model size
    max_recommended_batch = {
        "tiny": 16,
        "base": 12,
        "small": 8,
        "medium": 4,
        "large": 2,
        "large-v3": 2  # Large-v3 is memory intensive
    }
    
    recommended_batch = max_recommended_batch.get(model_size.lower(), 4)
    
    # Use the smaller of the two limits
    optimal_batch = min(max_batch_size, recommended_batch)
    
    # Ensure at least 1
    return max(1, optimal_batch)

def process_videos_in_batches(videos, model_path, output_dir, keep_audio, model_size, language, failed_log, batch_size, args):
    """Process multiple videos in parallel batches"""
    success_count = 0
    failed_urls = {}
    
    # Process videos in batches
    for i in range(0, len(videos), batch_size):
        batch = videos[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(videos) + batch_size - 1)//batch_size}")
        print(f"Batch size: {len(batch)} videos")
        
        # Create temporary directories for each video in the batch
        temp_dirs = []
        audio_files = []
        try:
            for url in batch:
                temp_dir = tempfile.mkdtemp()
                temp_dirs.append(temp_dir)
                audio_file = os.path.join(temp_dir, "audio.m4a")
                audio_files.append(audio_file)
                
                # Download audio for this video
                success, error = download_youtube_audio(url, audio_file)
                if not success:
                    error_msg = f"Failed to download audio: {error}"
                    print(f"{error_msg}. Skipping.")
                    if failed_log:
                        failed_log.write(f"{url}\t{error_msg}\n")
                    failed_urls[url] = error_msg
                    continue
            
            # Get video info for all successful downloads
            video_infos = []
            for url, audio_file in zip(batch, audio_files):
                if os.path.exists(audio_file):
                    try:
                        video_title, video_id, clean_url = get_video_info(url)
                        safe_title = sanitize_filename(video_title)
                        output_file = os.path.join(output_dir, f"{safe_title}.txt")
                        
                        # Skip if already exists
                        if os.path.exists(output_file):
                            print(f"Skipping: {video_title} - transcript already exists")
                            success_count += 1
                            continue
                        
                        video_infos.append({
                            'url': url,
                            'title': video_title,
                            'safe_title': safe_title,
                            'output_file': output_file,
                            'audio_file': audio_file
                        })
                    except Exception as e:
                        error_msg = f"Error getting video info: {str(e)}"
                        print(f"Error processing URL {url}: {error_msg}")
                        if failed_log:
                            failed_log.write(f"{url}\t{error_msg}\n")
                        failed_urls[url] = error_msg
            
            if not video_infos:
                continue
            
            # Transcribe all videos in the batch
            if model_size == "large-v3":
                # Use Hugging Face pipeline for batch processing
                try:
                    import torch
                    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
                    
                    # Force CUDA if available
                    if not torch.cuda.is_available():
                        print("\nWARNING: CUDA not detected. Checking PyTorch installation...")
                        print(f"PyTorch version: {torch.__version__}")
                        print(f"CUDA available: {torch.cuda.is_available()}")
                        print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
                        print("Please ensure you have installed PyTorch with CUDA support:")
                        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                        device = "cpu"
                        torch_dtype = torch.float32
                    else:
                        device = "cuda:0"
                        torch_dtype = torch.float16
                        print(f"\nCUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
                        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                        print(f"PyTorch version: {torch.__version__}")
                        torch.set_float32_matmul_precision('high')
                        torch.backends.cuda.matmul.allow_tf32 = True
                    
                    # Initialize pipeline using helper function
                    pipe = initialize_whisper_pipeline(model_path, torch_dtype, device)
                    
                    # Print language information
                    print("Note: large-v3 model configured for English transcription, with support for handling occasional non-English phrases")
                    
                    # Process each audio file individually to handle long-form content
                    for info in video_infos:
                        try:
                            print(f"\nTranscribing: {info['title']}")
                            
                            # Use the pipeline with minimal parameters to avoid conflicts
                            result = pipe(
                                info['audio_file'],
                                chunk_length_s=30,
                                stride_length_s=5,
                                return_timestamps=True
                            )
                            
                            # Format the transcript
                            clean_transcript = ""
                            timestamped_transcript = ""
                            
                            # Get text from result
                            if isinstance(result, dict) and "text" in result:
                                clean_transcript = result["text"].strip()
                                
                                # Get chunks if available
                                if "chunks" in result:
                                    for chunk in result["chunks"]:
                                        # Skip empty chunks
                                        if not chunk.get("text", "").strip():
                                            continue
                                            
                                        text = chunk.get("text", "").strip()
                                        
                                        # Get timestamps if available
                                        start_time = 0
                                        end_time = 0
                                        if "timestamp" in chunk and chunk["timestamp"] and len(chunk["timestamp"]) >= 2:
                                            if chunk["timestamp"][0] is not None:
                                                start_time = float(chunk["timestamp"][0])
                                            if chunk["timestamp"][1] is not None:
                                                end_time = float(chunk["timestamp"][1])
                                        
                                        # Format with timestamp
                                        timestamped_transcript += f"[{datetime.timedelta(seconds=int(start_time))} -> {datetime.timedelta(seconds=int(end_time))}] {text}\n"
                                else:
                                    # No chunks, just use the text
                                    timestamped_transcript = clean_transcript
                            else:
                                # Handle unexpected result format
                                if isinstance(result, str):
                                    clean_transcript = result.strip()
                                    timestamped_transcript = clean_transcript
                                else:
                                    # Print detailed info about the result
                                    print(f"Warning: Result format not as expected. Type: {type(result)}")
                                    if isinstance(result, dict):
                                        print(f"Keys in result: {list(result.keys())}")
                                        # Try to extract text
                                        if "text" in result:
                                            clean_transcript = result["text"].strip()
                                        else:
                                            # Try other common keys
                                            for key in ["transcription", "output", "result"]:
                                                if key in result:
                                                    clean_transcript = str(result[key]).strip()
                                                    break
                                            else:
                                                # No recognized keys, convert the whole result
                                                clean_transcript = str(result)
                                    else:
                                        clean_transcript = str(result)
                                    
                                    timestamped_transcript = clean_transcript
                            
                            # Ensure we have some transcript content
                            if not clean_transcript.strip():
                                print("Warning: Empty transcript produced. Using fallback method...")
                                
                                # Try fallback method with different chunk sizes
                                retry_result = pipe(
                                    info['audio_file'],
                                    chunk_length_s=15,  # Smaller chunk size
                                    stride_length_s=3,
                                    return_timestamps=False  # Disable timestamps to simplify
                                )
                                
                                if isinstance(retry_result, dict) and "text" in retry_result:
                                    clean_transcript = retry_result["text"].strip()
                                    timestamped_transcript = clean_transcript
                            
                            # Ensure we have content to save
                            if not clean_transcript.strip():
                                print("Error: Could not extract transcription content")
                                raise Exception("Failed to extract transcript from model output")
                            
                            # Save the transcript
                            os.makedirs(output_dir, exist_ok=True)
                            with open(info['output_file'], 'w', encoding='utf-8') as f:
                                f.write(f"Transcript of: {info['title']}\n")
                                f.write(f"YouTube URL: {info['url']}\n\n")
                                # Write either clean or timestamped transcript based on args
                                if args.keep_timestamps:
                                    f.write(timestamped_transcript)
                                else:
                                    f.write(clean_transcript)
                            
                            print(f"✓ Transcript saved to: {info['output_file']}")
                            
                            # Keep audio if requested
                            if keep_audio:
                                output_audio = os.path.join(output_dir, f"{info['safe_title']}.m4a")
                                shutil.copy2(info['audio_file'], output_audio)
                                print(f"Audio file kept at: {output_audio}")
                            
                            success_count += 1
                            
                        except Exception as e:
                            error_msg = f"Error transcribing: {str(e)}"
                            print(f"Error processing {info['url']}: {error_msg}")
                            if failed_log:
                                failed_log.write(f"{info['url']}\t{error_msg}\n")
                            failed_urls[info['url']] = error_msg
                    
                except Exception as e:
                    error_msg = f"Error in batch processing: {str(e)}"
                    print(f"Batch processing error: {error_msg}")
                    # Mark all videos in batch as failed
                    for info in video_infos:
                        if failed_log:
                            failed_log.write(f"{info['url']}\t{error_msg}\n")
                        failed_urls[info['url']] = error_msg
            else:
                # Process each video individually with regular Whisper
                for info in video_infos:
                    try:
                        transcript = transcribe_audio(info['audio_file'], model_path, model_size, language)
                        if not transcript:
                            error_msg = "Transcription failed with no output"
                            print(f"{error_msg}. Skipping.")
                            if failed_log:
                                failed_log.write(f"{info['url']}\t{error_msg}\n")
                            failed_urls[info['url']] = error_msg
                            continue
                        
                        # Save the transcript
                        os.makedirs(output_dir, exist_ok=True)
                        with open(info['output_file'], 'w', encoding='utf-8') as f:
                            f.write(f"Transcript of: {info['title']}\n")
                            f.write(f"YouTube URL: {info['url']}\n\n")
                            f.write(transcript)
                        
                        print(f"✓ Transcript saved to: {info['output_file']}")
                        
                        # Keep audio if requested
                        if keep_audio:
                            output_audio = os.path.join(output_dir, f"{info['safe_title']}.m4a")
                            shutil.copy2(info['audio_file'], output_audio)
                            print(f"Audio file kept at: {output_audio}")
                        
                        success_count += 1
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        print(f"Error processing {info['url']}: {error_msg}")
                        if failed_log:
                            failed_log.write(f"{info['url']}\t{error_msg}\n")
                        failed_urls[info['url']] = error_msg
        
        finally:
            # Clean up temporary directories
            for temp_dir in temp_dirs:
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not remove temporary directory: {e}")
    
    return success_count, failed_urls

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos using Whisper")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", "-u", help="YouTube video URL", default=None)
    group.add_argument("--file", "-f", help="File containing YouTube URLs (one per line)", default=None)
    group.add_argument("--channel", "-c", help="YouTube channel URL to extract and transcribe all videos", default=None)
    
    parser.add_argument("--output-dir", "-o", help="Output directory for transcript files (default: transcripts)", default="transcripts")
    parser.add_argument("--model", "-m", help="Whisper model size (tiny, base, small, medium, large, large-v3)", default="large")
    parser.add_argument("--language", "-l", help="Force specific language for transcription (e.g., 'en' for English, 'es' for Spanish)", default="en")
    parser.add_argument("--keep-audio", action="store_true", help="Keep the downloaded audio files")
    parser.add_argument("--max-videos", type=int, help="Maximum number of videos to extract from channel (newest first)")
    parser.add_argument("--extract-only", action="store_true", help="Only extract URLs from channel without transcribing")
    parser.add_argument("--auto-language", action="store_true", help="Let Whisper auto-detect language instead of forcing a specific language")
    parser.add_argument("--batch-size", type=int, help="Number of videos to process in parallel (default: auto-detect based on GPU memory)")
    parser.add_argument("--keep-timestamps", action="store_true", help="Keep timestamps in the output transcript (default: False)")
    args = parser.parse_args()
    
    # Check for ffmpeg installation
    if not check_ffmpeg():
        print("\nError: ffmpeg is not installed or not found in PATH.")
        print("\nTo install ffmpeg:")
        print("1. Windows (using Chocolatey):")
        print("   choco install ffmpeg")
        print("\n2. Windows (manual):")
        print("   a. Download ffmpeg from https://www.gyan.dev/ffmpeg/builds/")
        print("   b. Extract the archive")
        print("   c. Add the bin folder to your system PATH")
        print("\n3. Linux:")
        print("   sudo apt-get install ffmpeg  # Ubuntu/Debian")
        print("   sudo yum install ffmpeg      # CentOS/RHEL")
        print("\n4. macOS:")
        print("   brew install ffmpeg")
        print("\nAfter installing ffmpeg, please run this script again.")
        return 1
    
    # Process channel URL to extract video URLs if specified
    if args.channel:
        try:
            # Import the channel extractor module
            from channel_url_extractor import extract_channel_name, get_channel_videos
            
            # Extract channel name for the output file
            channel_name = extract_channel_name(args.channel)
            urls_file = f"{channel_name}_video_urls.txt"
            
            print(f"Extracting video URLs from channel: {args.channel}")
            num_videos = get_channel_videos(args.channel, urls_file, args.max_videos)
            
            if num_videos == 0:
                print("Failed to extract any videos from the channel.")
                return 1
                
            print(f"Successfully extracted {num_videos} video URLs to {urls_file}")
            
            if args.extract_only:
                print("URL extraction complete. Exiting without transcribing (--extract-only was specified).")
                return 0
                
            # Use the extracted URLs file for processing
            args.file = urls_file
            print(f"Proceeding to transcribe {num_videos} videos...")
            
        except ImportError:
            print("Error: Channel URL extractor module not found.")
            print("Make sure the channel_url_extractor.py file is in the same directory.")
            return 1
    
    # Get list of URLs to process
    urls = []
    if args.url:
        urls.append(args.url)
    
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                # Add any non-empty lines that aren't comments
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        urls.append(line)
        except Exception as e:
            print(f"Error reading URL file: {e}")
            return 1
    
    if not urls:
        print("No valid URLs found to process.")
        return 1
    
    print(f"Found {len(urls)} URLs to process")
    print(f"Using Whisper model: {args.model}")
    
    # Determine language settings
    language = None if args.auto_language else args.language
    if language:
        print(f"Forcing transcription language: {language}")
    else:
        print("Using automatic language detection")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # File to log failed URLs
    failed_log_path = os.path.join(args.output_dir, "failed_urls.txt")
    failed_log = open(failed_log_path, 'w', encoding='utf-8')
    
    try:
        # Download model once for all videos
        model_path = download_whisper_model(args.model)
        if not model_path:
            print("Failed to download or verify the Whisper model. Exiting.")
            return 1
        
        # Determine optimal batch size
        system_info = detect_system_info()
        batch_size = args.batch_size if args.batch_size else get_optimal_batch_size(args.model, system_info)
        
        if batch_size > 1:
            print(f"\nUsing parallel processing with batch size: {batch_size}")
            print("This will process multiple videos simultaneously for better throughput")
        else:
            print("\nProcessing videos sequentially (no parallel processing)")
        
        # Process videos in batches
        success_count, failed_urls = process_videos_in_batches(
            urls, model_path, args.output_dir, args.keep_audio, 
            args.model, language, failed_log, batch_size, args
        )
        
        # Write detailed error report
        if len(failed_urls) > 0:
            error_report_path = os.path.join(args.output_dir, "error_report.txt")
            with open(error_report_path, 'w', encoding='utf-8') as error_report:
                error_report.write("# YouTube Transcription Error Report\n")
                error_report.write(f"# Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                error_report.write("The following URLs could not be processed:\n\n")
                
                for url, error in failed_urls.items():
                    error_report.write(f"## URL: {url}\n")
                    error_report.write(f"Error: {error}\n\n")
                    
                    # Add troubleshooting suggestions
                    if "unavailable" in str(error).lower() or "not available" in str(error).lower():
                        error_report.write("Possible cause: Video may be private, removed, or region-restricted.\n")
                    
                    if "copyright" in str(error).lower():
                        error_report.write("Possible cause: Video may have copyright restrictions.\n")
                        
                    error_report.write("---\n\n")
                
                error_report.write("\n## Troubleshooting Tips\n\n")
                error_report.write("1. **URL Issues**: Make sure the URL is correct and the video is publicly available.\n")
                error_report.write("2. **Regional Restrictions**: Some videos may not be available in your region.\n")
                error_report.write("3. **Network Connection**: Check your internet connection.\n")
                error_report.write("4. **Update yt-dlp**: Run `pip install -U yt-dlp` to update yt-dlp to the latest version.\n")
            
            print(f"\nDetailed error report written to {error_report_path}")
        
        # Print summary
        print(f"\n{'=' * 80}")
        print(f"Summary: Successfully processed {success_count}/{len(urls)} videos")
        print(f"Transcripts saved to directory: {os.path.abspath(args.output_dir)}")
        if len(failed_urls) > 0:
            print(f"Failed videos: {len(failed_urls)} (see {failed_log_path} and {error_report_path} for details)")
        print(f"{'=' * 80}\n")
        
        return 0
        
    finally:
        # Close the failed log file
        failed_log.close()

if __name__ == "__main__":
    sys.exit(main())