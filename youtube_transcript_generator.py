#!/usr/bin/env python3
"""
YouTube Transcript Generator

Downloads audio from a YouTube video, transcribes it locally with Whisper,
and optionally summarizes the transcript with Gemma 4B via Ollama.

Requirements:
    pip install yt-dlp openai-whisper
    brew install ffmpeg          # or: sudo apt install ffmpeg

    For summarization (--summarize):
        1. Install Ollama: https://ollama.com
        2. Pull the model: ollama pull gemma3:4b
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import whisper
import yt_dlp


def sanitize_filename(name: str) -> str:
    """Turn a video title into a safe filename."""
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    return name[:120]


def extract_audio(url: str, output_dir: str) -> tuple[str, str]:
    """Download audio from a YouTube URL. Returns (audio_path, video_title)."""
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "video")
        # yt-dlp renames the file after post-processing to .mp3
        audio_path = os.path.join(output_dir, f"{ydl.prepare_filename(info).rsplit('.', 1)[0]}.mp3")

    if not os.path.exists(audio_path):
        # Fallback: find the mp3 in the output dir
        mp3s = list(Path(output_dir).glob("*.mp3"))
        if not mp3s:
            raise FileNotFoundError("Audio extraction failed — no mp3 file found.")
        audio_path = str(mp3s[0])

    print(f"Audio saved to: {audio_path}")
    return audio_path, title


def transcribe(audio_path: str, model_name: str = "base") -> dict:
    """Transcribe audio using Whisper. Returns the full result dict."""
    print(f"Loading Whisper model '{model_name}' ...")
    model = whisper.load_model(model_name)

    print("Transcribing (this may take a while) ...")
    result = model.transcribe(audio_path)
    return result


def summarize_with_gemma(text: str, ollama_url: str = "http://localhost:11434") -> str:
    """Send the transcript to Gemma 4B running in Ollama and return a summary."""
    import urllib.request

    prompt = (
        "You are an expert content summarizer. Please summarize the following transcript "
        "using the following structure:\n"
        "1. ## Executive Summary: A 2-3 sentence high-level overview.\n"
        "2. ## Key Takeaways: Use a bulleted list for the most important points.\n"
        "3. ## Detailed Breakdown: Use short paragraphs to explain the core concepts.\n\n"
        "Use Markdown for all formatting (bolding, headers, lists).\n\n"
        f"TRANSCRIPT:\n{text}\n\nSUMMARY:"
    )

    payload = json.dumps({
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3  # Lower temperature makes the output more structured/less random
        }
    }).encode()

    req = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    print("Generating summary with Gemma 4B (via Ollama) ...")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode())
            return body.get("response", "").strip()
    except Exception as e:
        print(f"Error contacting Ollama: {e}")
        print("Make sure Ollama is running (`ollama serve`) and you've pulled the model (`ollama pull gemma3:4b`).")
        sys.exit(1)


def save_text(text: str, path: str) -> None:
    Path(path).write_text(text, encoding="utf-8")
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download a YouTube video, transcribe it with Whisper, and optionally summarize with Gemma 4B."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "-m", "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium). Larger = more accurate but slower.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Directory to save outputs (default: current directory)",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Summarize the transcript using Gemma 4B via Ollama",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the downloaded audio file after transcription",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Download audio ---
    print(f"\n{'='*60}")
    print("Step 1: Downloading audio from YouTube")
    print(f"{'='*60}")
    audio_path, title = extract_audio(args.url, output_dir)
    safe_title = sanitize_filename(title)

    # --- 2. Transcribe ---
    print(f"\n{'='*60}")
    print("Step 2: Transcribing with Whisper")
    print(f"{'='*60}")
    result = transcribe(audio_path, args.model)
    #transcript_text = result["text"].strip()
    transcript_text = "\n".join([segment["text"].strip() for segment in result.get("segments", [])])

    transcript_path = os.path.join(output_dir, f"{safe_title}_transcript.txt")
    save_text(transcript_text, transcript_path)

    # --- 3. Summarize (optional) ---
    if args.summarize:
        print(f"\n{'='*60}")
        print("Step 3: Summarizing with Gemma 4B")
        print(f"{'='*60}")
        summary = summarize_with_gemma(transcript_text, args.ollama_url)
        summary_path = os.path.join(output_dir, f"{safe_title}_summary.txt")
        save_text(summary, summary_path)

    # --- Cleanup ---
    if not args.keep_audio:
        os.remove(audio_path)
        print(f"Removed temporary audio: {audio_path}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"  Transcript: {transcript_path}")
    if args.summarize:
        print(f"  Summary:    {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
