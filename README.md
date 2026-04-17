This script is a comprehensive **AI-powered pipeline** that converts YouTube video content into structured text and concise summaries. It is designed to run entirely locally, ensuring privacy and avoiding API costs.

---

## **Core Technologies & Tools**

The script orchestrates several high-performance tools to handle different stages of the media processing pipeline:

| Tool | Role | Description |
| :--- | :--- | :--- |
| **`yt-dlp`** | **Media Downloader** | A powerful command-line utility that extracts the highest quality audio streams from YouTube URLs. |
| **`FFmpeg`** | **Audio Processor** | The industry-standard multimedia framework used here to convert raw video/audio streams into standardized `.mp3` files. |
| **`OpenAI Whisper`** | **STT (Speech-to-Text)** | A general-purpose speech recognition model. It performs local transcription, handling multiple languages and accents with high accuracy. |
| **`Ollama`** | **LLM Runner** | A local server environment that allows you to run Large Language Models (LLMs) like Gemma, Llama, or Mistral on your own hardware. |
| **`Gemma 3:4B`** | **Summarizer** | Google's lightweight, state-of-the-art open model. It processes the raw transcript to extract key insights and main ideas. |

---

## **Workflow Pipeline**

The script follows a linear, three-step process:

### **1. Extraction**
The `extract_audio` function uses `yt-dlp` to fetch the audio. It creates a sanitized filename based on the video title to ensure the script doesn't crash due to illegal characters in the file system.

### **2. Transcription**
The `transcribe` function loads the **Whisper** model into memory. 
* **Model Sizes**: Users can choose from `tiny` (fastest) to `large` (most accurate). 
* **Processing**: It analyzes the audio and produces a dictionary containing the full text and individual "segments" (sentences with timestamps).

### **3. Summarization (Optional)**
If the `--summarize` flag is used, the `summarize_with_gemma` function:
* Sends the transcript text to a local **Ollama** server via an internal API call.
* Uses a "System Prompt" to tell the AI how to behave (e.g., "Summarize concisely in clear paragraphs").
* Returns the AI-generated summary.

---

## **Technical Requirements**

To run this script, your system needs the following installed:

* **Python 3.8+**
* **System Binaries**: `ffmpeg` (required for audio conversion) and `ollama` (for summarization).
* **Python Libraries**:
    ```bash
    pip install yt-dlp openai-whisper torch
    ```

---

## **Key Features**

* **Cleanup Logic**: By default, the script deletes the bulky `.mp3` file after transcription to save disk space, unless the `--keep-audio` flag is used.
* **Local-First**: No data is sent to OpenAI or Google; everything is processed on your CPU/GPU.
* **Flexible Output**: Saves results as standard `.txt` files for the transcript and formatted `.md` or `.txt` for the summary.

> **Note on Performance**: Transcription speed depends heavily on your hardware. If you have an NVIDIA GPU, Whisper will automatically use **CUDA** to speed up the process significantly.