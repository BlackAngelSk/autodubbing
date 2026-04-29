# Auto Dubbing for Video

This project provides a command-line tool that can:

1. Extract audio from a video.
2. Transcribe speech with Whisper.
3. Translate each segment to a target language.
4. Generate synthetic speech for each translated segment.
5. Rebuild a dubbed track aligned to the original timing.
6. Merge the dubbed track back into the video.

It also includes a web UI for easier navigation.

## Important Notes

- This is a practical starter implementation, not studio-quality dubbing.
- Best results come from clear speech and short sentence segments.
- `edge-tts`, `gTTS`, and `deep-translator` rely on online services.
- You can also use direct YouTube links (requires `yt-dlp`).
- `ffmpeg` must be installed on your system.

## Install

```bash
python -m venv .venv
pip install -r requirements.txt
```

Activate the virtual environment based on your shell:

```bash
# bash/zsh
source .venv/bin/activate

# fish
source .venv/bin/activate.fish
```

Optional for faster Whisper model downloads and to avoid the HF Hub auth warning:

```bash
export HF_TOKEN=hf_your_token_here
```

### Windows (automatic)

Run either of these from the project folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_windows.ps1
```

or just double-click:

```text
install_windows.bat
```

This will:
- install Python and FFmpeg with `winget` when available
- create `.venv`
- install everything from `requirements.txt`

Then start the UI with either:

```text
start.bat
```

or:

```text
run_ui_windows.bat
```

Install ffmpeg (Linux):

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Quick start on Linux/macOS:

```bash
bash ./run_ui_unix.sh
```

## Auto Update

Update to the latest version and refresh dependencies:

Linux/macOS:

```bash
bash ./update_unix.sh
```

Windows:

```text
update_windows.bat
```

## Usage

### Web UI (Recommended)

Start the UI:

```bash
python ui.py
```

Then open the local URL shown in terminal (usually `http://127.0.0.1:7860`).

UI flow:

1. Upload video or paste a YouTube URL.
2. Pick a `Quality Preset` (`Fast`, `Balanced`, or `Best Quality`).
3. Choose target language and, if needed, manually override the Whisper model/device.
4. Use `Optimization Profile`:
   - `Auto` for normal use
   - `Balanced` for manual middle-ground control
   - `Short video quality` for clips/songs
   - `Long video stability` for longer videos
5. Keep `TTS Engine` on `edge` for more natural neural speech.
6. Leave `ASR Engine` on `Auto` unless you need to force Whisper compatibility.
7. Optionally switch `Translation Provider` between `Google` and `MyMemory`.
8. Choose an `Edge Voice` (auto-updates when language changes).
9. Optionally enable:
   - `Keep original audio quietly in background` (turn off for dubbed voice only)
   - `Export translated subtitles (.srt)`
   - `Resume previous job if possible`
   - `Glossary Overrides` like `death => smrti`
10. Optionally set `Start Time` and `End Time` to dub only a part of the video.
11. Click **Generate Dubbed Video**.
12. Preview output, download the generated `.srt`, and read logs.

Generated videos are saved in `outputs/`, and resume caches are stored under `outputs/.autodub_resume/`.

If using YouTube URL in UI, install:

```bash
pip install yt-dlp
```

### CLI

```bash
python autodub.py \
  --input input_video.mp4 \
  --output dubbed_output.mp4 \
  --target-lang es \
  --start-time 30 \
  --end-time 180
```

Optional arguments:

- `--whisper-model` (default: `small`) options include `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `large-v3-turbo`, `distil-large-v3`
- `--device` (default: `auto`) values: `auto`, `cpu`, `cuda`
- `--translation-provider` values: `google` (default), `mymemory`
- `--hf-token` optional Hugging Face token for authenticated Whisper model downloads
- `--tts-engine` values: `edge` (default), `gtts`
- `--edge-voice` custom voice, example: `en-US-AriaNeural`
- `--start-time` start second for dubbing window (default: `0`)
- `--end-time` optional end second for dubbing window
- `--keep-temp` keep intermediate files for debugging
- `--disable-original-audio` output dubbed speech only (no original source audio mixed in)
- `--optimization-profile` choose `auto`, `balanced`, `short`, or `long`
- `--asr-engine` values: `auto` (default, prefers `stable-ts` when available), `whisper`, `stable-ts`
- `--no-export-srt` skip translated `.srt` output
- `--no-resume` disable resume-cache reuse
- `--glossary-file` load glossary overrides from a text file

Example with Hindi dubbing:

```bash
python autodub.py --input talk.mp4 --output talk_hi.mp4 --target-lang hi --tts-engine edge --edge-voice hi-IN-SwaraNeural
```

## Language Codes

Use ISO language codes for `--target-lang`, for example:

- `en` English
- `es` Spanish
- `fr` French
- `de` German
- `hi` Hindi
- `ja` Japanese
- `pt` Portuguese
- `ru` Russian
- `sk` Slovak

## What the script creates

By default, temp files are generated in a temporary folder and deleted when complete.

If `--keep-temp` is passed, the script will show the temp directory path so you can inspect:

- extracted audio
- per-segment TTS files
- final dubbed WAV before muxing

## Troubleshooting

- If transcription is slow, try `--whisper-model base`.
- For the best speech pickup, keep `--asr-engine auto`; it now prefers `stable-ts` and uses a higher-recall Whisper fallback path.
- For best transcription quality on a capable GPU, try `--whisper-model large-v3`.
- The first run of larger Whisper models may download several GB of weights.
- If you see an HF Hub unauthenticated warning, set `HF_TOKEN` in your shell or paste it into the UI's optional token field.
- If one translation service struggles, switch the UI or CLI to `mymemory` and retry.
- If translation fails for some segments, retry later (service/network issue).
- Translation now runs with bounded parallel workers for longer jobs to improve speed while keeping provider load controlled.
- If voices sound too fast/slow, edit `fit_audio_to_duration` logic in `autodub.py`.
- For more natural speech, use `edge` engine and a voice matching your target language.
- If the UI does not open, check that Gradio installed successfully: `pip install -r requirements.txt`.
- On older Windows CPUs, use `install_windows.bat` or `start.bat` so a compatible `numpy==1.26.4` build is installed automatically.
- If you have an NVIDIA GPU but runs still use CPU, set `--device cuda` explicitly and update GPU dependencies (`faster-whisper`/`ctranslate2` + CUDA runtime libs). The loader now tries multiple CUDA compute modes automatically.
- If YouTube download fails, update downloader: `pip install -U yt-dlp`.
- If a YouTube link still fails, try a direct watch URL (not playlist/channel links).
