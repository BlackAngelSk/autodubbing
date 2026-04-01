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
source .venv/bin/activate
pip install -r requirements.txt
```

Install ffmpeg (Linux):

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
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
2. Pick target language and model/device.
3. Keep `TTS Engine` on `edge` for more natural neural speech.
4. Choose an `Edge Voice` (auto-updates when language changes).
5. Optionally set `Start Time` and `End Time` to dub only a part of the video.
6. Click **Generate Dubbed Video**.
7. Watch live progress with ETA while it processes.
8. Use **Cancel Current Run** if you need to stop processing.
9. Expand **Advanced Voice Timing Controls** to tune:
  - Original audio mix level under dubbed speech.
  - Min/Max stretch speed for timing fit.
  - Silence trim around synthesized speech segments.
10. Preview output and read logs.

Generated videos are saved in `outputs/`.

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

- `--whisper-model` (default: `small`)
- `--device` (default: `auto`) values: `auto`, `cpu`, `cuda`
- `--tts-engine` values: `edge` (default), `gtts`
- `--edge-voice` custom voice, example: `en-US-AriaNeural`
- `--start-time` start second for dubbing window (default: `0`)
- `--end-time` optional end second for dubbing window
- `--keep-temp` keep intermediate files for debugging

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
- If translation fails for some segments, retry later (service/network issue).
- If voices sound too fast/slow, edit `fit_audio_to_duration` logic in `autodub.py`.
- For more natural speech, use `edge` engine and a voice matching your target language.
- If the UI does not open, check that Gradio installed successfully: `pip install -r requirements.txt`.
- If YouTube download fails, update downloader: `pip install -U yt-dlp`.
- If a YouTube link still fails, try a direct watch URL (not playlist/channel links).
