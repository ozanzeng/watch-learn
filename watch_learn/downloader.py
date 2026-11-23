"""Download YouTube transcripts, audio, and video via yt-dlp."""

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import webvtt
import yt_dlp

from . import DownloadResult, TranscriptEntry

logger = logging.getLogger(__name__)

# Çok dilli ses kanalı (otomatik dublaj) olan videolarda "en" çeviri olur ve 429 alır;
# asıl İngilizce altyazı "en-orig" adıyla gelir, bu yüzden ayrı grup olarak önce denenir.
_SUBTITLE_LANG_GROUPS = (("tr", "tr-orig"), ("en-orig",), ("en", "en-US", "en-GB"))

_VIDEO_ID_RE = re.compile(
    r"(?:[?&]v=|youtu\.be/|/embed/|/shorts/|/live/|/v/)([A-Za-z0-9_-]{11})"
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ASS_TAG_RE = re.compile(r"\{\\[^}]*\}")
_WHITESPACE_RE = re.compile(r"\s+")
# whisper writes music/noise as "[Music]", "(müzik)", "♪ ... ♪"; those are not speech
_NON_SPEECH_RE = re.compile(r"[\[(*♪][^\[\]()*♪]*[\])*♪]")
# A rolling caption repeats the previous text and appends new words; merge those cues.
_ROLLING_WINDOW = 0.75

_YTDLP_COMMON_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "noprogress": True,
    "noplaylist": True,
    "retries": 3,
    "fragment_retries": 3,
    "socket_timeout": 30,
}

_AUDIO_TIMEOUT = 600
_TRANSCRIBE_TIMEOUT = 3600


def extract_video_id(url: str) -> str:
    """Extract a YouTube video ID from a URL. Falls back to 'unknown'."""
    match = _VIDEO_ID_RE.search(url)
    return match.group(1) if match else "unknown"


def download_video(
    url: str,
    output_dir: Path,
    groq_api_key: str | None = None,
    whisper_model: Path | None = None,
    language: str = "auto",
) -> DownloadResult:
    """Orchestrate transcript + video download with Whisper fallback."""
    video_id = extract_video_id(url)
    transcript_path, fmt = _download_transcript(url, output_dir)
    video_path, title, info_id = _download_video_file(url, output_dir)
    if info_id:
        video_id = info_id

    if transcript_path is not None:
        entries = _parse_transcript_file(transcript_path, fmt)
        source = fmt
        logger.debug(f"Parsed {len(entries)} entries from {fmt} transcript")
        if not entries:
            logger.warning(f"{fmt} transcript was empty; falling back to audio transcription")
            entries, source = _transcribe_fallback(
                video_path, output_dir, groq_api_key, whisper_model, language,
            )
    else:
        entries, source = _transcribe_fallback(
            video_path, output_dir, groq_api_key, whisper_model, language,
        )

    return DownloadResult(
        video_path=video_path,
        transcript_entries=entries,
        transcript_source=source,
        temp_dir=output_dir,
        video_id=video_id,
        title=title,
    )


def load_local_video(
    path: Path,
    output_dir: Path,
    groq_api_key: str | None = None,
    whisper_model: Path | None = None,
    language: str = "auto",
) -> DownloadResult:
    """Use a video file on disk: a sidecar .srt/.vtt wins, otherwise transcribe its audio."""
    entries: list[TranscriptEntry] = []
    source = "none"
    for ext in ("srt", "vtt"):
        sidecar = path.with_suffix(f".{ext}")
        if sidecar.exists():
            entries, source = _parse_transcript_file(sidecar, ext), ext
            if entries:
                break
            logger.warning(f"{sidecar.name} was empty; ignoring it")
    else:
        entries, source = _transcribe_fallback(path, output_dir, groq_api_key, whisper_model, language)

    return DownloadResult(
        video_path=path,
        transcript_entries=entries,
        transcript_source=source,
        temp_dir=output_dir,
        video_id=local_video_id(path),
        title=path.stem,
    )


def local_video_id(path: Path) -> str:
    """Folder-safe ID for a local file (its name without the extension)."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")[:60] or "local"


def _transcribe_fallback(
    video_path: Path,
    work_dir: Path,
    groq_api_key: str | None,
    whisper_model: Path | None,
    language: str,
) -> tuple[list[TranscriptEntry], str]:
    """No subtitles: local whisper.cpp first (audio stays on this machine), then Groq."""
    if whisper_model is not None and shutil.which("whisper-cli"):
        logger.info("No transcript found — transcribing locally with whisper.cpp...")
        try:
            entries = _transcribe_with_whisper_cpp(video_path, work_dir, whisper_model, language)
            logger.debug(f"whisper.cpp returned {len(entries)} entries")
            if entries:
                return entries, "whisper.cpp"
            logger.warning("whisper.cpp returned no speech segments")
        except Exception as e:
            logger.warning(f"whisper.cpp failed: {e}")

    if groq_api_key:
        logger.info("No transcript found — sending audio to Groq Whisper...")
        try:
            audio_path = _extract_audio(video_path, work_dir / "audio.mp3")
            entries = _transcribe_with_whisper(audio_path, groq_api_key)
            logger.debug(f"Whisper returned {len(entries)} entries")
            if entries:
                return entries, "whisper"
            logger.warning("Groq Whisper returned no speech segments")
        except Exception as e:
            logger.warning(f"Groq Whisper failed: {e}")

    logger.warning("No transcript, no whisper.cpp model and no GROQ_API_KEY — frames will be evenly spaced")
    return [], "synthetic"


def _download_transcript(url: str, output_dir: Path) -> tuple[Path | None, str]:
    """Try manual then auto-generated subtitles; return (path, format) or (None, 'none')."""
    last_error: Exception | None = None
    # Dil grupları sırayla denenir; ilk başarılı grupta durulur. Hepsini tek istekte
    # istemek, çevrilmiş (en) altyazıda 429 alıp bulunmuş tr altyazısını da boşa çıkarıyordu.
    for langs in _SUBTITLE_LANG_GROUPS:
        opts = {
            **_YTDLP_COMMON_OPTS,
            "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": list(langs),
            "subtitlesformat": "vtt",
            "skip_download": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as e:
            last_error = e

        for ext in ("vtt", "srt"):
            for lang in langs:
                matches = sorted(output_dir.glob(f"*.{lang}.{ext}"))
                if matches:
                    return matches[0], ext

    # A subtitle error (often HTTP 429) is not fatal: the video download below reports
    # inaccessible videos, and a missing transcript falls back to Whisper.
    if last_error is not None:
        logger.warning(f"Subtitle download failed: {str(last_error)[:200]}")
    return None, "none"


def _extract_audio(video_path: Path, out_path: Path) -> Path:
    """Extract mono audio with ffmpeg: 16 kHz WAV for whisper.cpp, 64 kbps MP3 for Groq."""
    codec = (
        ["-ar", "16000", "-c:a", "pcm_s16le"]
        if out_path.suffix == ".wav"
        else ["-ar", "16000", "-b:a", "64k"]
    )
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", str(video_path), "-vn", "-ac", "1", *codec, str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_AUDIO_TIMEOUT, check=False)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"audio extraction timed out after {_AUDIO_TIMEOUT}s") from e
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"audio extraction failed: {result.stderr.strip()[-200:]}")
    return out_path


def _download_video_file(url: str, output_dir: Path) -> tuple[Path, str, str]:
    """Download an MP4 (max 720p) for frame extraction; return (path, title, video_id)."""
    opts = {
        **_YTDLP_COMMON_OPTS,
        "outtmpl": str(output_dir / "%(id)s_video.%(ext)s"),
        # 720p üstü kare çıkarma için gereksiz; indirme süresini ve disk kullanımını düşürür
        "format": (
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]"
            "/bestvideo[height<=720]+bestaudio/best[height<=720]/best"
        ),
        "merge_output_format": "mp4",
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True) or {}
    except yt_dlp.utils.DownloadError as e:
        raise ValueError(f"Cannot access video: {e}") from e

    matches = sorted(output_dir.glob("*_video.mp4"))
    if not matches:
        # Some videos download without the _video suffix when merge happens
        matches = sorted(output_dir.glob("*.mp4"))
    if not matches:
        raise RuntimeError("Video download produced no .mp4 file")
    return matches[0], info.get("title") or "", info.get("id") or ""


def _transcribe_with_whisper_cpp(
    video_path: Path,
    work_dir: Path,
    model: Path,
    language: str,
) -> list[TranscriptEntry]:
    """Transcribe locally with the whisper.cpp CLI (whisper-cli) and its JSON output."""
    wav_path = _extract_audio(video_path, work_dir / "audio16k.wav")
    out_base = work_dir / "whisper_cpp"
    cmd = [
        "whisper-cli", "-m", str(model), "-l", language,
        "-t", str(min(8, os.cpu_count() or 4)),
        "-f", str(wav_path), "-oj", "-of", str(out_base), "-np",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TRANSCRIBE_TIMEOUT, check=False)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"whisper.cpp timed out after {_TRANSCRIBE_TIMEOUT}s") from e
    json_path = out_base.with_suffix(".json")
    if result.returncode != 0 or not json_path.exists():
        raise RuntimeError(result.stderr.strip()[-300:] or "no JSON output")

    data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    entries = []
    for seg in data.get("transcription", []):
        text = _WHITESPACE_RE.sub(" ", seg.get("text", "")).strip()
        if not text or _NON_SPEECH_RE.fullmatch(text):
            continue
        offsets = seg.get("offsets", {})
        entries.append(TranscriptEntry(
            start_sec=offsets.get("from", 0) / 1000,
            end_sec=offsets.get("to", 0) / 1000,
            text=text,
        ))
    return entries


def _transcribe_with_whisper(audio_path: Path, api_key: str) -> list[TranscriptEntry]:
    """Transcribe audio via Groq Whisper API with segment-level timestamps."""
    from groq import Groq, RateLimitError

    client = Groq(api_key=api_key)
    response = None

    for attempt in range(2):
        try:
            with audio_path.open("rb") as f:
                response = client.audio.transcriptions.create(
                    file=(audio_path.name, f),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            break
        except RateLimitError:
            if attempt == 0:
                logger.warning("Groq rate limit hit — retrying in 60s...")
                time.sleep(60)
            else:
                raise
        except Exception as e:
            raise RuntimeError(f"Groq Whisper request failed: {e}") from e

    if response is None:
        raise RuntimeError("Groq Whisper returned no response")

    entries = []
    for seg in response.segments or []:
        text = _WHITESPACE_RE.sub(" ", seg.text.strip())
        if text:
            entries.append(TranscriptEntry(
                start_sec=float(seg.start),
                end_sec=float(seg.end),
                text=text,
            ))
    return entries


def _parse_transcript_file(path: Path, fmt: str) -> list[TranscriptEntry]:
    if fmt == "vtt":
        return _parse_vtt(path)
    return _parse_srt(path)


def _parse_vtt(path: Path) -> list[TranscriptEntry]:
    """Parse WebVTT using webvtt-py; strips tags and merges rolling-window captions."""
    try:
        captions = webvtt.read(str(path))
    except Exception:
        # Some auto-captions have malformed headers; try a buffer read, then raw regex.
        try:
            captions = webvtt.from_buffer(
                path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            )
        except Exception:
            logger.warning(f"{path.name} is not valid WebVTT; parsing it as SRT-style cues")
            return _parse_srt(path)

    entries: list[TranscriptEntry] = []
    seen: set[str] = set()
    for cap in captions:
        clean = _HTML_TAG_RE.sub("", cap.text)
        clean = _ASS_TAG_RE.sub("", clean)
        clean = _WHITESPACE_RE.sub(" ", clean).strip()
        if not clean:
            continue
        start = _timecode_to_seconds(cap.start)
        end = _timecode_to_seconds(cap.end)

        if clean in seen:
            # Exact repeat: just extend the previous cue instead of duplicating text
            if entries and entries[-1].text == clean:
                entries[-1].end_sec = max(entries[-1].end_sec, end)
            continue

        if entries and _is_rolling_caption(entries[-1], clean, start):
            entries[-1].text = clean
            entries[-1].end_sec = max(entries[-1].end_sec, end)
            seen.add(clean)
            continue

        entries.append(TranscriptEntry(start_sec=start, end_sec=end, text=clean))
        seen.add(clean)
    return entries


def _is_rolling_caption(previous: TranscriptEntry, text: str, start: float) -> bool:
    """True when `text` is the previous caption plus a few new words."""
    return (
        start <= previous.end_sec + _ROLLING_WINDOW
        and len(text) > len(previous.text)
        and text.startswith(previous.text)
    )


# The cue number is optional: some tools emit SRT blocks without it.
_SRT_BLOCK_RE = re.compile(
    r"(?:^[ \t]*\d+[ \t]*\r?\n)?"
    r"[ \t]*(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})[ \t]*-->[ \t]*"
    r"(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})[^\n]*\r?\n"
    r"([\s\S]*?)(?=\r?\n[ \t]*\r?\n|\Z)",
    re.MULTILINE,
)


def _parse_srt(path: Path) -> list[TranscriptEntry]:
    """Parse SRT subtitle file with pure regex."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    entries = []
    for m in _SRT_BLOCK_RE.finditer(text):
        start_raw, end_raw, body = m.groups()
        clean = _HTML_TAG_RE.sub("", body)
        clean = _ASS_TAG_RE.sub("", clean)
        clean = _WHITESPACE_RE.sub(" ", clean).strip()
        if clean:
            entries.append(TranscriptEntry(
                start_sec=_timecode_to_seconds(start_raw),
                end_sec=_timecode_to_seconds(end_raw),
                text=clean,
            ))
    return entries


def _timecode_to_seconds(tc: str) -> float:
    """Convert HH:MM:SS.mmm, HH:MM:SS,mmm or MM:SS.mmm to float seconds."""
    parts = tc.strip().replace(",", ".").split(":")
    try:
        values = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"invalid timecode: {tc!r}") from e
    if len(values) == 2:
        minutes, seconds = values
        return minutes * 60 + seconds
    if len(values) == 3:
        hours, minutes, seconds = values
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"invalid timecode: {tc!r}")
