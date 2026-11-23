"""Watch & Learn CLI entry point."""

import logging
import shutil
import tempfile
import time
from pathlib import Path

import click

from . import __version__
from .extractor import check_ffmpeg, get_video_duration

MAX_SHOT_FRAMES = 150  # shot storyboard pages beyond this get sampled evenly
LONG_SHOT_STEP = 3.0   # seconds between extra frames inside long shots


@click.command()
@click.version_option(version=__version__, prog_name="watch-learn")
@click.argument("url")
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Directory to save storyboard images.",
)
@click.option(
    "--groq-api-key", "-k",
    envvar="GROQ_API_KEY",
    default=None,
    help="Groq API key for Whisper transcription fallback (or set GROQ_API_KEY env var).",
)
@click.option(
    "--max-frames", "-n",
    type=int,
    default=30,
    show_default=True,
    help="Maximum smart frames to extract.",
)
@click.option(
    "--jpeg-quality", "-q",
    type=click.IntRange(1, 95),
    default=85,
    show_default=True,
    help="JPEG compression quality for storyboard output (1–95).",
)
@click.option(
    "--silence-gap", "-g",
    type=float,
    default=5.0,
    show_default=True,
    help="Minimum silence gap in seconds to trigger frame extraction (Rule B).",
)
@click.option(
    "--frame-format",
    type=click.Choice(["jpg", "png"]),
    default="jpg",
    show_default=True,
    help="Intermediate frame format (jpg=faster, png=lossless).",
)
@click.option(
    "--interval", "-i",
    type=click.FloatRange(min=0.2),
    default=None,
    help="Take a frame every N seconds instead of the transcript-driven selection.",
)
@click.option(
    "--cuts",
    is_flag=True,
    default=False,
    help="Edit-rhythm analysis: detect cuts, one frame per shot, audio hits and loudness.",
)
@click.option(
    "--cut-threshold",
    type=click.FloatRange(1.5, 10.0),
    default=3.0,
    show_default=True,
    help="How many times more a frame must change than its neighbours to count as a cut (lower = more sensitive).",
)
@click.option(
    "--whisper-model",
    envvar="WHISPER_CPP_MODEL",
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    default=None,
    help="whisper.cpp ggml model for local transcription when there are no subtitles (or set WHISPER_CPP_MODEL).",
)
@click.option(
    "--lang",
    default="auto",
    show_default=True,
    help="Spoken language for whisper.cpp (e.g. tr, en, auto).",
)
@click.option(
    "--keep-video",
    is_flag=True,
    default=False,
    help="Copy the downloaded video into the output folder as video.mp4.",
)
@click.option(
    "--keep-temp",
    is_flag=True,
    default=False,
    help="Keep temporary video/frame files after processing.",
)
@click.option(
    "--no-learn",
    is_flag=True,
    default=False,
    help="Skip self-learning keyword store update after this run.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging.",
)
def cli(
    url: str,
    output_dir: Path,
    groq_api_key: str | None,
    max_frames: int,
    jpeg_quality: int,
    silence_gap: float,
    frame_format: str,
    interval: float | None,
    cuts: bool,
    cut_threshold: float,
    whisper_model: Path | None,
    lang: str,
    keep_video: bool,
    keep_temp: bool,
    no_learn: bool,
    verbose: bool,
) -> None:
    """Watch & Learn: YouTube video storyboard generator for Vision LLMs.

    Downloads a YouTube video, extracts semantically relevant frames using
    transcript NLP analysis, and compiles them into annotated storyboard grids
    optimized for LLM token efficiency. URL may also be a local video file.

    \b
    Example:
      watch-learn "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o ./output -n 12 -v
      watch-learn "https://youtu.be/VIDEO_ID" --cuts --keep-video
      watch-learn ~/Movies/IMG_1234.MOV --interval 1 --lang tr
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        level=log_level,
    )
    log = logging.getLogger("watch_learn")

    temp_dir = Path(tempfile.mkdtemp(prefix="watch_learn_"))
    t_start = time.time()

    steps = 5 if cuts else 4
    try:
        check_ffmpeg()

        local_path = Path(url).expanduser()
        if local_path.is_file():
            log.info(f"Step 1/{steps} ▸ Reading local video + transcript...")
            from .downloader import load_local_video
            result = load_local_video(local_path.resolve(), temp_dir, groq_api_key, whisper_model, lang)
        else:
            log.info(f"Step 1/{steps} ▸ Downloading transcript + video...")
            from .downloader import download_video
            result = download_video(url, temp_dir, groq_api_key, whisper_model, lang)
        entries = result.transcript_entries
        log.info(
            f"          Transcript: {result.transcript_source} "
            f"({len(entries)} entries)"
        )

        # Each video gets its own subdirectory so storyboards never overwrite each other
        video_output_dir = output_dir / result.video_id
        video_output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"          Output dir: {video_output_dir}")

        from .analyzer import extract_smart_timestamps, fill_gaps, interval_timestamps
        duration = get_video_duration(result.video_path) or (entries[-1].end_sec if entries else 0.0)
        fixed_count = _estimate_fixed_count(duration)
        if interval:
            log.info(f"Step 2/{steps} ▸ Taking a frame every {interval:g}s...")
            timestamps = interval_timestamps(entries, duration, interval, limit=max_frames)
        else:
            log.info(f"Step 2/{steps} ▸ Analyzing transcript for smart timestamps...")
            timestamps = extract_smart_timestamps(
                entries,
                silence_threshold=silence_gap,
                max_timestamps=max_frames,
            )
            if not timestamps and duration:
                # Nothing to steer by (no speech): spread the frame budget evenly
                timestamps = interval_timestamps(entries, duration, max(2.0, duration / max_frames), limit=max_frames)
            elif duration:
                # Long stretches without a keyword or pause would otherwise get no frame at all
                timestamps = fill_gaps(timestamps, entries, duration, max(10.0, duration / max_frames), max_frames)
            log.info(
                f"          Found {len(timestamps)} smart timestamps "
                f"(vs ~{fixed_count} at fixed 30s intervals)"
            )
        if verbose:
            for ts in timestamps:
                log.debug(f"    {ts.time_sec:7.1f}s  [{ts.reason}]  {ts.transcript_text[:60]}")

        log.info(f"Step 3/{steps} ▸ Extracting frames with ffmpeg...")
        from .extractor import extract_frames
        frames = extract_frames(
            result.video_path,
            timestamps,
            temp_dir,
            fmt=frame_format,
        )
        log.info(f"          Extracted {len(frames)} frames successfully")

        log.info(f"Step 4/{steps} ▸ Compiling storyboard grids...")
        from .compiler import compile_storyboards
        boards = compile_storyboards(frames, video_output_dir, jpeg_quality=jpeg_quality)

        report = _report_header(url, result, duration, frames, boards)
        shot_boards: list[Path] = []
        if cuts:
            log.info(f"Step 5/{steps} ▸ Measuring edit rhythm (cuts, shots, audio hits, loudness)...")
            try:
                shot_boards, rhythm = _analyze_edit(
                    result.video_path, entries, duration, cut_threshold, temp_dir,
                    video_output_dir, frame_format, jpeg_quality, log,
                )
                report += rhythm
            except RuntimeError as exc:
                # The storyboard is still useful without the rhythm part
                log.warning(f"Edit-rhythm analysis failed: {exc}")
                report += ["", f"- Kurgu ritmi analizi başarısız: {exc}"]

        (video_output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        if entries:
            (video_output_dir / "transcript.txt").write_text(
                "".join(f"[{_mmss(e.start_sec)}] {e.text}\n" for e in entries), encoding="utf-8",
            )
        if keep_video and result.video_path.parent == temp_dir:
            shutil.copy2(result.video_path, video_output_dir / "video.mp4")

        if not no_learn and entries and timestamps and not interval:
            from .analyzer import update_keyword_store
            new_kw = update_keyword_store(entries, timestamps)
            if new_kw:
                log.info(f"          Keyword store: learned {new_kw} new term(s) from this video")

        elapsed = time.time() - t_start
        log.info("=" * 62)
        log.info(f"  Done in {elapsed:.1f}s")
        log.info(
            f"  Extracted {len(frames)} frames instead of "
            f"~{fixed_count} fixed-interval frames."
        )
        log.info(f"  Compressed into {len(boards)} storyboard grid(s).")
        for b in boards + shot_boards:
            size_kb = b.stat().st_size // 1024
            log.info(f"  → {video_output_dir.name}/{b.name}  ({size_kb} KB)")
        log.info(f"  → {video_output_dir.name}/report.md")
        log.info("=" * 62)

    except ValueError as exc:
        logging.getLogger("watch_learn").error(f"Video error: {exc}")
        raise click.Abort() from exc
    except (OSError, RuntimeError) as exc:
        logging.getLogger("watch_learn").error(str(exc))
        raise click.Abort() from exc
    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _analyze_edit(
    video_path: Path,
    entries: list,
    duration: float,
    cut_threshold: float,
    temp_dir: Path,
    video_output_dir: Path,
    frame_format: str,
    jpeg_quality: int,
    log: logging.Logger,
) -> tuple[list[Path], list[str]]:
    """Cuts → shots → one frame per shot (shots_page_*.jpg) + rhythm/audio report lines."""
    from . import SmartTimestamp
    from .analyzer import text_at
    from .compiler import compile_storyboards
    from .editing import (
        analyze_audio,
        audio_summary,
        burst_summary,
        detect_cuts,
        rhythm_summary,
        shot_table,
        shots_from_cuts,
    )
    from .extractor import extract_frames

    cut_times, bursts = detect_cuts(video_path, cut_threshold)
    shots = shots_from_cuts(cut_times, duration)
    texts = [text_at(entries, s.start_sec + s.duration / 2) for s in shots]
    log.info(f"          {len(shots)} shots, {len(cut_times)} cuts (threshold {cut_threshold:g})")

    # Middle frame of every shot; long shots get one frame per ~LONG_SHOT_STEP seconds, since
    # an animated transition (page turn, wipe, morph) inside them is not detected as a cut.
    samples: list[tuple[float, str]] = []
    for s in shots:
        n = 1 if s.duration < 2 * LONG_SHOT_STEP else round(s.duration / LONG_SHOT_STEP)
        for k in range(n):
            label = f"#{s.index}" if n == 1 else f"#{s.index}.{k + 1}"
            t = s.start_sec + (k + 0.5) * s.duration / n
            samples.append((round(t, 3), f"{label}  {_mmss(t)}  {s.duration:.2f}s"))
    if len(samples) > MAX_SHOT_FRAMES:
        step = len(samples) / MAX_SHOT_FRAMES
        samples = [samples[int(i * step)] for i in range(MAX_SHOT_FRAMES)]
    badges = dict(samples)
    shot_ts = [
        SmartTimestamp(time_sec=t, reason="shot", transcript_text=text_at(entries, t))
        for t in badges
    ]
    shot_dir = temp_dir / "shots"
    shot_dir.mkdir(exist_ok=True)
    shot_frames = extract_frames(video_path, shot_ts, shot_dir, fmt=frame_format)
    for frame in shot_frames:
        frame.badge = badges[round(frame.time_sec, 3)]
    boards = compile_storyboards(shot_frames, video_output_dir, jpeg_quality=jpeg_quality, prefix="shots_page")

    audio = analyze_audio(video_path)
    lines = ["", "## Kurgu ritmi", ""]
    lines += rhythm_summary(shots, duration) if duration else []
    lines += burst_summary(bursts)
    lines.append(f"- Kesim eşiği {cut_threshold:g}: çapraz geçiş ve kararma gibi yavaş geçişler kesim sayılmaz")
    lines.append(
        f"- shots_page: her planın ortasından bir kare, {2 * LONG_SHOT_STEP:g} sn'den uzun planlardan"
        f" ~{LONG_SHOT_STEP:g} sn'de bir kare (#5.2 = 5. planın 2. karesi); toplam {len(shot_frames)} kare"
    )
    lines += ["", "## Ses", ""] + audio_summary(audio, cut_times, duration)
    lines += ["", "## Plan listesi", ""] + shot_table(shots, texts)
    return boards, lines


def _report_header(url: str, result, duration: float, frames: list, boards: list[Path]) -> list[str]:
    from .extractor import get_video_fps

    fps = get_video_fps(result.video_path)
    size = ""
    if frames:
        from PIL import Image
        with Image.open(frames[0].path) as img:
            size = f" · {img.width}x{img.height}"
    return [
        f"# {result.title or result.video_id}",
        "",
        f"- Kaynak: {url}",
        f"- Süre: {_mmss(duration)}{size}" + (f" · {fps:.3g} fps" if fps else ""),
        f"- Transkript: {result.transcript_source} ({len(result.transcript_entries)} satır)",
        f"- Storyboard: {len(frames)} kare, {len(boards)} sayfa (storyboard_page_*.jpg)",
    ]


def _mmss(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m}:{s:02d}"


def _estimate_fixed_count(duration_sec: float | None, interval: int = 30) -> int:
    if duration_sec is None:
        return 100
    return max(1, int(duration_sec / interval))


if __name__ == "__main__":
    cli()
