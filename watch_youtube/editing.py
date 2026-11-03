"""Edit-rhythm analysis: scene cuts, shot lengths, audio hits and loudness via ffmpeg + numpy."""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import Shot

logger = logging.getLogger(__name__)

_LUFS_RE = re.compile(r"I:\s+(-?[\d.]+) LUFS")
_LRA_RE = re.compile(r"LRA:\s+([\d.]+) LU")
_PEAK_RE = re.compile(r"Peak:\s+(-?[\d.inf]+) dBFS")

ANALYSIS_FPS = 30         # cut detection resamples every video to this rate (VFR-safe)
THUMB_W, THUMB_H = 64, 36
CUT_WINDOW = 2            # neighbour frames on each side for the adaptive cut test

ONSET_SR = 11025          # onset detection does not need more bandwidth
ONSET_HOP = 128           # ~12 ms per envelope step
ONSET_FFT = 512
# Spectral flux jumps as soon as a hit enters the analysis window, i.e. about one
# window before the hit itself; shift onset times by this much to line them up.
ONSET_LEAD = (ONSET_FFT - ONSET_HOP / 2) / ONSET_SR
BEAT_TOLERANCE = 0.07     # a cut within ±70 ms (~2 frames at 30 fps) counts as "on the hit"


@dataclass
class AudioStats:
    lufs: float | None = None
    lra: float | None = None
    peak_dbfs: float | None = None
    tempo_bpm: float | None = None
    onsets: list[float] | None = None


# ---------------------------------------------------------------------------
# Cuts and shots
# ---------------------------------------------------------------------------

def detect_cuts(
    video_path: Path,
    threshold: float = 3.0,
    min_score: float = 0.04,
    burst_gap: float = 0.25,
) -> tuple[list[float], list[tuple[float, float]]]:
    """Hard-cut times in seconds + burst spans.

    Each frame's colour difference to the previous one (64x36 RGB thumbnails at 30 fps)
    is compared with its neighbours' (±2 frames): a cut is a frame that changes
    `threshold` times more than its neighbours and at least `min_score` overall. This
    keeps fast motion, which is high on every frame, from counting as a cut and still
    catches cuts between shots of similar brightness (ffmpeg's scene score is mostly luma).

    Detections closer than `burst_gap` to the previous one (flash frames, strobes,
    animated transitions) are merged into one cut at the start of the run; runs of 3+
    come back as (start, end) spans. Dissolves and fades are gradual and are missed.
    """
    diffs = _frame_differences(video_path)
    raw: list[float] = []
    for i in range(1, len(diffs)):
        score = diffs[i]
        if score < min_score:
            continue
        neighbours = diffs[max(1, i - CUT_WINDOW):i] + diffs[i + 1:i + 1 + CUT_WINDOW]
        base = sum(neighbours) / len(neighbours) if neighbours else 0.0
        if score >= threshold * max(base, 0.004):
            raw.append(i / ANALYSIS_FPS)

    runs: list[list[float]] = []
    for t in raw:
        if runs and t - runs[-1][-1] < burst_gap:
            runs[-1].append(t)
        else:
            runs.append([t])
    cuts = [run[0] for run in runs]
    bursts = [(run[0], run[-1]) for run in runs if len(run) >= 3]
    return cuts, bursts


def _frame_differences(video_path: Path) -> list[float]:
    """Mean absolute RGB difference (0..1) of every analysis frame to the one before it."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path), "-an",
        "-vf", f"fps={ANALYSIS_FPS},scale={THUMB_W}:{THUMB_H}:flags=area,format=rgb24",
        "-f", "rawvideo", "-",
    ]
    frame_bytes = THUMB_W * THUMB_H * 3
    diffs: list[float] = [0.0]
    prev = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        cur = np.frombuffer(buf, dtype=np.uint8).astype(np.int16)
        if prev is not None:
            diffs.append(float(np.abs(cur - prev).mean()) / 255.0)
        prev = cur
    _, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"frame analysis failed: {err.decode(errors='replace').strip()[-200:]}")
    return diffs


def shots_from_cuts(cuts: list[float], duration: float) -> list[Shot]:
    bounds = [0.0] + [c for c in cuts if 0.0 < c < duration] + [duration]
    return [Shot(index=i + 1, start_sec=a, end_sec=b) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def analyze_audio(video_path: Path) -> AudioStats:
    stats = AudioStats()
    stats.lufs, stats.lra, stats.peak_dbfs = _loudness(video_path)
    samples = _decode_mono(video_path, ONSET_SR)
    if samples is not None and len(samples) > ONSET_SR * 2:
        env = _onset_envelope(samples)
        stats.onsets = _pick_onsets(env)
        stats.tempo_bpm = _estimate_tempo(env)
    return stats


def _loudness(video_path: Path) -> tuple[float | None, float | None, float | None]:
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(video_path), "-vn",
        "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    summary = result.stderr[result.stderr.rfind("Summary:"):] if "Summary:" in result.stderr else ""

    def grab(rx: re.Pattern) -> float | None:
        m = rx.search(summary)
        try:
            return float(m.group(1)) if m else None
        except ValueError:
            return None

    return grab(_LUFS_RE), grab(_LRA_RE), grab(_PEAK_RE)


def _decode_mono(video_path: Path, sr: int) -> np.ndarray | None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return None
    return np.frombuffer(result.stdout, dtype=np.float32)


def _onset_envelope(x: np.ndarray) -> np.ndarray:
    """Spectral flux: summed positive change of the log spectrum, normalised to ~0..1."""
    window = np.hanning(ONSET_FFT).astype(np.float32)
    n_frames = 1 + (len(x) - ONSET_FFT) // ONSET_HOP
    flux = np.zeros(n_frames, dtype=np.float32)
    prev = None
    chunk = 4096
    for start in range(0, n_frames, chunk):
        idx = np.arange(start, min(n_frames, start + chunk))[:, None] * ONSET_HOP + np.arange(ONSET_FFT)
        spec = np.log1p(100.0 * np.abs(np.fft.rfft(x[idx] * window, axis=1)))
        joined = spec if prev is None else np.vstack([prev, spec])
        diff = np.maximum(0.0, np.diff(joined, axis=0)).sum(axis=1)
        # diff[k] is the change into frame start+k (or start+1+k for the very first chunk)
        first = start if prev is not None else start + 1
        flux[first:first + len(diff)] = diff
        prev = spec[-1:]
    scale = np.percentile(flux, 99) or 1.0
    return np.clip(flux / scale, 0.0, 1.5)


def _pick_onsets(env: np.ndarray, delta: float = 0.15, min_gap: float = 0.15) -> list[float]:
    """Peaks that are the local maximum (±70 ms) and stand out from the local mean (±0.5 s).

    Times are shifted by ONSET_LEAD so they mark the hit, not the window that first saw it.
    """
    step = ONSET_HOP / ONSET_SR
    w_max = max(1, int(0.07 / step))
    w_avg = max(1, int(0.5 / step))
    kernel = np.ones(2 * w_avg + 1, dtype=np.float32) / (2 * w_avg + 1)
    local_mean = np.convolve(env, kernel, mode="same")

    onsets: list[float] = []
    for i in range(w_max, len(env) - w_max):
        v = env[i]
        if v < 0.1 or v < local_mean[i] + delta:
            continue
        if v != env[i - w_max:i + w_max + 1].max():
            continue
        t = i * step + ONSET_LEAD
        if not onsets or t - onsets[-1] >= min_gap:
            onsets.append(t)
    return onsets


def _estimate_tempo(env: np.ndarray, lo_bpm: float = 70.0, hi_bpm: float = 180.0) -> float | None:
    """Autocorrelation peak of the onset envelope within lo..hi BPM (octave errors possible)."""
    step = ONSET_HOP / ONSET_SR
    e = env - env.mean()
    n = len(e)
    if n < 4:
        return None
    spec = np.fft.rfft(e, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    lag_lo = int(60.0 / hi_bpm / step)
    lag_hi = min(n - 1, int(60.0 / lo_bpm / step))
    if lag_hi <= lag_lo:
        return None
    lag = lag_lo + int(np.argmax(ac[lag_lo:lag_hi + 1]))
    # Parabolic interpolation around the peak: the envelope step alone is ~12 ms
    if 0 < lag < n - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = a - 2 * b + c
        if denom:
            lag = lag + 0.5 * (a - c) / denom
    return round(60.0 / (lag * step), 1)


# ---------------------------------------------------------------------------
# Cut ↔ audio alignment
# ---------------------------------------------------------------------------

def cut_alignment(cuts: list[float], onsets: list[float], duration: float) -> tuple[float, float] | None:
    """(share of cuts within ±BEAT_TOLERANCE of an audio hit, share expected by chance)."""
    if not cuts or not onsets or duration <= 0:
        return None
    on = np.asarray(onsets)
    hits = sum(1 for c in cuts if np.min(np.abs(on - c)) <= BEAT_TOLERANCE)

    # Chance level = part of the timeline covered by the ±tolerance windows around hits
    covered, last_end = 0.0, -1.0
    for o in onsets:
        a, b = max(0.0, o - BEAT_TOLERANCE), min(duration, o + BEAT_TOLERANCE)
        covered += max(0.0, b - max(a, last_end))
        last_end = max(last_end, b)
    return hits / len(cuts), covered / duration


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def burst_summary(bursts: list[tuple[float, float]], burst_gap: float = 0.25) -> list[str]:
    if not bursts:
        return []
    spans = " · ".join(f"{_mmss(a, frac=True)}-{_mmss(b, frac=True)}" for a, b in bursts[:12])
    more = f" (+{len(bursts) - 12})" if len(bursts) > 12 else ""
    return [
        f"- Hızlı hareket ya da efektli geçiş: {len(bursts)} bölge, her biri tek kesim sayıldı"
        f" ({burst_gap:g} sn'den sık ardışık tespitler): {spans}{more}"
    ]


def rhythm_summary(shots: list[Shot], duration: float) -> list[str]:
    """Markdown lines describing shot count, lengths and how the pace changes."""
    lengths = np.array([s.duration for s in shots])
    lines = [
        f"- Plan sayısı: {len(shots)} · dakikada {60 * max(0, len(shots) - 1) / duration:.1f} kesim",
        f"- Plan süresi: ortalama {lengths.mean():.2f} sn · medyan {np.median(lengths):.2f} sn"
        f" · en kısa {lengths.min():.2f} sn · en uzun {lengths.max():.2f} sn",
    ]
    edges = [0, 0.5, 1, 2, 4, 8, float("inf")]
    labels = ["<0.5", "0.5-1", "1-2", "2-4", "4-8", "8+"]
    counts = np.histogram(lengths, bins=edges)[0]
    lines.append("- Dağılım (sn): " + " · ".join(f"{l}: {c}" for l, c in zip(labels, counts)))

    window = 30.0 if duration > 120 else 10.0
    parts = []
    t = 0.0
    while t < duration:
        n = sum(1 for s in shots[1:] if t <= s.start_sec < t + window)
        parts.append(f"{_mmss(t)} {n}")
        t += window
    lines.append(f"- Her {window:g} sn'deki kesim sayısı: " + " · ".join(parts))
    return lines


def audio_summary(stats: AudioStats, cuts: list[float], duration: float) -> list[str]:
    lines = []
    if stats.lufs is not None:
        extra = ""
        if stats.lra is not None:
            extra += f", LRA {stats.lra:g} LU"
        if stats.peak_dbfs is not None:
            extra += f", tepe {stats.peak_dbfs:g} dBFS"
        lines.append(f"- Ses yüksekliği: {stats.lufs:g} LUFS{extra}")
    if stats.tempo_bpm:
        lines.append(f"- Tahmini tempo: ~{stats.tempo_bpm:g} BPM (yarısı/iki katı da olabilir)")
    if stats.onsets is not None:
        lines.append(f"- Ses vurgusu: {len(stats.onsets)} (müzik vuruşu, efekt ya da hece başı)")
        aligned = cut_alignment(cuts, stats.onsets, duration)
        if aligned:
            share, chance = aligned
            lines.append(
                f"- Ses vurgusuna ±{BEAT_TOLERANCE * 1000:.0f} ms içinde denk gelen kesim: %{100 * share:.0f}"
                f" (rastgele kesimde beklenen %{100 * chance:.0f}; belirgin fark yoksa kesimler vuruşa oturtulmamış)"
            )
    return lines


def shot_table(shots: list[Shot], texts: list[str]) -> list[str]:
    lines = ["| # | Başlangıç | Süre | Konuşma |", "|---|---|---|---|"]
    for shot, text in zip(shots, texts):
        clean = text.replace("|", "/")[:80]
        lines.append(f"| {shot.index} | {_mmss(shot.start_sec, frac=True)} | {shot.duration:.2f} sn | {clean} |")
    return lines


def _mmss(t: float, frac: bool = False) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:05.2f}" if frac else f"{int(m)}:{int(s):02d}"
