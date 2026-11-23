"""Tests for storyboard layout and compilation."""

from pathlib import Path

import pytest
from PIL import Image
from watch_learn import ExtractedFrame
from watch_learn.compiler import (
    _frame_aspect,
    _load_and_fit_frame,
    compile_storyboards,
    compute_grid_layout,
)


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, (1, 1, 1280)),
        (1, (1, 1, 1280)),
        (2, (2, 1, 1024)),
        (3, (2, 2, 960)),
        (4, (2, 2, 960)),
        (5, (3, 2, 800)),
        (6, (3, 2, 800)),
        (7, (3, 3, 720)),
        (9, (3, 3, 720)),
        (10, (4, 3, 640)),
        (12, (4, 3, 640)),
        (20, (3, 3, 720)),
    ],
)
def test_compute_grid_layout_landscape(count: int, expected: tuple[int, int, int]) -> None:
    assert compute_grid_layout(count) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, (1, 1, 720)),
        (2, (2, 1, 600)),
        (5, (4, 2, 480)),
        (8, (4, 2, 480)),
        (9, (4, 3, 480)),
    ],
)
def test_compute_grid_layout_portrait(count: int, expected: tuple[int, int, int]) -> None:
    assert compute_grid_layout(count, portrait=True) == expected


def _write_frame(path: Path, size: tuple[int, int] = (320, 180), color=(10, 20, 30)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_frame_aspect_detects_portrait(tmp_path: Path) -> None:
    portrait = _write_frame(tmp_path / "p.png", size=(90, 160))
    landscape = _write_frame(tmp_path / "l.png", size=(160, 90))

    assert _frame_aspect(portrait) == pytest.approx(160 / 90)
    assert _frame_aspect(landscape) == pytest.approx(90 / 160)


def test_frame_aspect_unreadable_file_defaults_to_portrait(tmp_path: Path) -> None:
    assert _frame_aspect(tmp_path / "missing.png") == pytest.approx(9 / 16)


def test_load_and_fit_frame_letterboxes_without_cropping(tmp_path: Path) -> None:
    path = _write_frame(tmp_path / "wide.png", size=(200, 100))

    cell = _load_and_fit_frame(path, target_w=100, target_h=100)

    assert cell.size == (100, 100)
    # Landscape source is letterboxed vertically: top/bottom rows stay black.
    assert cell.getpixel((50, 0)) == (0, 0, 0)
    assert cell.getpixel((50, 50)) == (10, 20, 30)


def test_load_and_fit_frame_returns_placeholder_for_broken_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")

    cell = _load_and_fit_frame(broken, target_w=64, target_h=36)

    assert cell.size == (64, 36)
    assert cell.getpixel((0, 0)) == (40, 40, 40)


def test_compile_storyboards_writes_pages(tmp_path: Path) -> None:
    frames = [
        ExtractedFrame(time_sec=float(i), path=_write_frame(tmp_path / f"f{i}.png"), transcript_text="")
        for i in range(10)
    ]

    pages = compile_storyboards(frames, tmp_path / "out", jpeg_quality=80)

    assert len(pages) == 2
    assert all(p.exists() for p in pages)
    assert pages[0].name == "storyboard_page_001.jpg"


def test_compile_storyboards_removes_stale_pages(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    stale = out / "storyboard_page_099.jpg"
    stale.write_bytes(b"old")
    frames = [ExtractedFrame(time_sec=0.0, path=_write_frame(tmp_path / "f.png"), transcript_text="")]

    compile_storyboards(frames, out)

    assert not stale.exists()


def test_compile_storyboards_empty_returns_nothing(tmp_path: Path) -> None:
    assert compile_storyboards([], tmp_path / "out") == []
