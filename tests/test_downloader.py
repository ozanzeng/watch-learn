"""Tests for transcript parsing and video ID handling in downloader.py."""

from pathlib import Path

import pytest
from watch_learn import TranscriptEntry
from watch_learn.downloader import (
    _is_rolling_caption,
    _parse_srt,
    _parse_vtt,
    _timecode_to_seconds,
    extract_video_id,
    local_video_id,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://example.com/not-a-video", "unknown"),
    ],
)
def test_extract_video_id(url: str, expected: str) -> None:
    assert extract_video_id(url) == expected


@pytest.mark.parametrize(
    ("timecode", "expected"),
    [
        ("00:00:01.500", 1.5),
        ("00:01:02,250", 62.25),
        ("01:02.500", 62.5),
        ("00:00:00.000", 0.0),
    ],
)
def test_timecode_to_seconds(timecode: str, expected: float) -> None:
    assert _timecode_to_seconds(timecode) == pytest.approx(expected)


def test_timecode_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        _timecode_to_seconds("not a timecode")


def test_parse_srt(tmp_path: Path) -> None:
    srt = tmp_path / "sample.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nHello <i>world</i>\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nSecond line\n",
        encoding="utf-8",
    )

    entries = _parse_srt(srt)

    assert [e.text for e in entries] == ["Hello world", "Second line"]
    assert entries[0].start_sec == pytest.approx(1.0)
    assert entries[1].end_sec == pytest.approx(6.0)


def test_parse_srt_without_cue_numbers(tmp_path: Path) -> None:
    srt = tmp_path / "no_numbers.srt"
    srt.write_text(
        "00:00:01,000 --> 00:00:02,000\nOnly text\n",
        encoding="utf-8",
    )

    assert [e.text for e in _parse_srt(srt)] == ["Only text"]


def test_parse_vtt_merges_rolling_captions(tmp_path: Path) -> None:
    vtt = tmp_path / "sample.vtt"
    vtt.write_text(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:03.000\nhere we have\n\n"
        "00:00:03.000 --> 00:00:05.000\nhere we have a chart\n",
        encoding="utf-8",
    )

    entries = _parse_vtt(vtt)

    assert len(entries) == 1
    assert entries[0].text == "here we have a chart"
    assert entries[0].end_sec == pytest.approx(5.0)


def test_parse_vtt_without_header_falls_back_to_cues(tmp_path: Path) -> None:
    vtt = tmp_path / "headerless.vtt"
    vtt.write_text(
        "00:00:01.000 --> 00:00:02.000\nstill readable\n",
        encoding="utf-8",
    )

    entries = _parse_vtt(vtt)

    assert [e.text for e in entries] == ["still readable"]


def test_rolling_caption_detection() -> None:
    previous = TranscriptEntry(start_sec=0.0, end_sec=2.0, text="look at this")
    assert _is_rolling_caption(previous, "look at this diagram", 2.2)
    assert not _is_rolling_caption(previous, "a completely new sentence", 2.2)
    # Too far away to be the same rolling caption
    assert not _is_rolling_caption(previous, "look at this diagram", 10.0)


def test_local_video_id_is_folder_safe() -> None:
    assert local_video_id(Path("/tmp/My Clip (final)!.MOV")) == "My_Clip_final"
    assert local_video_id(Path("/tmp/🎬.mp4")) == "local"
