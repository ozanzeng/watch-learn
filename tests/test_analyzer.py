"""Tests for smart timestamp selection and the self-learning keyword store."""

import json
from pathlib import Path

import pytest
from watch_learn import KeywordEntry, KeywordStore, SmartTimestamp, TranscriptEntry
from watch_learn.analyzer import (
    _deduplicate_timestamps,
    _extract_silence_timestamps,
    _reason_priority,
    _silence_gap_seconds,
    _spread_over_time,
    extract_smart_timestamps,
    fill_gaps,
    interval_timestamps,
    load_keyword_store,
    save_keyword_store,
    text_at,
)


def make_entries(*spans: tuple[float, float, str]) -> list[TranscriptEntry]:
    return [TranscriptEntry(start_sec=s, end_sec=e, text=t) for s, e, t in spans]


def test_text_at_returns_speech_inside_entry() -> None:
    entries = make_entries((0.0, 2.0, "first"), (5.0, 7.0, "second"))

    assert text_at(entries, 1.0) == "first"
    assert text_at(entries, 6.0) == "second"
    assert text_at(entries, 3.0) == ""


def test_text_at_handles_empty_entries() -> None:
    assert text_at([], 1.0) == ""


def test_silence_gap_detection_uses_midpoint() -> None:
    entries = make_entries((0.0, 2.0, "a"), (12.0, 14.0, "b"))

    gaps = _extract_silence_timestamps(entries, threshold=5.0)

    assert len(gaps) == 1
    assert gaps[0].time_sec == pytest.approx(7.0)
    assert gaps[0].reason == "silence_gap:10.0s"


def test_short_silence_is_ignored() -> None:
    entries = make_entries((0.0, 2.0, "a"), (4.0, 6.0, "b"))

    assert _extract_silence_timestamps(entries, threshold=5.0) == []


def test_deduplicate_prefers_keyword_over_silence() -> None:
    timestamps = [
        SmartTimestamp(time_sec=10.0, reason="silence_gap:6.0s", transcript_text=""),
        SmartTimestamp(time_sec=10.5, reason="keyword:look at", transcript_text="look at this"),
    ]

    kept = _deduplicate_timestamps(timestamps, min_spacing=2.0)

    assert len(kept) == 1
    assert kept[0].reason == "keyword:look at"


def test_deduplicate_keeps_spaced_timestamps() -> None:
    timestamps = [
        SmartTimestamp(time_sec=1.0, reason="silence_gap:6.0s", transcript_text=""),
        SmartTimestamp(time_sec=10.0, reason="silence_gap:6.0s", transcript_text=""),
    ]

    assert len(_deduplicate_timestamps(timestamps, min_spacing=2.0)) == 2


def test_spread_over_time_caps_and_preserves_end() -> None:
    timestamps = [
        SmartTimestamp(time_sec=float(t), reason="keyword:x", transcript_text="")
        for t in range(0, 100, 5)
    ]

    spread = _spread_over_time(timestamps, limit=4)

    assert len(spread) == 4
    assert spread[0].time_sec == 0.0
    assert spread[-1].time_sec >= 70.0


def test_interval_timestamps_respects_limit() -> None:
    entries = make_entries((0.0, 60.0, "talk"))

    timestamps = interval_timestamps(entries, duration=60.0, interval=1.0, limit=10)

    assert len(timestamps) == 10
    assert timestamps[0].time_sec == pytest.approx(0.5)
    assert all(ts.reason == "interval:1s" for ts in timestamps)


def test_interval_timestamps_without_limit() -> None:
    timestamps = interval_timestamps([], duration=10.0, interval=2.0)

    assert [ts.time_sec for ts in timestamps] == [1.0, 3.0, 5.0, 7.0, 9.0]


def test_fill_gaps_adds_frames_in_long_stretches() -> None:
    timestamps = [SmartTimestamp(time_sec=50.0, reason="keyword:x", transcript_text="")]

    filled = fill_gaps(timestamps, [], duration=100.0, max_gap=20.0, limit=10)

    assert len(filled) > 1
    assert filled == sorted(filled, key=lambda t: t.time_sec)
    assert any(ts.reason.startswith("gap_fill:") for ts in filled)


def test_fill_gaps_returns_input_when_no_room() -> None:
    timestamps = [SmartTimestamp(time_sec=50.0, reason="keyword:x", transcript_text="")]

    assert fill_gaps(timestamps, [], duration=100.0, max_gap=20.0, limit=1) == timestamps


def test_reason_priority_order() -> None:
    assert _reason_priority("combined") == 3
    assert _reason_priority("keyword:look at") == 2
    assert _reason_priority("spacy:deictic_verb") == 2
    assert _reason_priority("silence_gap:6.0s") == 1


def test_silence_gap_seconds_parsing() -> None:
    assert _silence_gap_seconds("silence_gap:6.5s") == pytest.approx(6.5)
    assert _silence_gap_seconds("keyword:x") == 0.0
    assert _silence_gap_seconds("silence_gap:nope") == 0.0


def test_keyword_store_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "store.json"
    store = KeywordStore(
        version=1,
        keywords=[KeywordEntry(phrase="diagram", weight=1.0, source="builtin")],
        domain_clusters={"data": ["chart", "table"]},
    )

    save_keyword_store(store, path)
    loaded = load_keyword_store(path)

    assert loaded.version == 1
    assert loaded.domain_clusters == {"data": ["chart", "table"]}
    assert any(k.phrase == "diagram" for k in loaded.keywords)


def test_keyword_store_injects_missing_builtins(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"version": 1, "keywords": []}), encoding="utf-8")

    loaded = load_keyword_store(path)

    assert any(k.phrase == "diagram" for k in loaded.keywords)
    assert all(k.source == "builtin" for k in loaded.keywords)


def test_keyword_store_corrupt_file_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    loaded = load_keyword_store(path)

    assert loaded.keywords


def test_extract_smart_timestamps_finds_keyword(tmp_path: Path) -> None:
    store_path = tmp_path / "store.json"
    entries = make_entries(
        (0.0, 2.0, "intro"),
        (3.0, 5.0, "here we have the chart"),
        (20.0, 22.0, "outro"),
    )

    timestamps = extract_smart_timestamps(entries, store_path=store_path)

    assert any(ts.reason.startswith("keyword:") for ts in timestamps)


def test_extract_smart_timestamps_empty_input() -> None:
    assert extract_smart_timestamps([]) == []
