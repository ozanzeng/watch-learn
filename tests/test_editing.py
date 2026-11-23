"""Tests for edit-rhythm helpers."""

import pytest
from watch_learn import Shot
from watch_learn.editing import _estimate_tempo, _mmss, burst_summary, rhythm_summary, shots_from_cuts


def test_shots_from_cuts_builds_contiguous_shots() -> None:
    shots = shots_from_cuts([2.0, 5.0], duration=10.0)

    assert [(s.index, s.start_sec, s.end_sec) for s in shots] == [
        (1, 0.0, 2.0),
        (2, 2.0, 5.0),
        (3, 5.0, 10.0),
    ]
    assert shots[0].duration == pytest.approx(2.0)


def test_shots_from_cuts_ignores_out_of_range_cuts() -> None:
    shots = shots_from_cuts([-1.0, 0.0, 5.0, 12.0], duration=10.0)

    assert len(shots) == 2
    assert shots[-1].end_sec == pytest.approx(10.0)


def test_burst_summary_empty() -> None:
    assert burst_summary([]) == []


def test_burst_summary_mentions_count_and_truncation() -> None:
    bursts = [(float(i), float(i) + 0.1) for i in range(15)]

    lines = burst_summary(bursts)

    assert len(lines) == 1
    assert "15" in lines[0]
    assert "(+3)" in lines[0]


def test_rhythm_summary_reports_shot_stats() -> None:
    shots = [Shot(index=1, start_sec=0.0, end_sec=2.0), Shot(index=2, start_sec=2.0, end_sec=4.0)]

    lines = rhythm_summary(shots, duration=4.0)

    assert any("Plan sayısı: 2" in line for line in lines)
    assert any("ortalama 2.00 sn" in line for line in lines)
    assert any("Her 10 sn'deki kesim sayısı" in line for line in lines)


def test_mmss_formats_minutes_and_optional_fraction() -> None:
    assert _mmss(65) == "1:05"
    assert _mmss(65.25, frac=True) == "1:05.25"


def test_estimate_tempo_returns_none_for_flat_signal() -> None:
    import numpy as np

    assert _estimate_tempo(np.zeros(2000)) is None
