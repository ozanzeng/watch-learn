"""Tests for post-learning cleanup of the downloaded source video."""

from pathlib import Path

from watch_learn import DownloadResult
from watch_learn.main import _discard_downloaded_video


def make_result(video_path: Path, temp_dir: Path) -> DownloadResult:
    return DownloadResult(
        video_path=video_path,
        transcript_entries=[],
        transcript_source="synthetic",
        temp_dir=temp_dir,
        video_id="test",
        title="Test",
    )


def test_discard_downloaded_video_removes_temp_file(tmp_path: Path) -> None:
    temp_dir = tmp_path / "work"
    temp_dir.mkdir()
    video = temp_dir / "abc_video.mp4"
    video.write_bytes(b"video")

    deleted = _discard_downloaded_video(make_result(video, temp_dir), keep_video=False, keep_temp=False)

    assert deleted is True
    assert not video.exists()


def test_discard_never_touches_local_source(tmp_path: Path) -> None:
    temp_dir = tmp_path / "work"
    temp_dir.mkdir()
    local = tmp_path / "my_footage.mov"
    local.write_bytes(b"local")

    deleted = _discard_downloaded_video(make_result(local, temp_dir), keep_video=False, keep_temp=False)

    assert deleted is False
    assert local.exists()


def test_discard_respects_keep_video(tmp_path: Path) -> None:
    temp_dir = tmp_path / "work"
    temp_dir.mkdir()
    video = temp_dir / "abc_video.mp4"
    video.write_bytes(b"video")

    deleted = _discard_downloaded_video(make_result(video, temp_dir), keep_video=True, keep_temp=False)

    assert deleted is False
    assert video.exists()


def test_discard_respects_keep_temp(tmp_path: Path) -> None:
    temp_dir = tmp_path / "work"
    temp_dir.mkdir()
    video = temp_dir / "abc_video.mp4"
    video.write_bytes(b"video")

    deleted = _discard_downloaded_video(make_result(video, temp_dir), keep_video=False, keep_temp=True)

    assert deleted is False
    assert video.exists()


def test_discard_missing_file_is_not_an_error(tmp_path: Path) -> None:
    temp_dir = tmp_path / "work"
    temp_dir.mkdir()

    deleted = _discard_downloaded_video(
        make_result(temp_dir / "gone.mp4", temp_dir), keep_video=False, keep_temp=False
    )

    assert deleted is False
