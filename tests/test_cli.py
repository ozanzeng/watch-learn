"""CLI smoke tests (no network, no ffmpeg work)."""

from click.testing import CliRunner
from watch_learn import __version__
from watch_learn.main import cli


def test_help_lists_core_options() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "--max-frames" in result.output
    assert "--cuts" in result.output


def test_version_flag() -> None:
    result = CliRunner().invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_missing_url_is_a_usage_error() -> None:
    result = CliRunner().invoke(cli, [])

    assert result.exit_code != 0
