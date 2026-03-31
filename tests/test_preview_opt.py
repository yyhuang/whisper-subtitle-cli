"""Tests for --preview-opt flag (non-interactive preview)."""
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from click.testing import CliRunner
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import main


def _mock_downloader_with_subs(subs=None, meta=None):
    """Helper to create a mock downloader with subtitle data."""
    if subs is None:
        subs = {'en': {'name': 'English'}, 'fr': {'name': 'French'}}
    if meta is None:
        meta = {'title': 'Test Video', 'channel': 'Test Channel'}
    mock = MagicMock()
    mock_instance = MagicMock()
    mock.return_value = mock_instance
    mock_instance.get_available_subtitles.return_value = (subs, meta)
    mock_instance.get_video_info.return_value = {
        'video_id': 'abc123',
        'upload_date': '20200101',
    }
    return mock, mock_instance


def _get_command_line(output: str) -> str:
    """Extract the command line from test output (line containing 'main.py')."""
    for line in output.strip().split('\n'):
        if 'main.py' in line:
            return line
    return ''


def _get_json_from_output(output: str) -> dict:
    """Extract and parse the JSON line from mixed stdout/stderr output.

    CliRunner mixes stdout and stderr. The JSON line starts with '{'.
    """
    for line in output.strip().split('\n'):
        line = line.strip()
        if line.startswith('{'):
            return json.loads(line)
    raise ValueError(f"No JSON line found in output:\n{output}")


class TestPreviewOptImpliesPreview:
    """--preview-opt should imply --preview mode."""

    @patch('main.VideoDownloader')
    def test_preview_opt_without_preview_flag(self, mock_cls):
        """--preview-opt alone (without --preview) should work in preview mode."""
        mock_cls, mock_instance = _mock_downloader_with_subs()

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', '1'],
            )

        assert result.exit_code == 0, result.output
        cmd = _get_command_line(result.output)
        assert '--subtitle 1' in cmd

    @patch('main.VideoDownloader')
    def test_preview_opt_with_preview_flag(self, mock_cls):
        """--preview-opt with --preview should also work."""
        mock_cls, mock_instance = _mock_downloader_with_subs()

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview', '--preview-opt', '1'],
            )

        assert result.exit_code == 0, result.output
        cmd = _get_command_line(result.output)
        assert '--subtitle 1' in cmd


class TestPreviewOptSelectSubtitle:
    """--preview-opt N selects subtitle by index without prompting."""

    @patch('main.VideoDownloader')
    def test_select_subtitle_1(self, mock_cls):
        """--preview-opt 1 should emit command with --subtitle 1."""
        mock_cls, _ = _mock_downloader_with_subs()

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', '1'],
            )

        assert result.exit_code == 0, result.output
        cmd = _get_command_line(result.output)
        assert '--subtitle 1' in cmd
        assert '-y' in cmd

    @patch('main.VideoDownloader')
    def test_select_subtitle_2(self, mock_cls):
        """--preview-opt 2 should emit command with --subtitle 2."""
        mock_cls, _ = _mock_downloader_with_subs()

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', '2'],
            )

        assert result.exit_code == 0, result.output
        cmd = _get_command_line(result.output)
        assert '--subtitle 2' in cmd

    @patch('main.VideoDownloader')
    def test_no_interactive_prompt(self, mock_cls):
        """--preview-opt should NOT prompt for input (no stdin needed)."""
        mock_cls, _ = _mock_downloader_with_subs()

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            # No input= provided — should not hang or error
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', '1'],
            )

        assert result.exit_code == 0, result.output


class TestPreviewOptTranscribe:
    """--preview-opt 0 selects transcribe."""

    @patch('main.VideoDownloader')
    def test_transcribe_with_subtitles_available(self, mock_cls):
        """--preview-opt 0 should emit transcribe command even when subtitles exist."""
        mock_cls, _ = _mock_downloader_with_subs()

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', '0'],
            )

        assert result.exit_code == 0, result.output
        cmd = _get_command_line(result.output)
        assert '--subtitle 0' in cmd

    @patch('main.VideoDownloader')
    def test_transcribe_no_subtitles(self, mock_cls):
        """--preview-opt 0 should emit transcribe command when no subtitles."""
        mock_cls, _ = _mock_downloader_with_subs(subs={})

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', '0'],
            )

        assert result.exit_code == 0, result.output
        cmd = _get_command_line(result.output)
        assert '--subtitle 0' in cmd


class TestPreviewOptSkip:
    """--preview-opt S skips the video."""

    @patch('main.VideoDownloader')
    def test_skip_emits_nothing(self, mock_cls):
        """--preview-opt S should emit no command to stdout."""
        mock_cls, _ = _mock_downloader_with_subs()

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', 'S'],
            )

        assert result.exit_code == 0, result.output
        assert 'main.py' not in result.output

    @patch('main.VideoDownloader')
    def test_skip_case_insensitive(self, mock_cls):
        """--preview-opt s (lowercase) should also skip."""
        mock_cls, _ = _mock_downloader_with_subs()

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', 's'],
            )

        assert result.exit_code == 0, result.output
        assert 'main.py' not in result.output

    @patch('main.VideoDownloader')
    def test_skip_no_subtitles(self, mock_cls):
        """--preview-opt S with no subtitles should also skip cleanly."""
        mock_cls, _ = _mock_downloader_with_subs(subs={})

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', 'S'],
            )

        assert result.exit_code == 0, result.output
        assert 'main.py' not in result.output


class TestPreviewOptList:
    """--preview-opt L outputs JSON subtitle list."""

    @patch('main.VideoDownloader')
    def test_list_outputs_json(self, mock_cls):
        """--preview-opt L should output valid JSON to stdout."""
        mock_cls, _ = _mock_downloader_with_subs()

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', 'L'],
            )

        assert result.exit_code == 0, result.output
        data = _get_json_from_output(result.output)
        assert 'subtitles' in data

    @patch('main.VideoDownloader')
    def test_list_contains_subtitle_entries(self, mock_cls):
        """--preview-opt L should include subtitle index, lang, and name."""
        mock_cls, _ = _mock_downloader_with_subs()

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', 'L'],
            )

        data = _get_json_from_output(result.output)
        assert len(data['subtitles']) == 2
        assert data['subtitles'][0]['index'] == 1
        assert data['subtitles'][0]['lang'] == 'en'
        assert data['subtitles'][0]['name'] == 'English'
        assert data['subtitles'][1]['index'] == 2
        assert data['subtitles'][1]['lang'] == 'fr'
        assert data['subtitles'][1]['name'] == 'French'

    @patch('main.VideoDownloader')
    def test_list_contains_video_info(self, mock_cls):
        """--preview-opt L should include video title and channel."""
        mock_cls, _ = _mock_downloader_with_subs()

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', 'L'],
            )

        data = _get_json_from_output(result.output)
        assert data['video_title'] == 'Test Video'
        assert data['channel'] == 'Test Channel'
        assert data['url'] == 'https://youtube.com/watch?v=abc123'

    @patch('main.VideoDownloader')
    def test_list_contains_can_transcribe(self, mock_cls):
        """--preview-opt L should include can_transcribe flag."""
        mock_cls, _ = _mock_downloader_with_subs()

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', 'L'],
            )

        data = _get_json_from_output(result.output)
        assert data['can_transcribe'] is True

    @patch('main.VideoDownloader')
    def test_list_empty_subtitles(self, mock_cls):
        """--preview-opt L with no subtitles should return empty list."""
        mock_cls, _ = _mock_downloader_with_subs(subs={})

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', 'L'],
            )

        data = _get_json_from_output(result.output)
        assert data['subtitles'] == []
        assert data['can_transcribe'] is True

    @patch('main.VideoDownloader')
    def test_list_case_insensitive(self, mock_cls):
        """--preview-opt l (lowercase) should also list."""
        mock_cls, _ = _mock_downloader_with_subs()

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', 'l'],
            )

        assert result.exit_code == 0, result.output
        data = _get_json_from_output(result.output)
        assert 'subtitles' in data


class TestPreviewOptValidation:
    """--preview-opt should validate its argument."""

    @patch('main.VideoDownloader')
    def test_invalid_index_out_of_range(self, mock_cls):
        """--preview-opt 5 with only 2 subtitles should error."""
        mock_cls, _ = _mock_downloader_with_subs()

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', '5'],
            )

        assert result.exit_code != 0

    @patch('main.VideoDownloader')
    def test_index_with_no_subtitles(self, mock_cls):
        """--preview-opt 1 with no subtitles should error."""
        mock_cls, _ = _mock_downloader_with_subs(subs={})

        runner = CliRunner()
        with patch('main.VideoDownloader', mock_cls):
            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview-opt', '1'],
            )

        assert result.exit_code != 0

    def test_invalid_value(self):
        """--preview-opt X (invalid) should error."""
        runner = CliRunner()
        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview-opt', 'X'],
        )

        assert result.exit_code != 0
