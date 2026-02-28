"""Tests for --subtitle and --preview CLI flags."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from click.testing import CliRunner
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import main


class TestSubtitleFlag:
    """Tests for --subtitle N flag."""

    @patch('main.VideoDownloader')
    @patch('main.AudioExtractor')
    @patch('main.Transcriber')
    @patch('main.SubtitleWriter')
    def test_subtitle_0_skips_subtitle_check(
        self, mock_writer, mock_transcriber, mock_extractor, mock_downloader
    ):
        """--subtitle 0 should skip subtitle check and go straight to transcription."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_downloader_instance = MagicMock()
            mock_downloader.return_value = mock_downloader_instance
            mock_downloader_instance.download.return_value = {
                'file_path': f'{tmpdir}/abc123.mp4',
                'title': 'Test Video',
                'video_id': 'abc123',
                'duration': 60.0,
                'platform': 'youtube',
            }
            Path(f'{tmpdir}/abc123.mp4').touch()

            mock_extractor_instance = MagicMock()
            mock_extractor.return_value = mock_extractor_instance

            mock_transcriber_instance = MagicMock()
            mock_transcriber.return_value = mock_transcriber_instance
            mock_transcriber_instance.transcribe.return_value = [
                {'start': 0.0, 'end': 1.0, 'text': 'Test'}
            ]

            mock_writer.return_value = MagicMock()

            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--subtitle', '0', '--output', tmpdir],
                input='n\n',
            )

            assert result.exit_code == 0, result.output
            # Should NOT call get_available_subtitles when --subtitle 0 is given
            mock_downloader_instance.get_available_subtitles.assert_not_called()
            # Should still download and transcribe
            mock_downloader_instance.download.assert_called_once()

    @patch('main.VideoDownloader')
    @patch('main.SubtitleWriter')
    def test_subtitle_n_selects_nth_without_prompting(self, mock_writer, mock_downloader):
        """--subtitle 1 should download the first subtitle without interactive prompt."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_downloader_instance = MagicMock()
            mock_downloader.return_value = mock_downloader_instance
            mock_downloader_instance.get_available_subtitles.return_value = {
                'en': {'name': 'English'},
                'es': {'name': 'Spanish'},
            }
            mock_downloader_instance.get_video_info.return_value = {
                'video_id': 'abc123',
                'title': 'Test Video',
                'duration': 60.0,
                'platform': 'youtube',
                'upload_date': '20200101',
            }

            def create_srt(url, lang, path):
                Path(path).write_text("1\n00:00:00,000 --> 00:00:01,000\nTest\n")
                return path

            mock_downloader_instance.download_subtitle.side_effect = create_srt

            mock_writer_instance = MagicMock()
            mock_writer_instance.parse_srt.return_value = [
                {'start': 0.0, 'end': 1.0, 'text': 'Test'}
            ]
            mock_writer.return_value = mock_writer_instance

            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--subtitle', '1', '--output', tmpdir],
                input='n\n',
            )

            assert result.exit_code == 0, result.output
            # Should download English (index 1) without prompting
            mock_downloader_instance.download_subtitle.assert_called_once()
            call_args = mock_downloader_instance.download_subtitle.call_args
            assert call_args[0][1] == 'en'

    @patch('main.VideoDownloader')
    def test_subtitle_out_of_range_exits_with_error(self, mock_downloader):
        """--subtitle N where N exceeds available count should exit with code 1."""
        runner = CliRunner()

        mock_downloader_instance = MagicMock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.get_available_subtitles.return_value = {
            'en': {'name': 'English'},
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--subtitle', '5'],
        )

        assert result.exit_code == 1

    @patch('main.VideoDownloader')
    def test_subtitle_n_with_no_subtitles_exits_with_error(self, mock_downloader):
        """--subtitle N (N > 0) when no subtitles available should exit with code 1."""
        runner = CliRunner()

        mock_downloader_instance = MagicMock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.get_available_subtitles.return_value = {}

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--subtitle', '1'],
        )

        assert result.exit_code == 1

    @patch('main.VideoDownloader')
    @patch('main.SubtitleWriter')
    def test_subtitle_second_option_selects_correctly(self, mock_writer, mock_downloader):
        """--subtitle 2 should download the second subtitle (Spanish)."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_downloader_instance = MagicMock()
            mock_downloader.return_value = mock_downloader_instance
            mock_downloader_instance.get_available_subtitles.return_value = {
                'en': {'name': 'English'},
                'es': {'name': 'Spanish'},
            }
            mock_downloader_instance.get_video_info.return_value = {
                'video_id': 'abc123',
                'title': 'Test Video',
                'duration': 60.0,
                'platform': 'youtube',
                'upload_date': '20200101',
            }

            def create_srt(url, lang, path):
                Path(path).write_text("1\n00:00:00,000 --> 00:00:01,000\nTest\n")
                return path

            mock_downloader_instance.download_subtitle.side_effect = create_srt

            mock_writer_instance = MagicMock()
            mock_writer_instance.parse_srt.return_value = [
                {'start': 0.0, 'end': 1.0, 'text': 'Test'}
            ]
            mock_writer.return_value = mock_writer_instance

            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--subtitle', '2', '--output', tmpdir],
                input='n\n',
            )

            assert result.exit_code == 0, result.output
            call_args = mock_downloader_instance.download_subtitle.call_args
            assert call_args[0][1] == 'es'


class TestPreviewFlag:
    """Tests for --preview flag."""

    def _get_command_line(self, output: str) -> str:
        """Extract the command line from test output (line containing 'main.py')."""
        for line in output.strip().split('\n'):
            if 'main.py' in line:
                return line
        return ''

    @patch('main.VideoDownloader')
    def test_preview_outputs_command_to_stdout(self, mock_downloader):
        """--preview should output the real command with --subtitle N."""
        runner = CliRunner()

        mock_downloader_instance = MagicMock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.get_available_subtitles.return_value = {
            'en': {'name': 'English'},
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
            input='1\n',
        )

        assert result.exit_code == 0, result.output
        assert '--subtitle 1' in result.output

    @patch('main.VideoDownloader')
    def test_preview_command_includes_y_flag(self, mock_downloader):
        """--preview output command should always include -y flag."""
        runner = CliRunner()

        mock_downloader_instance = MagicMock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.get_available_subtitles.return_value = {
            'en': {'name': 'English'},
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
            input='1\n',
        )

        assert result.exit_code == 0
        cmd = self._get_command_line(result.output)
        assert '-y' in cmd

    @patch('main.VideoDownloader')
    def test_preview_preserves_model_flag(self, mock_downloader):
        """--preview output command should preserve --model flag when non-default."""
        runner = CliRunner()

        mock_downloader_instance = MagicMock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.get_available_subtitles.return_value = {
            'en': {'name': 'English'},
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview', '--model', 'large'],
            input='1\n',
        )

        assert result.exit_code == 0
        cmd = self._get_command_line(result.output)
        assert '--model large' in cmd

    @patch('main.VideoDownloader')
    def test_preview_does_not_include_preview_flag_in_output(self, mock_downloader):
        """--preview output command should NOT include --preview."""
        runner = CliRunner()

        mock_downloader_instance = MagicMock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.get_available_subtitles.return_value = {
            'en': {'name': 'English'},
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
            input='1\n',
        )

        assert result.exit_code == 0
        # The generated command line should not contain --preview
        cmd = self._get_command_line(result.output)
        assert cmd != '', f"No command line found in output:\n{result.output}"
        assert '--preview' not in cmd

    def test_preview_with_local_file_outputs_subtitle_0(self):
        """--preview with local file should output command with --subtitle 0."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / 'test_video.mp4'
            video_path.touch()

            result = runner.invoke(
                main.main,
                [str(video_path), '--preview'],
            )

            assert result.exit_code == 0, result.output
            assert '--subtitle 0' in result.output

    @patch('main.VideoDownloader')
    def test_preview_exits_without_processing(self, mock_downloader):
        """--preview should exit without downloading or transcribing the video."""
        runner = CliRunner()

        mock_downloader_instance = MagicMock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.get_available_subtitles.return_value = {
            'en': {'name': 'English'},
        }
        mock_downloader_instance.get_video_info.return_value = {
            'video_id': 'abc123',
            'upload_date': '20200101',
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
            input='0\n',  # user selects transcribe (0)
        )

        assert result.exit_code == 0
        # Should NOT download the video
        mock_downloader_instance.download.assert_not_called()
        mock_downloader_instance.download_subtitle.assert_not_called()

    @patch('main.VideoDownloader')
    def test_preview_with_no_subtitles_outputs_subtitle_0(self, mock_downloader):
        """--preview with no subtitles available should output command with --subtitle 0."""
        runner = CliRunner()

        mock_downloader_instance = MagicMock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.get_available_subtitles.return_value = {}
        mock_downloader_instance.get_video_info.return_value = {
            'video_id': 'abc123',
            'upload_date': '20200101',
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
        )

        assert result.exit_code == 0, result.output
        assert '--subtitle 0' in result.output

    @patch('main.VideoDownloader')
    def test_preview_includes_url_in_output_command(self, mock_downloader):
        """--preview output command should include the original URL."""
        runner = CliRunner()

        mock_downloader_instance = MagicMock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.get_available_subtitles.return_value = {
            'en': {'name': 'English'},
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
            input='1\n',
        )

        assert result.exit_code == 0
        cmd = self._get_command_line(result.output)
        assert 'abc123' in cmd

    @patch('main.VideoDownloader')
    def test_preview_preserves_output_flag(self, mock_downloader):
        """--preview output command should preserve --output flag if specified."""
        runner = CliRunner()

        mock_downloader_instance = MagicMock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.get_available_subtitles.return_value = {
            'en': {'name': 'English'},
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview', '--output', '/tmp/subs'],
            input='1\n',
        )

        assert result.exit_code == 0
        cmd = self._get_command_line(result.output)
        assert '--output' in cmd

    @patch('main.VideoDownloader')
    def test_preview_omits_default_model(self, mock_downloader):
        """--preview output command should NOT include --model when it's the default."""
        runner = CliRunner()

        mock_downloader_instance = MagicMock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.get_available_subtitles.return_value = {
            'en': {'name': 'English'},
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
            input='1\n',
        )

        assert result.exit_code == 0
        cmd = self._get_command_line(result.output)
        assert '--model' not in cmd

    @patch('main.VideoDownloader')
    def test_preview_user_selects_transcribe(self, mock_downloader):
        """--preview with user selecting 0 should output command with --subtitle 0."""
        runner = CliRunner()

        mock_downloader_instance = MagicMock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.get_available_subtitles.return_value = {
            'en': {'name': 'English'},
            'fr': {'name': 'French'},
        }
        mock_downloader_instance.get_video_info.return_value = {
            'video_id': 'abc123',
            'upload_date': '20200101',
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
            input='0\n',
        )

        assert result.exit_code == 0
        assert '--subtitle 0' in result.output


class TestSubtitleDownloadTranslationSourceLanguage:
    """Test that downloaded subtitle's language is used for translation, not --language."""

    @patch('main.translate_subtitles')
    @patch('main.VideoDownloader')
    @patch('main.SubtitleWriter')
    def test_subtitle_download_uses_subtitle_lang_not_flag_lang(
        self, mock_writer, mock_downloader, mock_translate
    ):
        """--subtitle 1 --language zh --yes should use English (not Chinese) as translation source."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_downloader_instance = MagicMock()
            mock_downloader.return_value = mock_downloader_instance
            mock_downloader_instance.get_available_subtitles.return_value = {
                'en': {'name': 'English'},
            }
            mock_downloader_instance.get_video_info.return_value = {
                'video_id': 'abc123',
                'title': 'Test Video',
                'duration': 60.0,
                'platform': 'youtube',
                'upload_date': '20200101',
            }

            def create_srt(url, lang, path):
                Path(path).write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n")
                return path

            mock_downloader_instance.download_subtitle.side_effect = create_srt

            mock_writer_instance = MagicMock()
            mock_writer_instance.parse_srt.return_value = [
                {'start': 0.0, 'end': 1.0, 'text': 'Hello'}
            ]
            mock_writer.return_value = mock_writer_instance

            mock_translate.return_value = None

            result = runner.invoke(
                main.main,
                [
                    'https://youtube.com/watch?v=abc123',
                    '--subtitle', '1',
                    '--language', 'zh',
                    '--yes',
                    '--output', tmpdir,
                ],
            )

            assert result.exit_code == 0, result.output
            # translate_subtitles should be called with language_name='English', not 'Chinese'
            mock_translate.assert_called_once()
            call_kwargs = mock_translate.call_args[1]
            assert call_kwargs.get('language_name') == 'English', (
                f"Expected language_name='English' but got {call_kwargs.get('language_name')!r}"
            )


class TestPreviewTwoPhaseWorkflow:
    """Tests for two-phase preview: separate transcribe + translate commands (VRAM constraint).

    All tests in this class run with auto_unload=True to verify two-phase behavior.
    """

    @pytest.fixture(autouse=True)
    def enable_auto_unload(self):
        """Patch load_config with auto_unload=True for all tests in this class."""
        config = {
            'ollama': {
                'model': 'translategemma:4b',
                'base_url': 'http://localhost:11434',
                'batch_size': 50,
                'keep_alive': '10m',
                'auto_unload': True,
            },
            'output': {'directory': None},
        }
        with patch('main.load_config', return_value=config):
            yield

    def _get_cmd_lines(self, output: str) -> list[str]:
        """Extract all command lines from test output."""
        return [l for l in output.strip().split('\n') if 'main.py' in l]

    @patch('main.VideoDownloader')
    def test_url_no_subtitles_outputs_two_commands(self, mock_downloader):
        """URL with no subtitles should output exactly 2 command lines."""
        runner = CliRunner()

        mock_instance = MagicMock()
        mock_downloader.return_value = mock_instance
        mock_instance.get_available_subtitles.return_value = {}
        mock_instance.get_video_info.return_value = {
            'video_id': 'abc123',
            'upload_date': '20200101',
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
        )

        assert result.exit_code == 0, result.output
        assert len(self._get_cmd_lines(result.output)) == 2

    @patch('main.VideoDownloader')
    def test_url_no_subtitles_first_cmd_has_subtitle_0_no_y(self, mock_downloader):
        """First command should have --subtitle 0 and no -y."""
        runner = CliRunner()

        mock_instance = MagicMock()
        mock_downloader.return_value = mock_instance
        mock_instance.get_available_subtitles.return_value = {}
        mock_instance.get_video_info.return_value = {
            'video_id': 'abc123',
            'upload_date': '20200101',
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
        )

        assert result.exit_code == 0
        first_cmd = self._get_cmd_lines(result.output)[0]
        assert '--subtitle 0' in first_cmd
        assert '-y' not in first_cmd

    @patch('main.VideoDownloader')
    def test_url_no_subtitles_second_cmd_has_srt_and_y(self, mock_downloader):
        """Second command should have .srt path and -y, no --subtitle."""
        runner = CliRunner()

        mock_instance = MagicMock()
        mock_downloader.return_value = mock_instance
        mock_instance.get_available_subtitles.return_value = {}
        mock_instance.get_video_info.return_value = {
            'video_id': 'abc123',
            'upload_date': '20200101',
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
        )

        assert result.exit_code == 0
        second_cmd = self._get_cmd_lines(result.output)[1]
        assert '.srt' in second_cmd
        assert '-y' in second_cmd
        assert '--subtitle' not in second_cmd

    @patch('main.VideoDownloader')
    def test_url_no_subtitles_srt_path_contains_video_id_and_date(self, mock_downloader):
        """SRT path in second command should include video_id and date prefix."""
        runner = CliRunner()

        mock_instance = MagicMock()
        mock_downloader.return_value = mock_instance
        mock_instance.get_available_subtitles.return_value = {}
        mock_instance.get_video_info.return_value = {
            'video_id': 'abc123',
            'upload_date': '20200101',
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
        )

        assert result.exit_code == 0
        second_cmd = self._get_cmd_lines(result.output)[1]
        assert 'abc123' in second_cmd
        assert '20200101' in second_cmd

    @patch('main.VideoDownloader')
    def test_url_subtitles_user_picks_0_outputs_two_commands(self, mock_downloader):
        """URL with subtitles, user picks 0 (transcribe) → 2 commands."""
        runner = CliRunner()

        mock_instance = MagicMock()
        mock_downloader.return_value = mock_instance
        mock_instance.get_available_subtitles.return_value = {
            'en': {'name': 'English'},
        }
        mock_instance.get_video_info.return_value = {
            'video_id': 'abc123',
            'upload_date': '20200101',
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
            input='0\n',
        )

        assert result.exit_code == 0, result.output
        assert len(self._get_cmd_lines(result.output)) == 2

    @patch('main.VideoDownloader')
    def test_url_subtitle_download_keeps_single_command(self, mock_downloader):
        """When user picks a subtitle to download (>0), single-command behavior is preserved."""
        runner = CliRunner()

        mock_instance = MagicMock()
        mock_downloader.return_value = mock_instance
        mock_instance.get_available_subtitles.return_value = {
            'en': {'name': 'English'},
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
            input='1\n',
        )

        assert result.exit_code == 0
        cmds = self._get_cmd_lines(result.output)
        assert len(cmds) == 1
        assert '-y' in cmds[0]

    def test_local_file_outputs_two_commands(self):
        """Local file --preview should output 2 commands."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / 'test_video.mp4'
            video_path.touch()

            result = runner.invoke(
                main.main,
                [str(video_path), '--preview'],
            )

            assert result.exit_code == 0, result.output
            assert len(self._get_cmd_lines(result.output)) == 2

    def test_local_file_second_cmd_has_srt_path_and_y(self):
        """Local file --preview second command should have .srt path and -y."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / 'test_video.mp4'
            video_path.touch()

            result = runner.invoke(
                main.main,
                [str(video_path), '--preview'],
            )

            assert result.exit_code == 0, result.output
            second_cmd = self._get_cmd_lines(result.output)[1]
            assert '.srt' in second_cmd
            assert '-y' in second_cmd

    @patch('main.VideoDownloader')
    def test_output_flag_preserved_in_both_commands(self, mock_downloader):
        """--output flag should appear in both commands."""
        runner = CliRunner()

        mock_instance = MagicMock()
        mock_downloader.return_value = mock_instance
        mock_instance.get_available_subtitles.return_value = {}
        mock_instance.get_video_info.return_value = {
            'video_id': 'abc123',
            'upload_date': '20200101',
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview', '--output', '/tmp/subs'],
        )

        assert result.exit_code == 0
        cmds = self._get_cmd_lines(result.output)
        assert '--output' in cmds[0]
        assert '--output' in cmds[1]

    @patch('main.VideoDownloader')
    def test_language_flag_in_both_commands(self, mock_downloader):
        """--language should appear in both commands (transcribe source + translate source)."""
        runner = CliRunner()

        mock_instance = MagicMock()
        mock_downloader.return_value = mock_instance
        mock_instance.get_available_subtitles.return_value = {}
        mock_instance.get_video_info.return_value = {
            'video_id': 'abc123',
            'upload_date': '20200101',
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview', '--language', 'en'],
        )

        assert result.exit_code == 0
        cmds = self._get_cmd_lines(result.output)
        assert '--language en' in cmds[0]
        assert '--language en' in cmds[1]


class TestPreviewSingleCommandWhenAutoUnloadDisabled:
    """When auto_unload=False (default), --preview outputs a single command for transcription paths."""

    def _get_cmd_lines(self, output: str) -> list[str]:
        return [l for l in output.strip().split('\n') if 'main.py' in l]

    def _make_config(self, auto_unload: bool = False) -> dict:
        return {
            'ollama': {
                'model': 'translategemma:4b',
                'base_url': 'http://localhost:11434',
                'batch_size': 50,
                'keep_alive': '10m',
                'auto_unload': auto_unload,
            },
            'output': {'directory': None},
        }

    @patch('main.load_config')
    @patch('main.VideoDownloader')
    def test_url_no_subtitles_single_command(self, mock_downloader, mock_config):
        """URL with no subtitles, auto_unload=False → single command with --subtitle 0 -y."""
        mock_config.return_value = self._make_config(auto_unload=False)
        runner = CliRunner()

        mock_instance = MagicMock()
        mock_downloader.return_value = mock_instance
        mock_instance.get_available_subtitles.return_value = {}

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
        )

        assert result.exit_code == 0, result.output
        cmds = self._get_cmd_lines(result.output)
        assert len(cmds) == 1
        assert '--subtitle 0' in cmds[0]
        assert '-y' in cmds[0]

    @patch('main.load_config')
    @patch('main.VideoDownloader')
    def test_url_subtitles_user_picks_0_single_command(self, mock_downloader, mock_config):
        """URL with subtitles, user picks 0, auto_unload=False → single command."""
        mock_config.return_value = self._make_config(auto_unload=False)
        runner = CliRunner()

        mock_instance = MagicMock()
        mock_downloader.return_value = mock_instance
        mock_instance.get_available_subtitles.return_value = {'en': {'name': 'English'}}

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
            input='0\n',
        )

        assert result.exit_code == 0, result.output
        cmds = self._get_cmd_lines(result.output)
        assert len(cmds) == 1
        assert '--subtitle 0' in cmds[0]
        assert '-y' in cmds[0]

    @patch('main.load_config')
    def test_local_file_single_command(self, mock_config):
        """Local file, auto_unload=False → single command with --subtitle 0 -y."""
        mock_config.return_value = self._make_config(auto_unload=False)
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / 'test_video.mp4'
            video_path.touch()

            result = runner.invoke(
                main.main,
                [str(video_path), '--preview'],
            )

            assert result.exit_code == 0, result.output
            cmds = self._get_cmd_lines(result.output)
            assert len(cmds) == 1
            assert '--subtitle 0' in cmds[0]
            assert '-y' in cmds[0]


class TestActionFlagInPreviewCommands:
    """Tests that --preview two-phase commands include --action flag."""

    @pytest.fixture(autouse=True)
    def enable_auto_unload(self):
        """Patch load_config with auto_unload=True for all tests in this class."""
        config = {
            'ollama': {
                'model': 'translategemma:4b',
                'base_url': 'http://localhost:11434',
                'batch_size': 50,
                'keep_alive': '10m',
                'auto_unload': True,
            },
            'output': {'directory': None},
        }
        with patch('main.load_config', return_value=config):
            yield

    def _get_cmd_lines(self, output: str) -> list[str]:
        return [l for l in output.strip().split('\n') if 'main.py' in l]

    @patch('main.VideoDownloader')
    def test_phase1_includes_action_transcribe(self, mock_downloader):
        """Phase 1 command (transcription) should include --action transcribe."""
        runner = CliRunner()

        mock_instance = MagicMock()
        mock_downloader.return_value = mock_instance
        mock_instance.get_available_subtitles.return_value = {}
        mock_instance.get_video_info.return_value = {
            'video_id': 'abc123',
            'upload_date': '20200101',
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
        )

        assert result.exit_code == 0, result.output
        first_cmd = self._get_cmd_lines(result.output)[0]
        assert '--action transcribe' in first_cmd

    @patch('main.VideoDownloader')
    def test_phase2_includes_action_translate(self, mock_downloader):
        """Phase 2 command (translation) should include --action translate."""
        runner = CliRunner()

        mock_instance = MagicMock()
        mock_downloader.return_value = mock_instance
        mock_instance.get_available_subtitles.return_value = {}
        mock_instance.get_video_info.return_value = {
            'video_id': 'abc123',
            'upload_date': '20200101',
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
        )

        assert result.exit_code == 0, result.output
        second_cmd = self._get_cmd_lines(result.output)[1]
        assert '--action translate' in second_cmd

    def test_local_file_phase1_includes_action_transcribe(self):
        """Local file --preview Phase 1 command should include --action transcribe."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / 'test_video.mp4'
            video_path.touch()

            result = runner.invoke(
                main.main,
                [str(video_path), '--preview'],
            )

            assert result.exit_code == 0, result.output
            first_cmd = self._get_cmd_lines(result.output)[0]
            assert '--action transcribe' in first_cmd

    def test_local_file_phase2_includes_action_translate(self):
        """Local file --preview Phase 2 command should include --action translate."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / 'test_video.mp4'
            video_path.touch()

            result = runner.invoke(
                main.main,
                [str(video_path), '--preview'],
            )

            assert result.exit_code == 0, result.output
            second_cmd = self._get_cmd_lines(result.output)[1]
            assert '--action translate' in second_cmd

    @patch('main.VideoDownloader')
    def test_phase1_no_action_flag_in_subtitle_download_command(self, mock_downloader):
        """Subtitle download command (user picks >0) should NOT include --action."""
        runner = CliRunner()

        mock_instance = MagicMock()
        mock_downloader.return_value = mock_instance
        mock_instance.get_available_subtitles.return_value = {
            'en': {'name': 'English'},
        }

        result = runner.invoke(
            main.main,
            ['https://youtube.com/watch?v=abc123', '--preview'],
            input='1\n',
        )

        assert result.exit_code == 0
        cmds = self._get_cmd_lines(result.output)
        assert len(cmds) == 1
        assert '--action' not in cmds[0]
