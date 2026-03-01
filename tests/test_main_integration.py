import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from click.testing import CliRunner
import tempfile
import sys
import os

# Add parent directory to path to import main
sys.path.insert(0, str(Path(__file__).parent.parent))
import main


class TestMainWithURLInput:
    """Integration tests for main.py with URL inputs."""

    @patch('main.VideoDownloader')
    @patch('main.AudioExtractor')
    @patch('main.Transcriber')
    @patch('main.SubtitleWriter')
    def test_main_with_youtube_url(self, mock_writer, mock_transcriber, mock_extractor, mock_downloader):
        """Test end-to-end processing with YouTube URL."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock VideoDownloader
            mock_downloader_instance = MagicMock()
            mock_downloader.return_value = mock_downloader_instance

            # Mock subtitle check - return empty (no subtitles)
            mock_downloader_instance.get_available_subtitles.return_value = ({}, {'title': 'Unknown', 'channel': None})

            mock_downloader_instance.download.return_value = {
                'file_path': f'{tmpdir}/abc123.mp4',
                'title': 'Test Video Title',
                'video_id': 'abc123',
                'duration': 120.5,
                'platform': 'youtube'
            }

            # Create the mock video file
            Path(f'{tmpdir}/abc123.mp4').touch()

            # Mock AudioExtractor
            mock_extractor_instance = MagicMock()
            mock_extractor.return_value = mock_extractor_instance
            mock_extractor_instance.extract_audio.return_value = f'{tmpdir}/test.wav'

            # Mock Transcriber
            mock_transcriber_instance = MagicMock()
            mock_transcriber.return_value = mock_transcriber_instance
            mock_transcriber_instance.transcribe.return_value = [
                {'start': 0.0, 'end': 2.0, 'text': 'Hello'},
                {'start': 2.0, 'end': 4.0, 'text': 'World'}
            ]

            # Mock SubtitleWriter
            mock_writer_instance = MagicMock()
            mock_writer.return_value = mock_writer_instance

            # Run the CLI with a YouTube URL (provide 'n' to skip translation)
            result = runner.invoke(main.main, ['https://www.youtube.com/watch?v=abc123', '--output', tmpdir], input='n\n')

            # Verify the command succeeded
            assert result.exit_code == 0

            # Verify VideoDownloader was called
            mock_downloader_instance.download.assert_called_once()

            # Verify the rest of the pipeline was called
            mock_extractor_instance.extract_audio.assert_called_once()
            mock_transcriber_instance.transcribe.assert_called_once()
            mock_writer_instance.write_srt.assert_called_once()

    @patch('main.SubtitleWriter')
    @patch('main.Transcriber')
    @patch('main.AudioExtractor')
    @patch('main.VideoDownloader')
    def test_main_with_url_uses_video_id_for_output(self, mock_downloader, mock_extractor, mock_transcriber, mock_writer):
        """Test that output files use video ID when processing URLs."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock VideoDownloader instance with download method
            mock_downloader_instance = MagicMock()
            mock_downloader.return_value = mock_downloader_instance

            # Mock subtitle check - return empty (no subtitles)
            mock_downloader_instance.get_available_subtitles.return_value = ({}, {'title': 'Unknown', 'channel': None})

            mock_downloader_instance.download.return_value = {
                'file_path': f'{tmpdir}/xyz789.mp4',
                'title': 'My Test Video: Part 1',
                'video_id': 'xyz789',
                'duration': 60.0,
                'platform': 'youtube'
            }

            # Create the mock video file
            Path(f'{tmpdir}/xyz789.mp4').touch()

            # Mock other components
            mock_extractor_instance = MagicMock()
            mock_extractor.return_value = mock_extractor_instance
            mock_transcriber_instance = MagicMock()
            mock_transcriber.return_value = mock_transcriber_instance
            mock_transcriber_instance.transcribe.return_value = [
                {'start': 0.0, 'end': 1.0, 'text': 'Test'}
            ]
            mock_writer_instance = MagicMock()
            mock_writer.return_value = mock_writer_instance

            # Run the CLI (provide 'n' to skip translation)
            result = runner.invoke(main.main, ['https://youtube.com/watch?v=xyz789', '--output', tmpdir], input='n\n')

            assert result.exit_code == 0

            # Verify write_srt was called with video ID in filename
            srt_call_args = mock_writer_instance.write_srt.call_args[0]
            srt_path = str(srt_call_args[1])
            # Should use video ID instead of title
            assert 'xyz789' in srt_path


class TestMainWithFilePathInput:
    """Integration tests to verify file path input still works (regression tests)."""

    @patch('main.AudioExtractor')
    @patch('main.Transcriber')
    @patch('main.SubtitleWriter')
    def test_main_with_file_path_still_works(self, mock_writer, mock_transcriber, mock_extractor):
        """Test that local file paths still work after URL feature is added."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real video file
            video_path = Path(tmpdir) / 'test_video.mp4'
            video_path.touch()

            # Mock components
            mock_extractor_instance = MagicMock()
            mock_extractor.return_value = mock_extractor_instance
            mock_transcriber_instance = MagicMock()
            mock_transcriber.return_value = mock_transcriber_instance
            mock_transcriber_instance.transcribe.return_value = [
                {'start': 0.0, 'end': 2.0, 'text': 'Test'}
            ]
            mock_writer_instance = MagicMock()
            mock_writer.return_value = mock_writer_instance

            # Run with file path (provide 'n' to skip translation)
            result = runner.invoke(main.main, [str(video_path)], input='n\n')

            assert result.exit_code == 0

            # Verify pipeline was called
            mock_extractor_instance.extract_audio.assert_called_once()
            mock_transcriber_instance.transcribe.assert_called_once()
            mock_writer_instance.write_srt.assert_called_once()

    @patch('main.AudioExtractor')
    @patch('main.Transcriber')
    @patch('main.SubtitleWriter')
    def test_main_with_file_path_uses_stem_for_output(self, mock_writer, mock_transcriber, mock_extractor):
        """Test that file path input uses filename stem for output naming."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a video file with a specific name
            video_path = Path(tmpdir) / 'my_video_file.mp4'
            video_path.touch()

            # Mock components
            mock_extractor_instance = MagicMock()
            mock_extractor.return_value = mock_extractor_instance
            mock_transcriber_instance = MagicMock()
            mock_transcriber.return_value = mock_transcriber_instance
            mock_transcriber_instance.transcribe.return_value = [
                {'start': 0.0, 'end': 1.0, 'text': 'Test'}
            ]
            mock_writer_instance = MagicMock()
            mock_writer.return_value = mock_writer_instance

            # Run with file path (provide 'n' to skip translation)
            result = runner.invoke(main.main, [str(video_path)], input='n\n')

            assert result.exit_code == 0

            # Verify SRT output uses the file stem
            srt_call_args = mock_writer_instance.write_srt.call_args[0]
            srt_path = str(srt_call_args[1])
            assert 'my_video_file' in srt_path


class TestMainErrorScenarios:
    """Integration tests for error handling."""

    @patch('main.VideoDownloader')
    def test_main_with_invalid_url_shows_error(self, mock_downloader):
        """Test that invalid URLs show clear error messages."""
        runner = CliRunner()

        # Mock downloader to raise an exception
        mock_downloader_instance = MagicMock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.download.side_effect = Exception("Unsupported URL")

        result = runner.invoke(main.main, ['https://invalid-url.com/video'])

        # Should exit with error
        assert result.exit_code == 1
        # Should show error message
        assert 'Error' in result.output or 'error' in result.output.lower()

    @patch('main.VideoDownloader')
    def test_main_with_download_failure_exits(self, mock_downloader):
        """Test that download failures cause graceful exit."""
        runner = CliRunner()

        # Mock downloader to simulate network error
        mock_downloader_instance = MagicMock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.download.side_effect = Exception("Network error")

        result = runner.invoke(main.main, ['https://youtube.com/watch?v=test'])

        # Should exit with error code
        assert result.exit_code != 0


class TestTranslationPrompt:
    """Integration tests for translation prompt."""

    @patch('main.OllamaTranslator')
    @patch('main.AudioExtractor')
    @patch('main.Transcriber')
    @patch('main.SubtitleWriter')
    def test_translation_prompt_with_no_skips_translation(self, mock_writer, mock_transcriber, mock_extractor, mock_translator):
        """Test that answering no to translation prompt skips translation."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real video file
            video_path = Path(tmpdir) / 'test_video.mp4'
            video_path.touch()

            # Mock components
            mock_extractor_instance = MagicMock()
            mock_extractor.return_value = mock_extractor_instance

            mock_transcriber_instance = MagicMock()
            mock_transcriber.return_value = mock_transcriber_instance
            mock_transcriber_instance.transcribe.return_value = [
                {'start': 0.0, 'end': 2.0, 'text': 'Test'}
            ]

            # Mock SubtitleWriter
            from src.subtitle_writer import SubtitleWriter
            real_writer = SubtitleWriter()
            mock_writer.return_value = real_writer

            # Run with file path and simulate user input: no to translation
            result = runner.invoke(
                main.main,
                [str(video_path), '--output', tmpdir],
                input='n\n'  # no to translation
            )

            # Verify the command succeeded
            assert result.exit_code == 0

            # Verify translator was not called
            mock_translator.return_value.translate_segments.assert_not_called()

            # Verify SRT file exists
            srt_files = list(Path(tmpdir).glob('*.srt'))
            assert len(srt_files) == 1


class TestOutputDirectoryPriority:
    """Tests for output directory configuration priority: CLI > config > default."""

    def test_cli_argument_has_highest_priority(self):
        """CLI --output flag should override both config and default."""
        with tempfile.TemporaryDirectory() as cli_dir:
            with tempfile.TemporaryDirectory() as config_dir:
                with tempfile.TemporaryDirectory() as default_dir:
                    config = {'output': {'directory': config_dir}}

                    result = main.get_output_directory(cli_dir, config, Path(default_dir))

                    assert result == Path(cli_dir)

    def test_config_has_second_priority(self):
        """Config output.directory should be used when CLI is not provided."""
        with tempfile.TemporaryDirectory() as config_dir:
            with tempfile.TemporaryDirectory() as default_dir:
                config = {'output': {'directory': config_dir}}

                result = main.get_output_directory(None, config, Path(default_dir))

                assert result == Path(config_dir)

    def test_default_used_when_no_cli_or_config(self):
        """Default path should be used when neither CLI nor config is set."""
        with tempfile.TemporaryDirectory() as default_dir:
            config = {'output': {'directory': None}}

            result = main.get_output_directory(None, config, Path(default_dir))

            assert result == Path(default_dir)

    def test_default_used_when_config_output_missing(self):
        """Default path should be used when config has no output section."""
        with tempfile.TemporaryDirectory() as default_dir:
            config = {'ollama': {'model': 'test'}}  # No output section

            result = main.get_output_directory(None, config, Path(default_dir))

            assert result == Path(default_dir)

    def test_default_used_when_config_empty(self):
        """Default path should be used when config is empty."""
        with tempfile.TemporaryDirectory() as default_dir:
            config = {}

            result = main.get_output_directory(None, config, Path(default_dir))

            assert result == Path(default_dir)

    def test_cli_creates_directory_if_not_exists(self):
        """CLI output directory should be created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as base_dir:
            new_dir = Path(base_dir) / 'new_output_dir'
            config = {'output': {'directory': None}}

            result = main.get_output_directory(str(new_dir), config, Path(base_dir))

            assert result == new_dir
            assert new_dir.exists()

    def test_config_creates_directory_if_not_exists(self):
        """Config output directory should be created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as base_dir:
            new_dir = Path(base_dir) / 'config_output_dir'
            config = {'output': {'directory': str(new_dir)}}

            result = main.get_output_directory(None, config, Path(base_dir))

            assert result == new_dir
            assert new_dir.exists()


class TestDefaultOutputDirectory:
    """Tests for default output directory: cwd for URLs, video dir for local files."""

    @patch('main.load_config', return_value={'ollama': {'model': 'test', 'base_url': 'http://localhost:11434', 'batch_size': 50, 'keep_alive': '10m'}})
    @patch('main.VideoDownloader')
    @patch('main.AudioExtractor')
    @patch('main.Transcriber')
    @patch('main.SubtitleWriter')
    def test_url_input_saves_to_cwd_by_default(self, mock_writer, mock_transcriber, mock_extractor, mock_downloader, mock_config):
        """URL input without --output should save subtitles to cwd, not temp dir."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock VideoDownloader
            mock_downloader_instance = MagicMock()
            mock_downloader.return_value = mock_downloader_instance
            mock_downloader_instance.get_available_subtitles.return_value = ({}, {'title': 'Unknown', 'channel': None})
            mock_downloader_instance.download.return_value = {
                'file_path': f'{tmpdir}/abc123.mp4',
                'title': 'Test Video',
                'video_id': 'abc123',
                'duration': 60.0,
                'platform': 'youtube',
                'upload_date': '20200101'
            }

            # Create mock video file in temp dir
            Path(f'{tmpdir}/abc123.mp4').touch()

            # Mock other components
            mock_extractor.return_value = MagicMock()
            mock_transcriber_instance = MagicMock()
            mock_transcriber.return_value = mock_transcriber_instance
            mock_transcriber_instance.transcribe.return_value = [
                {'start': 0.0, 'end': 1.0, 'text': 'Test'}
            ]
            mock_writer_instance = MagicMock()
            mock_writer.return_value = mock_writer_instance

            # Run without --output (provide 'n' to skip translation)
            result = runner.invoke(main.main, ['https://youtube.com/watch?v=abc123'], input='n\n')

            assert result.exit_code == 0

            # Verify write_srt path is in cwd, not in temp dir
            srt_call_args = mock_writer_instance.write_srt.call_args[0]
            srt_path = str(srt_call_args[1])
            assert tmpdir not in srt_path  # Should NOT be in temp dir
            assert str(Path.cwd()) in srt_path  # Should be in cwd

    @patch('main.load_config', return_value={'ollama': {'model': 'test', 'base_url': 'http://localhost:11434', 'batch_size': 50, 'keep_alive': '10m'}})
    @patch('main.AudioExtractor')
    @patch('main.Transcriber')
    @patch('main.SubtitleWriter')
    def test_local_file_saves_to_video_dir_by_default(self, mock_writer, mock_transcriber, mock_extractor, mock_config):
        """Local file without --output should save subtitles to video's directory."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create video file in a specific directory
            video_path = Path(tmpdir) / 'my_video.mp4'
            video_path.touch()

            # Mock components
            mock_extractor.return_value = MagicMock()
            mock_transcriber_instance = MagicMock()
            mock_transcriber.return_value = mock_transcriber_instance
            mock_transcriber_instance.transcribe.return_value = [
                {'start': 0.0, 'end': 1.0, 'text': 'Test'}
            ]
            mock_writer_instance = MagicMock()
            mock_writer.return_value = mock_writer_instance

            # Run without --output (provide 'n' to skip translation)
            result = runner.invoke(main.main, [str(video_path)], input='n\n')

            assert result.exit_code == 0

            # Verify write_srt path is in the video's directory
            srt_call_args = mock_writer_instance.write_srt.call_args[0]
            srt_path = str(srt_call_args[1])
            assert tmpdir in srt_path  # Should be in video's directory


class TestActionFlag:
    """Integration tests for --action flag."""

    @patch('main.translate_subtitles')
    @patch('main.AudioExtractor')
    @patch('main.Transcriber')
    @patch('main.SubtitleWriter')
    def test_action_transcribe_skips_translation(self, mock_writer, mock_transcriber, mock_extractor, mock_translate):
        """--action transcribe should skip translation entirely (no prompt, no call)."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / 'test_video.mp4'
            video_path.touch()

            mock_extractor.return_value = MagicMock()
            mock_transcriber_instance = MagicMock()
            mock_transcriber.return_value = mock_transcriber_instance
            mock_transcriber_instance.transcribe.return_value = [
                {'start': 0.0, 'end': 1.0, 'text': 'Test'}
            ]
            mock_writer.return_value = MagicMock()

            result = runner.invoke(
                main.main,
                [str(video_path), '--action', 'transcribe', '--output', tmpdir],
            )

            assert result.exit_code == 0, result.output
            # translate_subtitles must NOT be called at all
            mock_translate.assert_not_called()

    @patch('main.handle_srt_translation')
    def test_action_translate_with_srt_file_calls_translation(self, mock_handle):
        """--action translate with SRT file should call handle_srt_translation."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / 'test.srt'
            srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n")

            mock_handle.return_value = None

            result = runner.invoke(
                main.main,
                [str(srt_path), '--action', 'translate', '-y'],
            )

            assert result.exit_code == 0, result.output
            mock_handle.assert_called_once()

    def test_action_translate_with_video_file_exits_with_error(self):
        """--action translate with a video file (non-SRT) should exit with error."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / 'test_video.mp4'
            video_path.touch()

            result = runner.invoke(
                main.main,
                [str(video_path), '--action', 'translate'],
            )

            assert result.exit_code == 1
            assert 'SRT' in result.output or 'srt' in result.output.lower()

    @patch('main.translate_subtitles')
    @patch('main.AudioExtractor')
    @patch('main.Transcriber')
    @patch('main.SubtitleWriter')
    def test_no_action_flag_preserves_current_behavior(self, mock_writer, mock_transcriber, mock_extractor, mock_translate):
        """No --action flag should prompt for translation (backward compat)."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / 'test_video.mp4'
            video_path.touch()

            mock_extractor.return_value = MagicMock()
            mock_transcriber_instance = MagicMock()
            mock_transcriber.return_value = mock_transcriber_instance
            mock_transcriber_instance.transcribe.return_value = [
                {'start': 0.0, 'end': 1.0, 'text': 'Test'}
            ]
            mock_writer.return_value = MagicMock()
            mock_translate.return_value = None

            result = runner.invoke(
                main.main,
                [str(video_path), '--output', tmpdir],
                input='n\n',  # decline translation prompt
            )

            assert result.exit_code == 0, result.output
            # translate_subtitles IS called (it just returns None when user says no)
            mock_translate.assert_called_once()


class TestSkipOptionInSubtitleMenu:
    """Tests for 'S. Skip this video' option in the subtitle selection menu."""

    def _make_downloader_with_subtitles(self, mock_downloader, tmpdir):
        """Set up a mock downloader that returns two available subtitles."""
        mock_dl = MagicMock()
        mock_downloader.return_value = mock_dl
        mock_dl.get_available_subtitles.return_value = (
            {'zh': {'name': 'Chinese'}, 'en': {'name': 'English'}},
            {'title': 'Test Video', 'channel': 'Test Channel'},
        )
        mock_dl.get_video_info.return_value = {
            'video_id': 'abc123',
            'upload_date': '20240101',
        }
        return mock_dl

    @patch('main.load_config')
    @patch('main.VideoDownloader')
    def test_skip_option_shown_in_interactive_menu(self, mock_downloader, mock_config):
        """'S. Skip this video' appears in the subtitle selection menu."""
        runner = CliRunner()
        mock_config.return_value = {
            'ollama': {'model': 'test', 'base_url': 'http://localhost:11434',
                       'batch_size': 50, 'keep_alive': '10m', 'auto_unload': False},
            'output': {'directory': None},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_downloader_with_subtitles(mock_downloader, tmpdir)

            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123'],
                input='s\n',  # choose skip
            )

            assert result.exit_code == 0, result.output
            assert 'Skip' in result.output or 'skip' in result.output.lower()

    @patch('main.load_config')
    @patch('main.VideoDownloader')
    def test_skip_in_interactive_exits_without_processing(self, mock_downloader, mock_config):
        """Choosing 's' in interactive mode exits cleanly without downloading or transcribing."""
        runner = CliRunner()
        mock_config.return_value = {
            'ollama': {'model': 'test', 'base_url': 'http://localhost:11434',
                       'batch_size': 50, 'keep_alive': '10m', 'auto_unload': False},
            'output': {'directory': None},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_dl = self._make_downloader_with_subtitles(mock_downloader, tmpdir)

            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123'],
                input='s\n',
            )

            assert result.exit_code == 0, result.output
            mock_dl.download.assert_not_called()
            mock_dl.download_subtitle.assert_not_called()

    @patch('main.load_config')
    @patch('main.VideoDownloader')
    def test_skip_in_preview_emits_no_command(self, mock_downloader, mock_config):
        """Choosing 's' in --preview mode emits no runnable command to output."""
        runner = CliRunner()
        mock_config.return_value = {
            'ollama': {'model': 'test', 'base_url': 'http://localhost:11434',
                       'batch_size': 50, 'keep_alive': '10m', 'auto_unload': False},
            'output': {'directory': None},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_downloader_with_subtitles(mock_downloader, tmpdir)

            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview'],
                input='s\n',
            )

            assert result.exit_code == 0, result.output
            # No runnable command should appear — the line "uv run python main.py ..."
            # is only emitted when a subtitle/transcription choice is made
            assert 'uv run python main.py' not in result.output

    @patch('main.load_config')
    @patch('main.VideoDownloader')
    def test_skip_option_shown_in_preview_menu(self, mock_downloader, mock_config):
        """'S. Skip' appears in the subtitle selection menu during --preview."""
        runner = CliRunner()
        mock_config.return_value = {
            'ollama': {'model': 'test', 'base_url': 'http://localhost:11434',
                       'batch_size': 50, 'keep_alive': '10m', 'auto_unload': False},
            'output': {'directory': None},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_downloader_with_subtitles(mock_downloader, tmpdir)

            result = runner.invoke(
                main.main,
                ['https://youtube.com/watch?v=abc123', '--preview'],
                input='s\n',
            )

            assert result.exit_code == 0
            assert 'Skip' in result.output or 'skip' in result.output.lower()
