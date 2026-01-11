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

            # Run the CLI with a YouTube URL
            result = runner.invoke(main.main, ['https://www.youtube.com/watch?v=abc123', '--output', tmpdir])

            # Verify the command succeeded
            assert result.exit_code == 0

            # Verify VideoDownloader was called
            mock_downloader_instance.download.assert_called_once()

            # Verify the rest of the pipeline was called
            mock_extractor_instance.extract_audio.assert_called_once()
            mock_transcriber_instance.transcribe.assert_called_once()
            mock_writer_instance.write_srt.assert_called_once()
            mock_writer_instance.write_txt.assert_called_once()

    @patch('main.SubtitleWriter')
    @patch('main.Transcriber')
    @patch('main.AudioExtractor')
    @patch('main.VideoDownloader')
    def test_main_with_url_uses_video_title_for_output(self, mock_downloader, mock_extractor, mock_transcriber, mock_writer):
        """Test that output files use sanitized video title when processing URLs."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock VideoDownloader instance with download method
            mock_downloader_instance = MagicMock()
            mock_downloader.return_value = mock_downloader_instance
            mock_downloader_instance.download.return_value = {
                'file_path': f'{tmpdir}/xyz789.mp4',
                'title': 'My Test Video: Part 1',  # Contains characters that need sanitizing
                'video_id': 'xyz789',
                'duration': 60.0,
                'platform': 'youtube'
            }

            # Mock the static sanitize_filename method
            mock_downloader.sanitize_filename = MagicMock(return_value='My_Test_Video__Part_1')

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

            # Run the CLI
            result = runner.invoke(main.main, ['https://youtube.com/watch?v=xyz789', '--output', tmpdir])

            assert result.exit_code == 0

            # Verify write_srt was called with sanitized filename
            srt_call_args = mock_writer_instance.write_srt.call_args[0]
            srt_path = str(srt_call_args[1])
            # Should use sanitized title (colon replaced)
            assert 'My_Test_Video__Part_1' in srt_path


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

            # Run with file path
            result = runner.invoke(main.main, [str(video_path)])

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

            # Run with file path
            result = runner.invoke(main.main, [str(video_path)])

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
