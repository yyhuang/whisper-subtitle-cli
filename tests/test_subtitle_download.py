import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from src.video_downloader import VideoDownloader


class TestGetAvailableSubtitles:
    """Tests for subtitle listing functionality."""

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_get_available_subtitles_returns_dict(self, mock_youtube_dl):
        """Test that get_available_subtitles returns subtitles as first element of tuple."""
        # Mock yt-dlp to return subtitle info
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            'subtitles': {
                'en': [{'ext': 'srt'}],
                'es': [{'ext': 'srt'}],
            },
            'automatic_captions': {}
        }

        downloader = VideoDownloader()
        result, _ = downloader.get_available_subtitles("https://youtube.com/watch?v=test")

        assert isinstance(result, dict)
        assert 'en' in result
        assert 'es' in result

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_get_available_subtitles_filters_auto_generated(self, mock_youtube_dl):
        """Test that auto-generated subtitles are filtered out."""
        # Mock yt-dlp to return both manual and auto-generated subtitles
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            'subtitles': {
                'en': [{'ext': 'srt'}],
                'fr': [{'ext': 'srt'}],
            },
            'automatic_captions': {
                'de': [{'ext': 'srv3'}],
                'ja': [{'ext': 'srv3'}],
            }
        }

        downloader = VideoDownloader()
        result, _ = downloader.get_available_subtitles("https://youtube.com/watch?v=test")

        # Should only include manual subtitles
        assert 'en' in result
        assert 'fr' in result
        # Should NOT include auto-generated
        assert 'de' not in result
        assert 'ja' not in result

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_get_available_subtitles_empty_when_none(self, mock_youtube_dl):
        """Test that empty dict is returned when no subtitles exist."""
        # Mock yt-dlp to return no subtitles
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            'subtitles': {},
            'automatic_captions': {
                'en': [{'ext': 'srv3'}],  # Only auto-generated
            }
        }

        downloader = VideoDownloader()
        result, _ = downloader.get_available_subtitles("https://youtube.com/watch?v=test")

        assert result == {}


class TestGetAvailableSubtitlesReturnsMeta:
    """Tests that get_available_subtitles returns video metadata alongside subtitles."""

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_returns_tuple_of_subtitles_and_meta(self, mock_youtube_dl):
        """get_available_subtitles should return (subtitles_dict, video_meta_dict)."""
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            'subtitles': {'en': [{'ext': 'srt'}]},
            'automatic_captions': {},
            'title': 'My Video',
            'channel': 'My Channel',
        }

        downloader = VideoDownloader()
        result = downloader.get_available_subtitles("https://youtube.com/watch?v=test")

        assert isinstance(result, tuple)
        assert len(result) == 2
        subtitles, meta = result
        assert isinstance(subtitles, dict)
        assert isinstance(meta, dict)

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_meta_contains_title_and_channel(self, mock_youtube_dl):
        """Video meta should contain title and channel fields."""
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            'subtitles': {'en': [{'ext': 'srt'}]},
            'automatic_captions': {},
            'title': 'My Video Title',
            'channel': 'Test Channel',
        }

        downloader = VideoDownloader()
        _, meta = downloader.get_available_subtitles("https://youtube.com/watch?v=test")

        assert meta['title'] == 'My Video Title'
        assert meta['channel'] == 'Test Channel'

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_meta_handles_missing_channel(self, mock_youtube_dl):
        """Video meta should handle missing channel gracefully."""
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            'subtitles': {},
            'automatic_captions': {},
            'title': 'My Video Title',
            # No channel field
        }

        downloader = VideoDownloader()
        _, meta = downloader.get_available_subtitles("https://youtube.com/watch?v=test")

        assert meta['title'] == 'My Video Title'
        assert meta['channel'] is None

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_error_returns_empty_subtitles_and_empty_meta(self, mock_youtube_dl):
        """On error, should return empty dict and empty meta."""
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.side_effect = Exception("Network error")

        downloader = VideoDownloader()
        result = downloader.get_available_subtitles("https://youtube.com/watch?v=test")

        assert isinstance(result, tuple)
        subtitles, meta = result
        assert subtitles == {}
        assert meta == {}


class TestDownloadSubtitle:
    """Tests for subtitle download functionality."""

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_download_subtitle_creates_file(self, mock_youtube_dl):
        """Test that download_subtitle creates a subtitle file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test.srt"

            # Mock yt-dlp download - create the expected file
            mock_instance = MagicMock()
            mock_youtube_dl.return_value.__enter__.return_value = mock_instance

            # Simulate yt-dlp creating the subtitle file
            def create_subtitle_file(urls):
                # yt-dlp creates: base_path.lang.srt
                subtitle_path = f"{tmpdir}/test.en.srt"
                Path(subtitle_path).write_text("Mock subtitle content")

            mock_instance.download.side_effect = create_subtitle_file

            downloader = VideoDownloader()
            result = downloader.download_subtitle(
                "https://youtube.com/watch?v=test",
                "en",
                output_path
            )

            # Should call yt-dlp download
            mock_instance.download.assert_called_once()
            assert result == output_path
            # Verify file was created
            assert Path(output_path).exists()

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_download_subtitle_correct_language(self, mock_youtube_dl):
        """Test that download_subtitle downloads the correct language."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test.srt"

            # Mock yt-dlp
            mock_instance = MagicMock()
            mock_youtube_dl.return_value.__enter__.return_value = mock_instance

            # Simulate yt-dlp creating the subtitle file for Spanish
            def create_subtitle_file(urls):
                subtitle_path = f"{tmpdir}/test.es.srt"
                Path(subtitle_path).write_text("Mock Spanish subtitle content")

            mock_instance.download.side_effect = create_subtitle_file

            downloader = VideoDownloader()
            downloader.download_subtitle(
                "https://youtube.com/watch?v=test",
                "es",
                output_path
            )

            # Verify yt-dlp was configured with correct language
            assert mock_youtube_dl.called
            # Verify the file was created
            assert Path(output_path).exists()

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_download_subtitle_raises_on_invalid_lang(self, mock_youtube_dl):
        """Test that download_subtitle raises exception for invalid language."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test.srt"

            # Mock yt-dlp to raise an error
            mock_instance = MagicMock()
            mock_youtube_dl.return_value.__enter__.return_value = mock_instance
            mock_instance.download.side_effect = Exception("Language not available")

            downloader = VideoDownloader()

            with pytest.raises(Exception):
                downloader.download_subtitle(
                    "https://youtube.com/watch?v=test",
                    "invalid",
                    output_path
                )


class TestGetVideoInfo:
    """Tests for getting video metadata without downloading."""

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_get_video_info_without_download(self, mock_youtube_dl):
        """Test that get_video_info doesn't download the video."""
        # Mock yt-dlp
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            'title': 'Test Video',
            'id': 'abc123',
            'duration': 120.5,
            'extractor': 'youtube'
        }

        downloader = VideoDownloader()
        result = downloader.get_video_info("https://youtube.com/watch?v=test")

        # Verify extract_info was called with download=False
        mock_instance.extract_info.assert_called_once_with(
            "https://youtube.com/watch?v=test",
            download=False
        )

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_get_video_info_returns_required_fields(self, mock_youtube_dl):
        """Test that get_video_info returns all required fields."""
        # Mock yt-dlp
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            'title': 'Test Video Title',
            'id': 'xyz789',
            'duration': 300.0,
            'extractor': 'youtube'
        }

        downloader = VideoDownloader()
        result = downloader.get_video_info("https://youtube.com/watch?v=test")

        # Verify all required fields are present
        assert 'title' in result
        assert 'video_id' in result
        assert 'duration' in result
        assert 'platform' in result

        # Verify values
        assert result['title'] == 'Test Video Title'
        assert result['video_id'] == 'xyz789'
        assert result['duration'] == 300.0
        assert result['platform'] == 'youtube'

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_get_video_info_includes_channel(self, mock_youtube_dl):
        """Test that get_video_info returns channel field."""
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            'title': 'Test Video',
            'id': 'abc123',
            'duration': 120.5,
            'extractor': 'youtube',
            'channel': 'Test Channel',
        }

        downloader = VideoDownloader()
        result = downloader.get_video_info("https://youtube.com/watch?v=test")

        assert result['channel'] == 'Test Channel'

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_get_video_info_missing_channel_returns_none(self, mock_youtube_dl):
        """Test that get_video_info handles missing channel gracefully."""
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            'title': 'Test Video',
            'id': 'abc123',
            'duration': 120.5,
            'extractor': 'youtube',
        }

        downloader = VideoDownloader()
        result = downloader.get_video_info("https://youtube.com/watch?v=test")

        assert result['channel'] is None
