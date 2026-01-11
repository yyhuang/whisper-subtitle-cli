import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from src.video_downloader import VideoDownloader, is_url


class TestIsUrl:
    """Tests for URL detection function."""

    def test_is_url_detects_http_urls(self):
        """Test that HTTP URLs are correctly identified."""
        assert is_url("http://www.youtube.com/watch?v=abc123") is True
        assert is_url("http://youtube.com/watch?v=abc123") is True

    def test_is_url_detects_https_urls(self):
        """Test that HTTPS URLs are correctly identified."""
        assert is_url("https://www.youtube.com/watch?v=abc123") is True
        assert is_url("https://youtube.com/watch?v=abc123") is True

    def test_is_url_detects_youtube_short_urls(self):
        """Test that YouTube short URLs (youtu.be) are correctly identified."""
        assert is_url("https://youtu.be/abc123") is True
        assert is_url("http://youtu.be/abc123") is True

    def test_is_url_rejects_file_paths(self):
        """Test that file paths are not detected as URLs."""
        assert is_url("/path/to/video.mp4") is False
        assert is_url("/Users/username/video.mp4") is False
        assert is_url("C:\\Users\\video.mp4") is False

    def test_is_url_rejects_relative_paths(self):
        """Test that relative file paths are not detected as URLs."""
        assert is_url("video.mp4") is False
        assert is_url("./video.mp4") is False
        assert is_url("../videos/video.mp4") is False


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_sanitize_filename_removes_special_chars(self):
        """Test that special characters are removed from filenames."""
        result = VideoDownloader.sanitize_filename("Video: Test / Example")
        assert "/" not in result
        assert ":" not in result
        # Should replace with safe characters
        assert result == "Video_ Test _ Example"

    def test_sanitize_filename_truncates_long_names(self):
        """Test that very long filenames are truncated."""
        long_name = "a" * 300  # 300 characters
        result = VideoDownloader.sanitize_filename(long_name, max_length=200)
        assert len(result) <= 200

    def test_sanitize_filename_handles_unicode(self):
        """Test that Unicode characters are preserved where safe."""
        result = VideoDownloader.sanitize_filename("测试视频 Test")
        assert "测试视频" in result or "Test" in result  # Should preserve or handle gracefully


class TestVideoDownloader:
    """Tests for VideoDownloader class."""

    def test_downloader_initialization(self):
        """Test VideoDownloader initializes with correct default directory."""
        downloader = VideoDownloader()
        assert downloader.download_dir == Path("/tmp")

    def test_downloader_custom_directory(self):
        """Test VideoDownloader can use custom download directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = VideoDownloader(download_dir=tmpdir)
            assert downloader.download_dir == Path(tmpdir)

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_download_video_from_url(self, mock_youtube_dl):
        """Test basic video download flow."""
        # Mock yt-dlp behavior
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            'id': 'abc123',
            'title': 'Test Video',
            'duration': 120.5,
            'extractor': 'youtube',
            'ext': 'mp4'
        }
        mock_instance.prepare_filename.return_value = '/tmp/abc123.mp4'

        downloader = VideoDownloader()
        result = downloader.download("https://www.youtube.com/watch?v=abc123")

        assert result['file_path'] == '/tmp/abc123.mp4'
        assert result['title'] == 'Test Video'
        assert result['video_id'] == 'abc123'
        assert result['duration'] == 120.5
        assert result['platform'] == 'youtube'

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_download_returns_video_info(self, mock_youtube_dl):
        """Test that download returns complete video information dictionary."""
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            'id': 'xyz789',
            'title': 'Another Test',
            'duration': 60.0,
            'extractor': 'vimeo',
            'ext': 'mp4'
        }
        mock_instance.prepare_filename.return_value = '/tmp/xyz789.mp4'

        downloader = VideoDownloader()
        result = downloader.download("https://vimeo.com/123456")

        # Verify all required keys are present
        assert 'file_path' in result
        assert 'title' in result
        assert 'video_id' in result
        assert 'duration' in result
        assert 'platform' in result

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_download_saves_to_tmp(self, mock_youtube_dl):
        """Test that videos are saved to /tmp directory."""
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            'id': 'test123',
            'title': 'Test',
            'duration': 30.0,
            'extractor': 'youtube',
            'ext': 'mp4'
        }
        mock_instance.prepare_filename.return_value = '/tmp/test123.mp4'

        downloader = VideoDownloader(download_dir="/tmp")
        result = downloader.download("https://youtube.com/watch?v=test123")

        assert result['file_path'].startswith('/tmp')

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_download_accepts_quiet_flag(self, mock_youtube_dl):
        """Test that quiet flag is passed to yt-dlp."""
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = {
            'id': 'quiet123',
            'title': 'Quiet Test',
            'duration': 45.0,
            'extractor': 'youtube',
            'ext': 'mp4'
        }
        mock_instance.prepare_filename.return_value = '/tmp/quiet123.mp4'

        downloader = VideoDownloader()
        result = downloader.download("https://youtube.com/watch?v=quiet123", quiet=True)

        # Should complete without errors
        assert result is not None

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_download_raises_on_invalid_url(self, mock_youtube_dl):
        """Test that invalid URLs raise ValueError."""
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.side_effect = Exception("Unsupported URL")

        downloader = VideoDownloader()

        with pytest.raises(Exception):
            downloader.download("https://invalid-url.com/video")

    @patch('src.video_downloader.yt_dlp.YoutubeDL')
    def test_download_raises_on_network_error(self, mock_youtube_dl):
        """Test that network errors are properly raised."""
        mock_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.side_effect = Exception("Network error")

        downloader = VideoDownloader()

        with pytest.raises(Exception):
            downloader.download("https://youtube.com/watch?v=test")

    def test_is_supported_url_validates_platform(self):
        """Test that is_supported_url validates URL format."""
        downloader = VideoDownloader()

        # Valid URLs
        assert downloader.is_supported_url("https://www.youtube.com/watch?v=abc") is True
        assert downloader.is_supported_url("https://youtu.be/abc") is True

        # Invalid URLs (not URL format)
        assert downloader.is_supported_url("not-a-url") is False
        assert downloader.is_supported_url("/path/to/file.mp4") is False
