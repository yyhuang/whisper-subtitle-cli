import re
import os
from pathlib import Path
from typing import Optional, Dict
import yt_dlp


def is_url(input_str: str) -> bool:
    """
    Detect if input string is a URL.

    Args:
        input_str: String to check

    Returns:
        True if input is a URL, False otherwise
    """
    url_pattern = re.compile(
        r'^(?:http|https)://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(input_str))


class VideoDownloader:
    """Downloads videos from URLs using yt-dlp."""

    def __init__(self, download_dir: str = "/tmp"):
        """
        Initialize video downloader.

        Args:
            download_dir: Directory to save downloaded videos (default: /tmp)
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, quiet: bool = False) -> Dict[str, any]:
        """
        Download video from URL.

        Args:
            url: Video URL to download
            quiet: Suppress yt-dlp output (default: False)

        Returns:
            Dictionary with:
                - file_path: Path to downloaded video file
                - title: Video title (sanitized for filename)
                - video_id: Platform video ID
                - duration: Duration in seconds
                - platform: Platform name (youtube, vimeo, etc.)

        Raises:
            Exception: If download fails
        """
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',  # Best quality
            'outtmpl': str(self.download_dir / '%(id)s.%(ext)s'),  # Use video ID as filename
            'quiet': quiet,
            'no_warnings': quiet,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video information
                info = ydl.extract_info(url, download=True)

                # Get the actual file path
                file_path = ydl.prepare_filename(info)

                # Return video information
                return {
                    'file_path': file_path,
                    'title': info.get('title', 'Unknown'),
                    'video_id': info.get('id', 'unknown'),
                    'duration': info.get('duration', 0.0),
                    'platform': info.get('extractor', 'unknown').lower()
                }

        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")

    def is_supported_url(self, url: str) -> bool:
        """
        Check if URL is supported (i.e., is a valid URL format).

        Args:
            url: URL string to check

        Returns:
            True if URL format is valid, False otherwise
        """
        return is_url(url)

    @staticmethod
    def sanitize_filename(title: str, max_length: int = 200) -> str:
        """
        Sanitize video title for use as filename.

        Removes or replaces characters that are invalid in filenames
        and truncates to maximum length.

        Args:
            title: Video title to sanitize
            max_length: Maximum filename length (default: 200)

        Returns:
            Sanitized filename string
        """
        # Replace invalid filename characters with underscores
        # Invalid chars: / \ : * ? " < > |
        invalid_chars = r'[/\\:*?"<>|]'
        sanitized = re.sub(invalid_chars, '_', title)

        # Remove any leading/trailing whitespace and underscores
        sanitized = sanitized.strip().strip('_')

        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].strip()

        # If somehow we end up with an empty string, use a default
        if not sanitized:
            sanitized = "video"

        return sanitized
