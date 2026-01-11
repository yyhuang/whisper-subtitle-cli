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

    def get_available_subtitles(self, url: str) -> Dict[str, Dict]:
        """
        Get list of available subtitles for a video.

        Only returns human-made subtitles (filters out auto-generated).

        Args:
            url: Video URL to check

        Returns:
            Dictionary mapping language codes to subtitle info:
            {
                'en': {'name': 'English', 'ext': 'srt'},
                'es': {'name': 'Spanish', 'ext': 'srt'},
                ...
            }
            Returns empty dict if no manual subtitles available.
        """
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info without downloading
                info = ydl.extract_info(url, download=False)

                # Get manual subtitles only (ignore automatic_captions)
                subtitles = info.get('subtitles', {})

                # Build result dictionary with language names
                result = {}
                for lang_code, sub_list in subtitles.items():
                    if sub_list:  # Check if subtitle list is not empty
                        result[lang_code] = {
                            'name': self._get_language_name(lang_code),
                            'ext': sub_list[0].get('ext', 'srt')
                        }

                return result

        except Exception as e:
            # If we can't get subtitles, return empty dict
            return {}

    def download_subtitle(self, url: str, language: str, output_path: str) -> str:
        """
        Download subtitle file for a specific language.

        Args:
            url: Video URL
            language: Language code (e.g., 'en', 'es')
            output_path: Where to save the subtitle file

        Returns:
            Path to downloaded subtitle file

        Raises:
            Exception: If download fails
        """
        # Extract base path without extension
        base_path = str(Path(output_path).with_suffix(''))

        ydl_opts = {
            'writesubtitles': True,  # Enable subtitle download
            'subtitleslangs': [language],  # Specific language
            'subtitlesformat': 'srt',  # Force SRT format
            'skip_download': True,  # Don't download video
            'outtmpl': base_path,  # Base path for output
            'quiet': True,
            'no_warnings': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # yt-dlp saves subtitles as: base_path.lang.srt
            # We need to rename it to the output_path
            downloaded_path = f"{base_path}.{language}.srt"
            if os.path.exists(downloaded_path):
                # Rename to the desired output path
                os.rename(downloaded_path, output_path)
                return output_path
            else:
                raise Exception(f"Subtitle file not found after download: {downloaded_path}")

        except Exception as e:
            raise Exception(f"Subtitle download failed: {str(e)}")

    def get_video_info(self, url: str) -> Dict[str, any]:
        """
        Get video metadata without downloading.

        Args:
            url: Video URL

        Returns:
            Dictionary with title, video_id, duration, platform
        """
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'video_id': info.get('id', 'unknown'),
                    'duration': info.get('duration', 0.0),
                    'platform': info.get('extractor', 'unknown').lower()
                }
        except Exception as e:
            raise Exception(f"Failed to get video info: {str(e)}")

    def _get_language_name(self, lang_code: str) -> str:
        """
        Get human-readable language name from language code.

        Args:
            lang_code: ISO 639-1 language code

        Returns:
            Language name in English
        """
        # Common language codes
        language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'nl': 'Dutch',
            'pl': 'Polish',
            'tr': 'Turkish',
            'vi': 'Vietnamese',
            'th': 'Thai',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
        }

        return language_names.get(lang_code, lang_code.upper())

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
