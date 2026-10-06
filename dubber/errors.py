"""Custom exception hierarchy for the Dubber tool."""

class DubberError(Exception):
    """Base exception for all Dubber pipeline errors."""
    pass


class DependencyMissingError(DubberError):
    """Raised when a required external tool (e.g., ffmpeg) is missing or unexecutable."""
    pass


class DownloadError(DubberError):
    """Raised when downloading video or audio fails."""
    pass


class UnsupportedLanguageError(DubberError):
    """Raised when the detected source language is not supported."""
    pass


class TranscriptionError(DubberError):
    """Raised when speech recognition fails."""
    pass


class TranslationError(DubberError):
    """Raised when text translation fails."""
    pass


class TTSError(DubberError):
    """Raised when speech synthesis fails."""
    pass


class AlignmentError(DubberError):
    """Raised when audio alignment / fitting fails."""
    pass


class AudioProcessingError(DubberError):
    """Raised when audio manipulation, timeline assembly, or muxing fails."""
    pass


class VRAMExhaustionError(DubberError):
    """Raised when GPU VRAM is exhausted and fallback fails."""
    pass
