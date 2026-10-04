"""Configure a bundled FFmpeg executable for WhisperX."""
import os
import tempfile
from pathlib import Path

import imageio_ffmpeg

_FFMPEG_BIN_DIRECTORY: str | None = None


def ensure_ffmpeg_available() -> str:
    """Put the imageio-ffmpeg executable on PATH and return its path."""
    global _FFMPEG_BIN_DIRECTORY

    executable = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(
            f"FFmpeg do pacote imageio-ffmpeg não está disponível: {executable}"
        )

    if _FFMPEG_BIN_DIRECTORY is None:
        _FFMPEG_BIN_DIRECTORY = tempfile.mkdtemp(prefix="whisperx-ffmpeg-")
        (Path(_FFMPEG_BIN_DIRECTORY) / "ffmpeg").symlink_to(executable)

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    path_entries = [
        entry
        for entry in path_entries
        if entry and entry != _FFMPEG_BIN_DIRECTORY
    ]
    os.environ["PATH"] = os.pathsep.join([_FFMPEG_BIN_DIRECTORY, *path_entries])

    return str(executable)
