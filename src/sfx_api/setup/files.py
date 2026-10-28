"""Private directories and atomic file replacement for what ``setup`` writes."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

PRIVATE_DIR = 0o700


def make_private_dirs(path: Path) -> None:
    """Create ``path`` and any missing parents, each 0700 (``mkdir(parents=True)`` would give
    the parents the default mode)."""
    missing: list[Path] = []
    current = path
    while not current.exists() and current != current.parent:
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(mode=PRIVATE_DIR, exist_ok=True)
        directory.chmod(PRIVATE_DIR)  # the umask may have taken bits away


def write_atomic(path: Path, data: bytes, mode: int = 0o600) -> None:
    """Write next to ``path`` and rename over it, so a reader never sees half a file."""
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        Path(tmp_name).chmod(mode)
        Path(tmp_name).replace(path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            Path(tmp_name).unlink()
