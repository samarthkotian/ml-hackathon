"""Validate trip names and keep dataset/output paths within their chosen directory."""
from pathlib import Path
import re

TRIP_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"


def validate_trip_id(trip_id: str) -> str:
    if re.fullmatch(TRIP_ID_PATTERN, trip_id) is None:
        raise ValueError("Trip ID must be 1-64 letters, digits, underscores or hyphens, starting with a letter or digit.")
    return trip_id


def safe_child(directory: str, name: str) -> str:
    """Reject traversal and existing symlinks that lead outside the directory."""
    root = Path(directory).resolve()
    child = (root / name).resolve()
    if not child.is_relative_to(root) or child == root:
        raise ValueError("Path must stay inside its configured directory.")
    return str(child)
