"""
File utility functions.
"""

import hashlib
import os
from pathlib import Path
from typing import List


def get_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hexadecimal MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Same path
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_dir(dir_path: Path) -> None:
    """
    Clear all contents of a directory.

    Args:
        dir_path: Directory path to clear
    """
    if dir_path.exists():
        import shutil
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def list_files(directory: Path, pattern: str = "*", recursive: bool = False) -> List[Path]:
    """
    List files in a directory matching a pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search recursively

    Returns:
        List of file paths
    """
    if recursive:
        return list(directory.rglob(pattern))
    return list(directory.glob(pattern))
