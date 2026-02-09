"""
Caching utility functions.
"""

import os
from pathlib import Path
import hashlib


class SimpleCache:
    """Simple in-memory cache with file-based backing."""

    def __init__(self, cache_dir: Path):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}

    def get(self, key: str):
        """Get value from cache."""
        if key in self.memory_cache:
            return self.memory_cache[key]

        cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
        if cache_file.exists():
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                    self.memory_cache[key] = value
                    return value
            except:
                pass
        return None

    def set(self, key: str, value):
        """Set value in cache."""
        self.memory_cache[key] = value
        cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except:
            pass
