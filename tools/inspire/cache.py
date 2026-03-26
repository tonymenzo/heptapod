"""
# cache.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Local cache for INSPIRE API responses to reduce API calls.

Provides TTL-based caching with automatic pruning and quick lookups
for author IDs and paper metadata.
"""

import json
import hashlib
import time
from typing import Optional, Dict, Any
from pathlib import Path


class InspireCache:
    """
    Simple file-based cache for INSPIRE API responses.

    Stores recently accessed data to reduce API calls and improve
    response times. Supports TTL-based expiration.

    Cache structure:
    {
        "version": "1.0",
        "entries": {
            "<cache_key_hash>": {
                "data": {...},
                "timestamp": 1234567890.0,
                "key_preview": "literature:q=witten..."
            }
        },
        "author_id_map": {
            "E. Witten": "Edward.Witten.1",
            "Edward Witten": "Edward.Witten.1",
            ...
        },
        "recid_metadata_map": {
            "451647": {
                "title": "String theory dynamics in various dimensions",
                "authors": ["Witten, Edward"],
                "cached_at": 1234567890.0
            }
        }
    }

    Example:
        cache = InspireCache()
        cache.set("literature:q=witten", response_data)
        cached = cache.get("literature:q=witten")
    """

    CACHE_VERSION = "1.0"
    DEFAULT_PATH = "~/.heptapod/inspire_cache.json"
    MAX_ENTRIES = 1000  # Limit cache size

    def __init__(
        self,
        cache_file: Optional[str] = None,
        ttl_hours: int = 24
    ):
        """
        Initialize cache.

        Args:
            cache_file: Path to cache file. Defaults to ~/.heptapod/inspire_cache.json
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_file = Path(cache_file or self.DEFAULT_PATH).expanduser()
        self.ttl_seconds = ttl_hours * 3600
        self._cache = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    if data.get("version") == self.CACHE_VERSION:
                        return data
            except (json.JSONDecodeError, IOError):
                pass

        # Initialize empty cache
        return {
            "version": self.CACHE_VERSION,
            "entries": {},
            "author_id_map": {},
            "recid_metadata_map": {}
        }

    def _save(self):
        """Save cache to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self._cache, f, indent=2)

    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached value if exists and not expired.

        Args:
            key: Cache key (e.g., "literature:q=witten")

        Returns:
            Cached data dictionary or None if not found/expired
        """
        key_hash = self._hash_key(key)
        entry = self._cache["entries"].get(key_hash)

        if entry is None:
            return None

        # Check TTL
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            del self._cache["entries"][key_hash]
            self._save()
            return None

        return entry["data"]

    def set(self, key: str, data: Dict[str, Any]):
        """
        Store value in cache.

        Args:
            key: Cache key
            data: Data to cache
        """
        key_hash = self._hash_key(key)

        # Prune if over limit
        if len(self._cache["entries"]) >= self.MAX_ENTRIES:
            self._prune_oldest()

        self._cache["entries"][key_hash] = {
            "data": data,
            "timestamp": time.time(),
            "key_preview": key[:100]  # For debugging
        }

        # Extract and cache metadata for quick lookups
        self._extract_metadata(data)

        self._save()

    def _prune_oldest(self):
        """Remove oldest 20% of entries."""
        entries = self._cache["entries"]
        if not entries:
            return

        sorted_keys = sorted(
            entries.keys(),
            key=lambda k: entries[k]["timestamp"]
        )
        prune_count = max(1, len(sorted_keys) // 5)
        for key in sorted_keys[:prune_count]:
            del entries[key]

    def _extract_metadata(self, data: Dict[str, Any]):
        """Extract author IDs and paper metadata for quick lookups."""
        # Handle single record responses
        if "metadata" in data:
            self._extract_from_metadata(data["metadata"])

        # Handle search results
        if "hits" in data and "hits" in data["hits"]:
            for hit in data["hits"]["hits"]:
                if "metadata" in hit:
                    self._extract_from_metadata(hit["metadata"])

    def _extract_from_metadata(self, metadata: Dict[str, Any]):
        """Extract metadata from a single record."""
        # Paper metadata
        if "control_number" in metadata and "titles" in metadata:
            recid = str(metadata["control_number"])
            self._cache["recid_metadata_map"][recid] = {
                "title": metadata.get("titles", [{}])[0].get("title", ""),
                "authors": [
                    a.get("full_name", "")
                    for a in metadata.get("authors", [])[:5]
                ],
                "cached_at": time.time()
            }

        # Author ID mapping from author records
        if "ids" in metadata and "name" in metadata:
            bai = None
            for id_entry in metadata.get("ids", []):
                if id_entry.get("schema") == "INSPIRE BAI":
                    bai = id_entry.get("value")
                    break

            if bai:
                name_data = metadata.get("name", {})
                # Try preferred name, then value
                name = name_data.get("preferred_name") or name_data.get("value", "")
                if name:
                    self._cache["author_id_map"][name] = bai

                # Also map name variants
                for variant in name_data.get("name_variants", []):
                    if variant:
                        self._cache["author_id_map"][variant] = bai

    def get_author_id(self, name: str) -> Optional[str]:
        """
        Look up cached author ID by name.

        Args:
            name: Author name (any variant)

        Returns:
            INSPIRE BAI if found in cache, else None
        """
        return self._cache["author_id_map"].get(name)

    def set_author_id(self, name: str, bai: str):
        """
        Store author ID mapping.

        Args:
            name: Author name
            bai: INSPIRE BAI
        """
        self._cache["author_id_map"][name] = bai
        self._save()

    def get_paper_metadata(self, recid: str) -> Optional[Dict[str, Any]]:
        """
        Get cached paper metadata for quick lookups.

        Args:
            recid: INSPIRE record ID

        Returns:
            Dict with title and authors if cached, else None
        """
        return self._cache["recid_metadata_map"].get(recid)

    def clear(self):
        """Clear all cache entries."""
        self._cache = {
            "version": self.CACHE_VERSION,
            "entries": {},
            "author_id_map": {},
            "recid_metadata_map": {}
        }
        self._save()

    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache["entries"]),
            "author_mappings": len(self._cache["author_id_map"]),
            "paper_metadata": len(self._cache["recid_metadata_map"]),
        }
