"""
# library.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Persistent reading list and notes for INSPIRE papers.

Provides storage for bookmarked papers and research notes that
persist across sessions.
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from .data_classes import ReadingListEntry, PaperNote


class InspireLibrary:
    """
    Persistent storage for reading list and paper notes.

    Stores bookmarked papers with tags and research notes in a
    JSON file for persistence across sessions.

    Library structure:
    {
        "version": "1.0",
        "reading_list": [
            {"recid": "451647", "title": "...", "authors": [...],
             "added_at": "...", "tags": ["string-theory", "to-read"]}
        ],
        "notes": {
            "451647": [
                {"note": "Key paper on string dualities",
                 "created_at": "...", "updated_at": "..."}
            ]
        }
    }

    Example:
        library = InspireLibrary()
        library.add_to_reading_list("451647", tags=["important"])
        library.add_note("451647", "Great introduction to the topic")
        entries = library.get_reading_list()
        notes = library.get_notes("451647")
    """

    LIBRARY_VERSION = "1.0"
    DEFAULT_PATH = "~/.heptapod/inspire_library.json"

    def __init__(self, library_file: Optional[str] = None):
        """
        Initialize library.

        Args:
            library_file: Path to library file. Defaults to ~/.heptapod/inspire_library.json
        """
        self.library_file = Path(library_file or self.DEFAULT_PATH).expanduser()
        self._library = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load library from disk."""
        if self.library_file.exists():
            try:
                with open(self.library_file, "r") as f:
                    data = json.load(f)
                    if data.get("version") == self.LIBRARY_VERSION:
                        return data
            except (json.JSONDecodeError, IOError):
                pass

        # Initialize empty library
        return {
            "version": self.LIBRARY_VERSION,
            "reading_list": [],
            "notes": {}
        }

    def _save(self):
        """Save library to disk."""
        self.library_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.library_file, "w") as f:
            json.dump(self._library, f, indent=2)

    def _now_iso(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"

    # ==================== Reading List Methods ==================== #

    def add_to_reading_list(
        self,
        recid: str,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> ReadingListEntry:
        """
        Add a paper to the reading list.

        If the paper is already in the list, updates tags and metadata.

        Args:
            recid: INSPIRE record ID
            title: Paper title (optional, for display)
            authors: Author list (optional, for display)
            tags: List of tags to categorize the paper

        Returns:
            The ReadingListEntry that was added/updated
        """
        # Check if already exists
        existing_idx = None
        for idx, entry in enumerate(self._library["reading_list"]):
            if entry["recid"] == recid:
                existing_idx = idx
                break

        now = self._now_iso()
        tags = tags or []

        if existing_idx is not None:
            # Update existing entry
            entry = self._library["reading_list"][existing_idx]
            if title:
                entry["title"] = title
            if authors:
                entry["authors"] = authors
            # Merge tags
            existing_tags = set(entry.get("tags", []))
            entry["tags"] = list(existing_tags.union(set(tags)))
        else:
            # Add new entry
            entry = {
                "recid": recid,
                "title": title,
                "authors": authors or [],
                "added_at": now,
                "tags": tags
            }
            self._library["reading_list"].append(entry)

        self._save()

        return ReadingListEntry(
            recid=entry["recid"],
            title=entry.get("title"),
            authors=entry.get("authors", []),
            added_at=entry.get("added_at"),
            tags=entry.get("tags", [])
        )

    def remove_from_reading_list(self, recid: str) -> bool:
        """
        Remove a paper from the reading list.

        Args:
            recid: INSPIRE record ID

        Returns:
            True if paper was found and removed, False otherwise
        """
        original_len = len(self._library["reading_list"])
        self._library["reading_list"] = [
            entry for entry in self._library["reading_list"]
            if entry["recid"] != recid
        ]

        if len(self._library["reading_list"]) < original_len:
            self._save()
            return True
        return False

    def get_reading_list(
        self,
        tag_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ReadingListEntry]:
        """
        Get papers from the reading list.

        Args:
            tag_filter: Only return papers with this tag
            limit: Maximum number of entries to return

        Returns:
            List of ReadingListEntry objects
        """
        entries = []
        for entry in self._library["reading_list"]:
            if tag_filter and tag_filter not in entry.get("tags", []):
                continue
            entries.append(ReadingListEntry(
                recid=entry["recid"],
                title=entry.get("title"),
                authors=entry.get("authors", []),
                added_at=entry.get("added_at"),
                tags=entry.get("tags", [])
            ))

        if limit:
            entries = entries[:limit]

        return entries

    def is_in_reading_list(self, recid: str) -> bool:
        """Check if a paper is in the reading list."""
        return any(e["recid"] == recid for e in self._library["reading_list"])

    def get_all_tags(self) -> List[str]:
        """Get all unique tags used in the reading list."""
        tags = set()
        for entry in self._library["reading_list"]:
            tags.update(entry.get("tags", []))
        return sorted(tags)

    def update_tags(self, recid: str, tags: List[str]) -> bool:
        """
        Replace tags for a paper in the reading list.

        Args:
            recid: INSPIRE record ID
            tags: New tag list

        Returns:
            True if paper was found and updated, False otherwise
        """
        for entry in self._library["reading_list"]:
            if entry["recid"] == recid:
                entry["tags"] = tags
                self._save()
                return True
        return False

    # ==================== Notes Methods ==================== #

    def add_note(self, recid: str, note: str) -> PaperNote:
        """
        Add a note to a paper.

        Multiple notes can be added to the same paper.

        Args:
            recid: INSPIRE record ID
            note: The note text

        Returns:
            The PaperNote that was created
        """
        now = self._now_iso()

        if recid not in self._library["notes"]:
            self._library["notes"][recid] = []

        note_entry = {
            "note": note,
            "created_at": now,
            "updated_at": None
        }
        self._library["notes"][recid].append(note_entry)
        self._save()

        return PaperNote(
            note=note,
            created_at=now,
            updated_at=None
        )

    def get_notes(self, recid: str) -> List[PaperNote]:
        """
        Get all notes for a paper.

        Args:
            recid: INSPIRE record ID

        Returns:
            List of PaperNote objects
        """
        notes = self._library["notes"].get(recid, [])
        return [
            PaperNote(
                note=n["note"],
                created_at=n["created_at"],
                updated_at=n.get("updated_at")
            )
            for n in notes
        ]

    def update_note(self, recid: str, note_index: int, new_text: str) -> bool:
        """
        Update an existing note.

        Args:
            recid: INSPIRE record ID
            note_index: Index of the note to update (0-based)
            new_text: New note text

        Returns:
            True if note was found and updated, False otherwise
        """
        notes = self._library["notes"].get(recid, [])
        if 0 <= note_index < len(notes):
            notes[note_index]["note"] = new_text
            notes[note_index]["updated_at"] = self._now_iso()
            self._save()
            return True
        return False

    def delete_note(self, recid: str, note_index: int) -> bool:
        """
        Delete a note.

        Args:
            recid: INSPIRE record ID
            note_index: Index of the note to delete (0-based)

        Returns:
            True if note was found and deleted, False otherwise
        """
        notes = self._library["notes"].get(recid, [])
        if 0 <= note_index < len(notes):
            del notes[note_index]
            if not notes:
                del self._library["notes"][recid]
            self._save()
            return True
        return False

    def search_notes(self, query: str) -> List[Dict[str, Any]]:
        """
        Search notes for matching text.

        Args:
            query: Text to search for (case-insensitive)

        Returns:
            List of dicts with recid, note_index, and note content
        """
        query_lower = query.lower()
        results = []

        for recid, notes in self._library["notes"].items():
            for idx, note in enumerate(notes):
                if query_lower in note["note"].lower():
                    results.append({
                        "recid": recid,
                        "note_index": idx,
                        "note": note["note"],
                        "created_at": note["created_at"]
                    })

        return results

    def get_papers_with_notes(self) -> List[str]:
        """Get list of recids that have notes."""
        return list(self._library["notes"].keys())

    # ==================== Utility Methods ==================== #

    def clear_reading_list(self):
        """Clear all entries from the reading list."""
        self._library["reading_list"] = []
        self._save()

    def clear_notes(self):
        """Clear all notes."""
        self._library["notes"] = {}
        self._save()

    def clear_all(self):
        """Clear entire library."""
        self._library = {
            "version": self.LIBRARY_VERSION,
            "reading_list": [],
            "notes": {}
        }
        self._save()

    def stats(self) -> Dict[str, int]:
        """Get library statistics."""
        total_notes = sum(len(notes) for notes in self._library["notes"].values())
        return {
            "reading_list_count": len(self._library["reading_list"]),
            "papers_with_notes": len(self._library["notes"]),
            "total_notes": total_notes,
            "unique_tags": len(self.get_all_tags())
        }

    def export_reading_list_bibtex_keys(self) -> List[str]:
        """Get list of recids suitable for BibTeX export."""
        return [entry["recid"] for entry in self._library["reading_list"]]


# Singleton library for convenience functions
_library: Optional[InspireLibrary] = None


def get_library() -> InspireLibrary:
    """Get or create singleton library."""
    global _library
    if _library is None:
        _library = InspireLibrary()
    return _library
