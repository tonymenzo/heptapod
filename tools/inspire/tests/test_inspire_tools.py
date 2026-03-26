#!/usr/bin/env python3
"""
# test_inspire_tools.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for INSPIRE HEP API tools.

Run with:
    python test_inspire_tools.py
"""

import sys
import json
import os
import tempfile
import shutil
from pathlib import Path

# Add repo root to path
SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent.parent
REPO_ROOT = TOOL_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.inspire.query_builder import QueryBuilder
from tools.inspire.cache import InspireCache
from tools.inspire.library import InspireLibrary
from tools.inspire.data_classes import (
    PaperInfo, AuthorInfo, InstitutionInfo, SearchResults
)


# ==================== Query Builder Tests ==================== #

def test_query_builder():
    """Test natural language to SPIRES query conversion."""
    print("=" * 60)
    print("Testing QueryBuilder")
    print("=" * 60)

    qb = QueryBuilder()
    all_passed = True

    # Passthrough SPIRES queries
    cases = [
        ("a witten", "a witten"),
        ("t supersymmetry", "t supersymmetry"),
        ("citedby:recid:451647", "citedby:recid:451647"),
    ]
    for query, expected in cases:
        result = qb.build(query)
        ok = result == expected
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: passthrough '{query}' -> '{result}'")

    # Author extraction
    result = qb.build("papers by Witten")
    ok = "a witten" in result.lower()
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: author extraction 'papers by Witten' -> '{result}'")

    result = qb.build("papers by Edward Witten")
    ok = "a edward witten" in result.lower()
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: full name 'papers by Edward Witten' -> '{result}'")

    # Topic extraction
    result = qb.build("papers on string theory")
    ok = "t string theory" in result.lower()
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: topic 'papers on string theory' -> '{result}'")

    result = qb.build("papers about supersymmetry")
    ok = "t supersymmetry" in result.lower()
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: topic 'papers about supersymmetry' -> '{result}'")

    # Combined author + topic
    result = qb.build("papers by Witten on string theory")
    ok = "a witten" in result.lower() and "t string theory" in result.lower() and "and" in result.lower()
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: combined 'papers by Witten on string theory' -> '{result}'")

    # Highly cited
    result = qb.build("highly cited papers on supersymmetry")
    ok = "topcite" in result.lower() and "t supersymmetry" in result.lower()
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: highly cited -> '{result}'")

    # Citation threshold
    result = qb.build("papers with 500+ citations")
    ok = "topcite 500+" in result
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: citation threshold -> '{result}'")

    # Citation/reference query builders
    ok = qb.build_citation_query("451647") == "citedby:recid:451647"
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: build_citation_query")

    ok = qb.build_reference_query("451647") == "refersto:recid:451647"
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: build_reference_query")

    ok = qb.build_author_papers_query("witten") == "a witten"
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: build_author_papers_query")

    print()
    return all_passed


# ==================== Cache Tests ==================== #

def test_cache():
    """Test the caching system."""
    print("=" * 60)
    print("Testing InspireCache")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp()
    all_passed = True

    try:
        cache_file = os.path.join(temp_dir, "test_cache.json")
        cache = InspireCache(cache_file, ttl_hours=1)

        # Set and get
        data = {"test": "data", "nested": {"key": "value"}}
        cache.set("test_key", data)
        retrieved = cache.get("test_key")
        ok = retrieved == data
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: set and get")

        # Cache miss
        ok = cache.get("nonexistent") is None
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: cache miss returns None")

        # Persistence
        cache.set("persistent", {"value": 42})
        new_cache = InspireCache(cache_file, ttl_hours=1)
        ok = new_cache.get("persistent") == {"value": 42}
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: persistence to disk")

        # Author ID mapping
        cache.set_author_id("Edward Witten", "E.Witten.1")
        ok = cache.get_author_id("Edward Witten") == "E.Witten.1" and cache.get_author_id("Unknown") is None
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: author ID mapping")

        # Stats
        cache2 = InspireCache(os.path.join(temp_dir, "test_cache2.json"), ttl_hours=1)
        cache2.set("key1", {"data": 1})
        cache2.set("key2", {"data": 2})
        cache2.set_author_id("Test Author", "Test.Author.1")
        stats = cache2.stats()
        ok = stats["entries"] == 2 and stats["author_mappings"] == 1
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: stats")

        # Clear
        cache2.clear()
        ok = cache2.get("key1") is None and cache2.get_author_id("Test Author") is None
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: clear")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print()
    return all_passed


# ==================== Library Tests ==================== #

def test_library():
    """Test the reading list and notes system."""
    print("=" * 60)
    print("Testing InspireLibrary")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp()
    all_passed = True

    try:
        library_file = os.path.join(temp_dir, "test_library.json")
        library = InspireLibrary(library_file)

        # Add to reading list
        entry = library.add_to_reading_list(
            "451647",
            title="String theory dynamics",
            authors=["Witten, Edward"],
            tags=["string-theory"]
        )
        ok = entry.recid == "451647" and entry.title == "String theory dynamics" and "string-theory" in entry.tags
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: add to reading list")

        # Remove from reading list
        library2 = InspireLibrary(os.path.join(temp_dir, "lib2.json"))
        library2.add_to_reading_list("451647")
        ok = library2.is_in_reading_list("451647")
        removed = library2.remove_from_reading_list("451647")
        ok = ok and removed and not library2.is_in_reading_list("451647")
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: remove from reading list")

        # Get reading list with tag filter
        library3 = InspireLibrary(os.path.join(temp_dir, "lib3.json"))
        library3.add_to_reading_list("1", tags=["tag-a"])
        library3.add_to_reading_list("2", tags=["tag-b"])
        library3.add_to_reading_list("3", tags=["tag-a", "tag-b"])
        all_entries = library3.get_reading_list()
        tag_a = library3.get_reading_list(tag_filter="tag-a")
        ok = len(all_entries) == 3 and len(tag_a) == 2
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: get reading list with tag filter")

        # Get all tags
        tags = library3.get_all_tags()
        ok = set(tags) == {"tag-a", "tag-b"}
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: get all tags")

        # Add note
        note = library.add_note("451647", "Great paper on string theory")
        ok = note.note == "Great paper on string theory" and note.created_at is not None
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: add note")

        # Get notes
        library.add_note("451647", "Second note")
        notes = library.get_notes("451647")
        ok = len(notes) == 2 and notes[0].note == "Great paper on string theory" and notes[1].note == "Second note"
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: get notes")

        # Search notes
        library4 = InspireLibrary(os.path.join(temp_dir, "lib4.json"))
        library4.add_note("1", "String theory is fascinating")
        library4.add_note("2", "Supersymmetry breaking")
        library4.add_note("3", "More string theory notes")
        results = library4.search_notes("string theory")
        ok = len(results) == 2
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: search notes")

        # Persistence
        library5 = InspireLibrary(os.path.join(temp_dir, "lib5.json"))
        library5.add_to_reading_list("451647", tags=["test"])
        library5.add_note("451647", "Test note")
        new_library = InspireLibrary(os.path.join(temp_dir, "lib5.json"))
        ok = new_library.is_in_reading_list("451647") and len(new_library.get_notes("451647")) == 1
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: persistence")

        # Stats
        library6 = InspireLibrary(os.path.join(temp_dir, "lib6.json"))
        library6.add_to_reading_list("1", tags=["a"])
        library6.add_to_reading_list("2", tags=["b"])
        library6.add_note("1", "Note 1")
        library6.add_note("1", "Note 2")
        stats = library6.stats()
        ok = stats["reading_list_count"] == 2 and stats["papers_with_notes"] == 1 and stats["total_notes"] == 2
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: stats")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print()
    return all_passed


# ==================== Data Class Tests ==================== #

def test_data_classes():
    """Test data class serialization."""
    print("=" * 60)
    print("Testing data classes")
    print("=" * 60)

    all_passed = True

    # PaperInfo to_dict
    paper = PaperInfo(
        recid="451647",
        title="String theory dynamics",
        authors=["Witten, Edward"],
        citation_count=3500
    )
    d = paper.to_dict()
    ok = d["recid"] == "451647" and d["title"] == "String theory dynamics" and d["citation_count"] == 3500
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: PaperInfo to_dict")

    # PaperInfo str
    paper2 = PaperInfo(
        recid="451647",
        title="String theory dynamics",
        authors=["Witten, Edward"],
        year=1995
    )
    s = str(paper2)
    ok = "String theory dynamics" in s and "Witten" in s
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: PaperInfo str representation")

    # SearchResults to_dict
    results = SearchResults(
        total=100,
        query="test",
        papers=[
            PaperInfo(recid="1", title="Paper 1", citation_count=10),
            PaperInfo(recid="2", title="Paper 2", citation_count=20)
        ]
    )
    d = results.to_dict()
    ok = d["total"] == 100 and d["returned"] == 2 and len(d["papers"]) == 2
    if not ok:
        all_passed = False
    print(f"  {'PASS' if ok else 'FAIL'}: SearchResults to_dict")

    print()
    return all_passed


# ==================== Integration Tests (API) ==================== #

def test_integration():
    """Integration tests that hit the real INSPIRE API (slow)."""
    print("=" * 60)
    print("Testing INSPIRE API integration")
    print("=" * 60)

    from tools.inspire import search_papers, get_paper, get_bibtex, search_authors, search_conferences

    all_passed = True

    # Search papers
    try:
        papers = search_papers("a witten", size=5)
        ok = len(papers) > 0 and all(hasattr(p, 'recid') for p in papers)
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: search papers ({len(papers)} results)")
    except Exception as e:
        all_passed = False
        print(f"  FAIL: search papers - {e}")

    # Get specific paper
    try:
        paper = get_paper("451647")
        ok = paper.recid == "451647"
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: get paper (recid 451647)")
    except Exception as e:
        all_passed = False
        print(f"  FAIL: get paper - {e}")

    # Get BibTeX
    try:
        bibtex = get_bibtex("451647")
        ok = "@" in bibtex
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: get bibtex")
    except Exception as e:
        all_passed = False
        print(f"  FAIL: get bibtex - {e}")

    # Search authors
    try:
        authors = search_authors("Witten", size=5)
        ok = len(authors) > 0
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: search authors ({len(authors)} results)")
    except Exception as e:
        all_passed = False
        print(f"  FAIL: search authors - {e}")

    # Search conferences
    try:
        conferences = search_conferences("Strings", size=5)
        ok = len(conferences) > 0
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: search conferences ({len(conferences)} results)")
    except Exception as e:
        all_passed = False
        print(f"  FAIL: search conferences - {e}")

    # Natural language query end-to-end
    try:
        papers = search_papers("papers by Witten on strings", size=5)
        ok = len(papers) > 0
        if not ok:
            all_passed = False
        print(f"  {'PASS' if ok else 'FAIL'}: natural language query ({len(papers)} results)")
    except Exception as e:
        all_passed = False
        print(f"  FAIL: natural language query - {e}")

    print()
    return all_passed


# ==================== Runner ==================== #

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("INSPIRE HEP API Tool Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Query Builder", test_query_builder),
        ("Cache", test_cache),
        ("Library", test_library),
        ("Data Classes", test_data_classes),
        ("API Integration", test_integration),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results.append((name, "ERROR"))

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, status in results:
        print(f"  {status}: {name}")

    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
