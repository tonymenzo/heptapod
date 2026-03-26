"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

INSPIRE HEP API Tool - Search physics literature and perform citation analysis.

This module provides agent-friendly access to the INSPIRE HEP database
for searching papers, authors, institutions, conferences, journals,
experiments, and managing a personal research library.

Quick Start:
    from tools.inspire import search_papers, get_paper, get_author

    # Search for papers
    results = search_papers("papers by Witten on string theory")
    for paper in results[:5]:
        print(f"{paper.title} ({paper.citation_count} citations)")

    # Get paper details
    paper = get_paper("451647")
    print(f"Title: {paper.title}")
    print(f"Abstract: {paper.abstract}")

    # Get author info
    author = get_author("Edward Witten")
    print(f"Name: {author.name}")
    print(f"Institution: {author.current_institution}")

    # Manage reading list
    add_to_reading_list("451647", tags=["string-theory"])
    add_note("451647", "Important foundational paper")

Tool Usage (for agents):
    tool = InspireSearchTool(base_directory="/tmp")
    tool.query = "papers by Witten on string theory"
    tool.size = 10
    result = tool._run()  # Returns JSON
"""

# Main tools
from .inspire_tools import (
    InspireSearchTool,
    InspirePaperTool,
    InspireCitationTool,
    InspireBibTeXTool,
    InspireAuthorTool,
    InspireInstitutionTool,
    InspireConferenceTool,
    InspireJournalTool,
    InspireExperimentTool,
    InspireReadingListTool,
    InspireNotesTool,
    SCHEMA_VERSION
)

# Interface
from .inspire_interface import (
    InspireInterface,
    RateLimiter,
    get_interface
)

# Data classes
from .data_classes import (
    PaperInfo,
    AuthorInfo,
    InstitutionInfo,
    ConferenceInfo,
    JournalInfo,
    ExperimentInfo,
    SearchResults,
    CitationInfo,
    ReadingListEntry,
    PaperNote
)

# Query builder
from .query_builder import QueryBuilder

# Cache
from .cache import InspireCache

# Library
from .library import InspireLibrary, get_library


# ==================== Convenience Functions ==================== #


def search_papers(
    query: str,
    sort: str = "mostcited",
    size: int = 25
) -> list:
    """
    Search INSPIRE for papers.

    Args:
        query: Natural language or SPIRES query
        sort: Sort order ("mostcited", "mostrecent")
        size: Max results

    Returns:
        List of PaperInfo objects
    """
    interface = get_interface()
    qb = QueryBuilder()
    spires_query = qb.build(query)
    results = interface.search_papers(spires_query, sort=sort, size=size)
    return results.papers


def get_paper(recid: str) -> PaperInfo:
    """
    Get paper by INSPIRE record ID.

    Args:
        recid: INSPIRE record ID (e.g., "451647")

    Returns:
        PaperInfo with full paper details
    """
    return get_interface().get_paper(recid)


def get_author(name: str) -> AuthorInfo:
    """
    Get author info by name.

    Args:
        name: Author name (e.g., "Edward Witten")

    Returns:
        AuthorInfo with author details

    Raises:
        ValueError: If author not found
    """
    authors = get_interface().search_authors(name, size=1)
    if not authors:
        raise ValueError(f"Author not found: {name}")
    return authors[0]


def get_citations(recid: str, limit: int = 50) -> list:
    """
    Get papers citing a given paper.

    Args:
        recid: INSPIRE record ID
        limit: Max citations to return

    Returns:
        List of PaperInfo for citing papers
    """
    return get_interface().get_paper_citations(recid, size=limit)


def get_references(recid: str, limit: int = 50) -> list:
    """
    Get papers referenced by a given paper.

    Args:
        recid: INSPIRE record ID
        limit: Max references to return

    Returns:
        List of PaperInfo for referenced papers
    """
    return get_interface().get_paper_references(recid, size=limit)


def get_bibtex(recid: str) -> str:
    """
    Get BibTeX entry for a paper.

    Args:
        recid: INSPIRE record ID

    Returns:
        BibTeX string
    """
    return get_interface().get_bibtex(recid)


def search_authors(query: str, size: int = 10) -> list:
    """
    Search for authors.

    Args:
        query: Author name or query
        size: Max results

    Returns:
        List of AuthorInfo objects
    """
    return get_interface().search_authors(query, size=size)


def search_institutions(query: str, size: int = 10) -> list:
    """
    Search for institutions.

    Args:
        query: Institution name or query
        size: Max results

    Returns:
        List of InstitutionInfo objects
    """
    return get_interface().search_institutions(query, size=size)


def search_conferences(query: str, size: int = 10) -> list:
    """
    Search for conferences.

    Args:
        query: Conference name or query
        size: Max results

    Returns:
        List of ConferenceInfo objects
    """
    return get_interface().search_conferences(query, size=size)


def search_journals(query: str, size: int = 10) -> list:
    """
    Search for journals.

    Args:
        query: Journal name or query
        size: Max results

    Returns:
        List of JournalInfo objects
    """
    return get_interface().search_journals(query, size=size)


def search_experiments(query: str, size: int = 10) -> list:
    """
    Search for experiments.

    Args:
        query: Experiment name or query
        size: Max results

    Returns:
        List of ExperimentInfo objects
    """
    return get_interface().search_experiments(query, size=size)


# ==================== Library Functions ==================== #


def add_to_reading_list(
    recid: str,
    tags: list = None,
    title: str = None,
    authors: list = None
) -> ReadingListEntry:
    """
    Add a paper to your reading list.

    Args:
        recid: INSPIRE record ID
        tags: Tags to categorize the paper
        title: Paper title (auto-fetched if not provided)
        authors: Author list (auto-fetched if not provided)

    Returns:
        ReadingListEntry that was added
    """
    library = get_library()

    # Auto-fetch metadata if not provided
    if title is None:
        try:
            paper = get_paper(recid)
            title = paper.title
            authors = paper.authors[:5] if authors is None else authors
        except Exception:
            pass

    return library.add_to_reading_list(recid, title=title, authors=authors, tags=tags)


def remove_from_reading_list(recid: str) -> bool:
    """
    Remove a paper from your reading list.

    Args:
        recid: INSPIRE record ID

    Returns:
        True if removed, False if not found
    """
    return get_library().remove_from_reading_list(recid)


def get_reading_list(tag_filter: str = None) -> list:
    """
    Get papers in your reading list.

    Args:
        tag_filter: Only return papers with this tag

    Returns:
        List of ReadingListEntry objects
    """
    return get_library().get_reading_list(tag_filter=tag_filter)


def add_note(recid: str, note: str) -> PaperNote:
    """
    Add a note to a paper.

    Args:
        recid: INSPIRE record ID
        note: Note text

    Returns:
        PaperNote that was created
    """
    return get_library().add_note(recid, note)


def get_notes(recid: str) -> list:
    """
    Get all notes for a paper.

    Args:
        recid: INSPIRE record ID

    Returns:
        List of PaperNote objects
    """
    return get_library().get_notes(recid)


def search_notes(query: str) -> list:
    """
    Search notes for matching text.

    Args:
        query: Text to search for

    Returns:
        List of dicts with recid and note info
    """
    return get_library().search_notes(query)


__all__ = [
    # Tools
    "InspireSearchTool",
    "InspirePaperTool",
    "InspireCitationTool",
    "InspireBibTeXTool",
    "InspireAuthorTool",
    "InspireInstitutionTool",
    "InspireConferenceTool",
    "InspireJournalTool",
    "InspireExperimentTool",
    "InspireReadingListTool",
    "InspireNotesTool",
    "SCHEMA_VERSION",

    # Interface
    "InspireInterface",
    "RateLimiter",
    "get_interface",

    # Data classes
    "PaperInfo",
    "AuthorInfo",
    "InstitutionInfo",
    "ConferenceInfo",
    "JournalInfo",
    "ExperimentInfo",
    "SearchResults",
    "CitationInfo",
    "ReadingListEntry",
    "PaperNote",

    # Query builder
    "QueryBuilder",

    # Cache
    "InspireCache",

    # Library
    "InspireLibrary",
    "get_library",

    # Convenience functions
    "search_papers",
    "get_paper",
    "get_author",
    "get_citations",
    "get_references",
    "get_bibtex",
    "search_authors",
    "search_institutions",
    "search_conferences",
    "search_journals",
    "search_experiments",
    "add_to_reading_list",
    "remove_from_reading_list",
    "get_reading_list",
    "add_note",
    "get_notes",
    "search_notes",
]
