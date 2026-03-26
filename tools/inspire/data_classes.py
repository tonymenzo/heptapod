"""
# data_classes.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Data classes for INSPIRE HEP API responses.

Provides structured representations of papers, authors, institutions,
conferences, journals, and experiments from the INSPIRE database.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class PaperInfo:
    """Complete paper information from INSPIRE."""

    recid: str  # INSPIRE record ID
    title: str
    authors: List[str] = field(default_factory=list)  # Author names
    abstract: Optional[str] = None
    arxiv_id: Optional[str] = None
    arxiv_category: Optional[str] = None
    doi: Optional[str] = None
    publication_info: Optional[str] = None  # e.g., "Phys.Rev.D 106 (2022)"
    journal: Optional[str] = None
    volume: Optional[str] = None
    year: Optional[int] = None
    pages: Optional[str] = None
    citation_count: int = 0
    citation_count_without_self: int = 0
    date: Optional[str] = None  # Publication/preprint date
    inspire_url: Optional[str] = None
    arxiv_url: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    collaborations: List[str] = field(default_factory=list)
    document_type: Optional[str] = None  # article, thesis, proceedings, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values and empty lists."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None and v != [] and v != 0:
                result[k] = v
            elif k in ('citation_count', 'citation_count_without_self'):
                result[k] = v  # Always include citation counts
        return result

    def __str__(self) -> str:
        """Human-readable string representation."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += f" et al. ({len(self.authors)} authors)"
        return f"{self.title} - {authors_str} ({self.year or 'n.d.'})"


@dataclass
class AuthorInfo:
    """Author information from INSPIRE."""

    inspire_id: str  # INSPIRE author ID (control number)
    bai: Optional[str] = None  # BAI (INSPIRE author identifier)
    name: str = ""
    native_name: Optional[str] = None
    name_variants: List[str] = field(default_factory=list)
    affiliations: List[str] = field(default_factory=list)
    current_institution: Optional[str] = None
    orcid: Optional[str] = None
    email: Optional[str] = None
    paper_count: int = 0
    citation_count: int = 0
    h_index: Optional[int] = None
    inspire_url: Optional[str] = None
    positions: List[str] = field(default_factory=list)  # Current/past positions
    arxiv_categories: List[str] = field(default_factory=list)  # Primary categories

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values and empty lists."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None and v != [] and v != 0:
                result[k] = v
            elif k in ('paper_count', 'citation_count'):
                result[k] = v
        return result

    def __str__(self) -> str:
        """Human-readable string representation."""
        affil = f" ({self.current_institution})" if self.current_institution else ""
        return f"{self.name}{affil}"


@dataclass
class InstitutionInfo:
    """Institution information from INSPIRE."""

    inspire_id: str  # INSPIRE institution ID
    name: str = ""
    name_variants: List[str] = field(default_factory=list)
    country: Optional[str] = None
    country_code: Optional[str] = None
    city: Optional[str] = None
    address: Optional[str] = None
    inspire_url: Optional[str] = None
    website: Optional[str] = None
    institution_type: Optional[str] = None  # University, Lab, etc.
    parent_institution: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values and empty lists."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None and v != []:
                result[k] = v
        return result

    def __str__(self) -> str:
        """Human-readable string representation."""
        loc = f", {self.city}, {self.country}" if self.city and self.country else ""
        return f"{self.name}{loc}"


@dataclass
class ConferenceInfo:
    """Conference/seminar information from INSPIRE."""

    inspire_id: str  # INSPIRE conference ID
    name: str = ""
    short_name: Optional[str] = None  # Acronym/abbreviation
    series: Optional[str] = None  # Conference series name
    opening_date: Optional[str] = None
    closing_date: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    address: Optional[str] = None
    inspire_url: Optional[str] = None
    website: Optional[str] = None
    cnum: Optional[str] = None  # INSPIRE conference number

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result

    def __str__(self) -> str:
        """Human-readable string representation."""
        date = self.opening_date or ""
        loc = f"{self.city}, {self.country}" if self.city else ""
        return f"{self.name} ({date}) - {loc}"


@dataclass
class JournalInfo:
    """Journal information from INSPIRE."""

    inspire_id: str  # INSPIRE journal ID
    name: str = ""
    short_name: Optional[str] = None  # Abbreviated title
    publisher: Optional[str] = None
    issn: Optional[str] = None
    coden: Optional[str] = None
    inspire_url: Optional[str] = None
    website: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result

    def __str__(self) -> str:
        """Human-readable string representation."""
        short = f" ({self.short_name})" if self.short_name else ""
        return f"{self.name}{short}"


@dataclass
class ExperimentInfo:
    """Experiment information from INSPIRE."""

    inspire_id: str  # INSPIRE experiment ID
    name: str = ""
    long_name: Optional[str] = None
    collaboration: Optional[str] = None
    institutions: List[str] = field(default_factory=list)
    status: Optional[str] = None  # Proposed, Running, Completed, etc.
    description: Optional[str] = None
    inspire_url: Optional[str] = None
    website: Optional[str] = None
    start_date: Optional[str] = None
    accelerator: Optional[str] = None
    experiment_type: Optional[str] = None  # Collider, Fixed Target, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values and empty lists."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None and v != []:
                result[k] = v
        return result

    def __str__(self) -> str:
        """Human-readable string representation."""
        status_str = f" [{self.status}]" if self.status else ""
        return f"{self.name}{status_str}"


@dataclass
class SearchResults:
    """Container for search results with pagination info."""

    total: int = 0  # Total number of results
    query: str = ""  # The query that was executed
    page: int = 1
    size: int = 25
    papers: List[PaperInfo] = field(default_factory=list)
    authors: List[AuthorInfo] = field(default_factory=list)
    institutions: List[InstitutionInfo] = field(default_factory=list)
    conferences: List[ConferenceInfo] = field(default_factory=list)
    journals: List[JournalInfo] = field(default_factory=list)
    experiments: List[ExperimentInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with appropriate results list."""
        result = {
            "total": self.total,
            "query": self.query,
            "page": self.page,
            "size": self.size,
        }

        # Include the non-empty results list
        if self.papers:
            result["papers"] = [p.to_dict() for p in self.papers]
            result["returned"] = len(self.papers)
        elif self.authors:
            result["authors"] = [a.to_dict() for a in self.authors]
            result["returned"] = len(self.authors)
        elif self.institutions:
            result["institutions"] = [i.to_dict() for i in self.institutions]
            result["returned"] = len(self.institutions)
        elif self.conferences:
            result["conferences"] = [c.to_dict() for c in self.conferences]
            result["returned"] = len(self.conferences)
        elif self.journals:
            result["journals"] = [j.to_dict() for j in self.journals]
            result["returned"] = len(self.journals)
        elif self.experiments:
            result["experiments"] = [e.to_dict() for e in self.experiments]
            result["returned"] = len(self.experiments)
        else:
            result["returned"] = 0

        return result


@dataclass
class CitationInfo:
    """Citation analysis information for a paper."""

    paper_recid: str
    paper_title: str
    total_citations: int = 0
    citing_papers: List[PaperInfo] = field(default_factory=list)
    top_citing_papers: List[PaperInfo] = field(default_factory=list)
    citation_trend: Dict[str, int] = field(default_factory=dict)  # year -> count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "paper_recid": self.paper_recid,
            "paper_title": self.paper_title,
            "total_citations": self.total_citations,
        }
        if self.top_citing_papers:
            result["top_citing_papers"] = [p.to_dict() for p in self.top_citing_papers]
        if self.citation_trend:
            result["citation_trend"] = self.citation_trend
        return result


@dataclass
class ReadingListEntry:
    """Entry in the reading list."""

    recid: str
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    added_at: Optional[str] = None  # ISO timestamp
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recid": self.recid,
            "title": self.title,
            "authors": self.authors,
            "added_at": self.added_at,
            "tags": self.tags,
        }


@dataclass
class PaperNote:
    """Note attached to a paper."""

    note: str
    created_at: str  # ISO timestamp
    updated_at: Optional[str] = None  # ISO timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "note": self.note,
            "created_at": self.created_at,
        }
        if self.updated_at:
            result["updated_at"] = self.updated_at
        return result
