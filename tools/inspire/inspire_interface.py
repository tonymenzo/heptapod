"""
# inspire_interface.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

INSPIRE HEP API interface with rate limiting and caching.

Provides methods to search papers, authors, institutions, conferences,
journals, and experiments from the INSPIRE database.
"""

import time
import requests
from typing import Optional, List, Dict, Any

from .cache import InspireCache
from .data_classes import (
    PaperInfo, AuthorInfo, InstitutionInfo, ConferenceInfo,
    JournalInfo, ExperimentInfo, SearchResults, CitationInfo
)

# Rate limit: 15 requests per 5 seconds
RATE_LIMIT_REQUESTS = 15
RATE_LIMIT_WINDOW = 5.0  # seconds
BASE_URL = "https://inspirehep.net/api/"


class RateLimiter:
    """
    Token bucket rate limiter for INSPIRE API.

    INSPIRE allows 15 requests per 5 second window. This class
    tracks request times and waits if necessary to avoid 429 errors.
    """

    def __init__(
        self,
        requests_per_window: int = RATE_LIMIT_REQUESTS,
        window_seconds: float = RATE_LIMIT_WINDOW
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_window: Max requests allowed in window
            window_seconds: Time window in seconds
        """
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.request_times: List[float] = []

    def wait_if_needed(self) -> float:
        """
        Wait if rate limit would be exceeded.

        Returns:
            Time waited in seconds (0 if no wait needed)
        """
        now = time.time()
        # Remove requests outside the window
        self.request_times = [
            t for t in self.request_times
            if now - t < self.window_seconds
        ]

        if len(self.request_times) >= self.requests_per_window:
            # Need to wait until oldest request expires from window
            oldest = min(self.request_times)
            wait_time = self.window_seconds - (now - oldest) + 0.1  # Small buffer
            if wait_time > 0:
                time.sleep(wait_time)
                # Clean up after waiting
                now = time.time()
                self.request_times = [
                    t for t in self.request_times
                    if now - t < self.window_seconds
                ]
                self.request_times.append(now)
                return wait_time

        self.request_times.append(time.time())
        return 0.0


class InspireInterface:
    """
    Interface to INSPIRE HEP API with rate limiting and caching.

    Provides methods to search papers, authors, institutions and
    perform citation analysis. Handles SPIRES-compatible query syntax.

    Example:
        interface = InspireInterface()
        results = interface.search_papers("a witten and t string theory")
        paper = interface.get_paper("451647")
        bibtex = interface.get_bibtex("451647")
    """

    def __init__(
        self,
        cache_file: Optional[str] = None,
        enable_cache: bool = True,
        cache_ttl_hours: int = 24
    ):
        """
        Initialize INSPIRE interface.

        Args:
            cache_file: Path to cache file (default: ~/.heptapod/inspire_cache.json)
            enable_cache: Whether to use caching
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.rate_limiter = RateLimiter()
        self.enable_cache = enable_cache
        self._cache = InspireCache(cache_file, ttl_hours=cache_ttl_hours) if enable_cache else None
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "HEPTAPOD/1.0 (Physics research tool)"
        })

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make rate-limited API request with optional caching.

        Args:
            endpoint: API endpoint (e.g., "literature", "authors/123")
            params: Query parameters
            use_cache: Whether to use cache for this request

        Returns:
            JSON response as dictionary

        Raises:
            requests.HTTPError: On API errors (including 429 rate limit)
        """
        url = f"{BASE_URL}{endpoint}"
        cache_key = f"{endpoint}:{sorted(params.items()) if params else ''}"

        # Check cache
        if use_cache and self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Rate limit
        self.rate_limiter.wait_if_needed()

        # Make request
        response = self._session.get(url, params=params, timeout=30)

        if response.status_code == 429:
            # Rate limited - wait and retry once
            retry_after = int(response.headers.get("Retry-After", 5))
            time.sleep(retry_after)
            self.rate_limiter.wait_if_needed()
            response = self._session.get(url, params=params, timeout=30)

        response.raise_for_status()
        data = response.json()

        # Cache response
        if use_cache and self._cache:
            self._cache.set(cache_key, data)

        return data

    # ==================== Literature Methods ==================== #

    def search_papers(
        self,
        query: str,
        sort: str = "mostrecent",
        size: int = 25,
        page: int = 1,
        fields: Optional[List[str]] = None
    ) -> SearchResults:
        """
        Search INSPIRE literature.

        Args:
            query: SPIRES-compatible query string
                   Examples:
                   - "a witten" (author search)
                   - "t supersymmetry" (title search)
                   - "author:witten and title:strings"
                   - "citedby:recid:451647" (papers citing a work)
            sort: Sort order ("mostrecent", "mostcited")
            size: Results per page (max 1000)
            page: Page number
            fields: Specific fields to return (optimization)

        Returns:
            SearchResults with list of PaperInfo objects
        """
        params: Dict[str, Any] = {
            "q": query,
            "sort": sort,
            "size": min(size, 1000),
            "page": page
        }
        if fields:
            params["fields"] = ",".join(fields)

        data = self._request("literature", params)
        return self._parse_paper_search_results(data, query)

    def get_paper(self, recid: str) -> PaperInfo:
        """
        Get paper details by INSPIRE record ID.

        Args:
            recid: INSPIRE record ID (e.g., "451647")

        Returns:
            PaperInfo with full paper details
        """
        data = self._request(f"literature/{recid}")
        return self._parse_paper(data)

    def get_paper_citations(
        self,
        recid: str,
        size: int = 100,
        sort: str = "mostcited"
    ) -> List[PaperInfo]:
        """
        Get papers that cite a given paper.

        Args:
            recid: INSPIRE record ID
            size: Maximum citations to return
            sort: Sort order

        Returns:
            List of PaperInfo for citing papers
        """
        query = f"citedby:recid:{recid}"
        results = self.search_papers(query, sort=sort, size=size)
        return results.papers

    def get_paper_references(
        self,
        recid: str,
        size: int = 100
    ) -> List[PaperInfo]:
        """
        Get papers referenced by a given paper.

        Args:
            recid: INSPIRE record ID
            size: Maximum references to return

        Returns:
            List of PaperInfo for referenced papers
        """
        query = f"refersto:recid:{recid}"
        results = self.search_papers(query, size=size)
        return results.papers

    def get_bibtex(self, recid: str) -> str:
        """
        Get BibTeX entry for a paper.

        Args:
            recid: INSPIRE record ID

        Returns:
            BibTeX string
        """
        url = f"{BASE_URL}literature/{recid}"
        params = {"format": "bibtex"}

        self.rate_limiter.wait_if_needed()
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.text

    # ==================== Author Methods ==================== #

    def search_authors(
        self,
        query: str,
        size: int = 25
    ) -> List[AuthorInfo]:
        """
        Search for authors.

        Args:
            query: Author name or query (ElasticSearch query string)
            size: Maximum results

        Returns:
            List of AuthorInfo objects
        """
        params = {"q": query, "size": size}
        data = self._request("authors", params)
        return self._parse_author_search_results(data)

    def get_author(self, author_id: str) -> AuthorInfo:
        """
        Get author details by INSPIRE author ID.

        Args:
            author_id: INSPIRE author ID or BAI

        Returns:
            AuthorInfo with full author details
        """
        data = self._request(f"authors/{author_id}")
        return self._parse_author(data)

    def get_author_papers(
        self,
        author_id: str,
        size: int = 100,
        sort: str = "mostcited"
    ) -> List[PaperInfo]:
        """
        Get papers by an author.

        Args:
            author_id: INSPIRE author ID or BAI
            size: Maximum papers to return
            sort: Sort order

        Returns:
            List of PaperInfo
        """
        # First get author to find their BAI
        try:
            author = self.get_author(author_id)
            bai = author.bai or author_id
        except Exception:
            bai = author_id

        query = f"a {bai}"
        results = self.search_papers(query, sort=sort, size=size)
        return results.papers

    # ==================== Institution Methods ==================== #

    def search_institutions(
        self,
        query: str,
        size: int = 25
    ) -> List[InstitutionInfo]:
        """
        Search for institutions.

        Args:
            query: Institution name or query
            size: Maximum results

        Returns:
            List of InstitutionInfo
        """
        params = {"q": query, "size": size}
        data = self._request("institutions", params)
        return self._parse_institution_search_results(data)

    def get_institution(self, institution_id: str) -> InstitutionInfo:
        """
        Get institution details.

        Args:
            institution_id: INSPIRE institution ID

        Returns:
            InstitutionInfo
        """
        data = self._request(f"institutions/{institution_id}")
        return self._parse_institution(data)

    # ==================== Conference Methods ==================== #

    def search_conferences(
        self,
        query: str,
        size: int = 25,
        sort: str = "dateasc"
    ) -> List[ConferenceInfo]:
        """
        Search for conferences.

        Args:
            query: Conference name or query
            size: Maximum results
            sort: Sort order ("dateasc", "datedesc")

        Returns:
            List of ConferenceInfo
        """
        params = {"q": query, "size": size, "sort": sort}
        data = self._request("conferences", params)
        return self._parse_conference_search_results(data)

    def get_conference(self, conference_id: str) -> ConferenceInfo:
        """
        Get conference details.

        Args:
            conference_id: INSPIRE conference ID

        Returns:
            ConferenceInfo
        """
        data = self._request(f"conferences/{conference_id}")
        return self._parse_conference(data)

    # ==================== Journal Methods ==================== #

    def search_journals(
        self,
        query: str,
        size: int = 25
    ) -> List[JournalInfo]:
        """
        Search for journals.

        Args:
            query: Journal name or query
            size: Maximum results

        Returns:
            List of JournalInfo
        """
        params = {"q": query, "size": size}
        data = self._request("journals", params)
        return self._parse_journal_search_results(data)

    def get_journal(self, journal_id: str) -> JournalInfo:
        """
        Get journal details.

        Args:
            journal_id: INSPIRE journal ID

        Returns:
            JournalInfo
        """
        data = self._request(f"journals/{journal_id}")
        return self._parse_journal(data)

    # ==================== Experiment Methods ==================== #

    def search_experiments(
        self,
        query: str,
        size: int = 25
    ) -> List[ExperimentInfo]:
        """
        Search for experiments.

        Args:
            query: Experiment name or query
            size: Maximum results

        Returns:
            List of ExperimentInfo
        """
        params = {"q": query, "size": size}
        data = self._request("experiments", params)
        return self._parse_experiment_search_results(data)

    def get_experiment(self, experiment_id: str) -> ExperimentInfo:
        """
        Get experiment details.

        Args:
            experiment_id: INSPIRE experiment ID

        Returns:
            ExperimentInfo
        """
        data = self._request(f"experiments/{experiment_id}")
        return self._parse_experiment(data)

    # ==================== Parsing Methods ==================== #

    def _parse_paper(self, data: Dict[str, Any]) -> PaperInfo:
        """Parse single paper response into PaperInfo."""
        metadata = data.get("metadata", data)
        recid = str(metadata.get("control_number", data.get("id", "")))

        # Extract title
        titles = metadata.get("titles", [])
        title = titles[0].get("title", "") if titles else ""

        # Extract authors
        authors_data = metadata.get("authors", [])
        authors = [a.get("full_name", "") for a in authors_data]

        # Extract abstract
        abstracts = metadata.get("abstracts", [])
        abstract = abstracts[0].get("value", "") if abstracts else None

        # Extract arXiv info
        arxiv_eprints = metadata.get("arxiv_eprints", [])
        arxiv_id = None
        arxiv_category = None
        if arxiv_eprints:
            arxiv_id = arxiv_eprints[0].get("value")
            categories = arxiv_eprints[0].get("categories", [])
            arxiv_category = categories[0] if categories else None

        # Extract DOI
        dois = metadata.get("dois", [])
        doi = dois[0].get("value") if dois else None

        # Extract publication info
        pub_info = metadata.get("publication_info", [])
        publication_info = None
        journal = None
        volume = None
        year = None
        pages = None
        if pub_info:
            pi = pub_info[0]
            journal = pi.get("journal_title")
            volume = pi.get("journal_volume")
            year = pi.get("year")
            pages = pi.get("page_start")
            if journal:
                parts = [journal]
                if volume:
                    parts.append(volume)
                if year:
                    parts.append(f"({year})")
                if pages:
                    parts.append(pages)
                publication_info = " ".join(parts)

        # Extract citation counts
        citation_count = metadata.get("citation_count", 0)
        citation_count_without_self = metadata.get("citation_count_without_self_citations", 0)

        # Extract date
        date = metadata.get("earliest_date")
        if not date:
            preprint_date = metadata.get("preprint_date")
            date = preprint_date

        # Extract keywords
        keywords_data = metadata.get("keywords", [])
        keywords = [k.get("value", "") for k in keywords_data if k.get("value")]

        # Extract collaborations
        collab_data = metadata.get("collaborations", [])
        collaborations = [c.get("value", "") for c in collab_data]

        # Extract document type
        doc_types = metadata.get("document_type", [])
        document_type = doc_types[0] if doc_types else None

        # Build URLs
        inspire_url = f"https://inspirehep.net/literature/{recid}"
        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None

        return PaperInfo(
            recid=recid,
            title=title,
            authors=authors,
            abstract=abstract,
            arxiv_id=arxiv_id,
            arxiv_category=arxiv_category,
            doi=doi,
            publication_info=publication_info,
            journal=journal,
            volume=volume,
            year=year,
            pages=pages,
            citation_count=citation_count,
            citation_count_without_self=citation_count_without_self,
            date=date,
            inspire_url=inspire_url,
            arxiv_url=arxiv_url,
            keywords=keywords,
            collaborations=collaborations,
            document_type=document_type
        )

    def _parse_paper_search_results(
        self,
        data: Dict[str, Any],
        query: str
    ) -> SearchResults:
        """Parse paper search response into SearchResults."""
        hits = data.get("hits", {})
        total = hits.get("total", 0)
        papers = [self._parse_paper(hit) for hit in hits.get("hits", [])]

        return SearchResults(
            total=total,
            query=query,
            papers=papers
        )

    def _parse_author(self, data: Dict[str, Any]) -> AuthorInfo:
        """Parse single author response into AuthorInfo."""
        metadata = data.get("metadata", data)
        inspire_id = str(metadata.get("control_number", data.get("id", "")))

        # Extract BAI
        bai = None
        for id_entry in metadata.get("ids", []):
            if id_entry.get("schema") == "INSPIRE BAI":
                bai = id_entry.get("value")
                break

        # Extract name
        name_data = metadata.get("name", {})
        name = name_data.get("preferred_name") or name_data.get("value", "")
        native_name = name_data.get("native_names", [None])[0] if name_data.get("native_names") else None
        name_variants = name_data.get("name_variants", [])

        # Extract affiliations
        positions = metadata.get("positions", [])
        affiliations = []
        current_institution = None
        for pos in positions:
            inst = pos.get("institution")
            if inst:
                affiliations.append(inst)
                if pos.get("current"):
                    current_institution = inst

        # Extract ORCID
        orcid = None
        for id_entry in metadata.get("ids", []):
            if id_entry.get("schema") == "ORCID":
                orcid = id_entry.get("value")
                break

        # Extract stats
        # Note: INSPIRE API may not return these directly
        paper_count = metadata.get("stub", False) == False  # Placeholder

        inspire_url = f"https://inspirehep.net/authors/{inspire_id}"

        # Extract arxiv categories
        arxiv_categories = metadata.get("arxiv_categories", [])

        return AuthorInfo(
            inspire_id=inspire_id,
            bai=bai,
            name=name,
            native_name=native_name,
            name_variants=name_variants,
            affiliations=affiliations,
            current_institution=current_institution,
            orcid=orcid,
            inspire_url=inspire_url,
            arxiv_categories=arxiv_categories
        )

    def _parse_author_search_results(self, data: Dict[str, Any]) -> List[AuthorInfo]:
        """Parse author search response."""
        hits = data.get("hits", {})
        return [self._parse_author(hit) for hit in hits.get("hits", [])]

    def _parse_institution(self, data: Dict[str, Any]) -> InstitutionInfo:
        """Parse single institution response."""
        metadata = data.get("metadata", data)
        inspire_id = str(metadata.get("control_number", data.get("id", "")))

        # Extract name
        legacy_icn = metadata.get("legacy_ICN", "")
        name_variants = metadata.get("name_variants", [])
        name_variants_list = [nv.get("value", "") for nv in name_variants] if name_variants else []

        # Primary name is often in institution_hierarchy
        inst_hierarchy = metadata.get("institution_hierarchy", [])
        name = inst_hierarchy[0].get("name", legacy_icn) if inst_hierarchy else legacy_icn

        # Extract address
        addresses = metadata.get("addresses", [])
        country = None
        country_code = None
        city = None
        if addresses:
            addr = addresses[0]
            country = addr.get("country")
            country_code = addr.get("country_code")
            city = addr.get("cities", [None])[0] if addr.get("cities") else None

        # Extract URLs
        urls = metadata.get("urls", [])
        website = urls[0].get("value") if urls else None

        inspire_url = f"https://inspirehep.net/institutions/{inspire_id}"

        return InstitutionInfo(
            inspire_id=inspire_id,
            name=name,
            name_variants=name_variants_list,
            country=country,
            country_code=country_code,
            city=city,
            inspire_url=inspire_url,
            website=website
        )

    def _parse_institution_search_results(self, data: Dict[str, Any]) -> List[InstitutionInfo]:
        """Parse institution search response."""
        hits = data.get("hits", {})
        return [self._parse_institution(hit) for hit in hits.get("hits", [])]

    def _parse_conference(self, data: Dict[str, Any]) -> ConferenceInfo:
        """Parse single conference response."""
        metadata = data.get("metadata", data)
        inspire_id = str(metadata.get("control_number", data.get("id", "")))

        # Extract titles
        titles = metadata.get("titles", [])
        name = titles[0].get("title", "") if titles else ""

        # Extract acronym
        acronyms = metadata.get("acronyms", [])
        short_name = acronyms[0] if acronyms else None

        # Extract series
        series = metadata.get("series", [])
        series_name = series[0].get("name") if series else None

        # Extract dates
        opening_date = metadata.get("opening_date")
        closing_date = metadata.get("closing_date")

        # Extract location
        addresses = metadata.get("addresses", [])
        city = None
        country = None
        if addresses:
            addr = addresses[0]
            city = addr.get("cities", [None])[0] if addr.get("cities") else None
            country = addr.get("country")

        # Extract URLs
        urls = metadata.get("urls", [])
        website = urls[0].get("value") if urls else None

        cnum = metadata.get("cnum")

        inspire_url = f"https://inspirehep.net/conferences/{inspire_id}"

        return ConferenceInfo(
            inspire_id=inspire_id,
            name=name,
            short_name=short_name,
            series=series_name,
            opening_date=opening_date,
            closing_date=closing_date,
            city=city,
            country=country,
            inspire_url=inspire_url,
            website=website,
            cnum=cnum
        )

    def _parse_conference_search_results(self, data: Dict[str, Any]) -> List[ConferenceInfo]:
        """Parse conference search response."""
        hits = data.get("hits", {})
        return [self._parse_conference(hit) for hit in hits.get("hits", [])]

    def _parse_journal(self, data: Dict[str, Any]) -> JournalInfo:
        """Parse single journal response."""
        metadata = data.get("metadata", data)
        inspire_id = str(metadata.get("control_number", data.get("id", "")))

        # Extract titles
        journal_title = metadata.get("journal_title", {})
        name = journal_title.get("title", "")

        short_name = metadata.get("short_title")
        publisher = metadata.get("publisher", [None])[0] if metadata.get("publisher") else None

        # Extract identifiers
        issns = metadata.get("issns", [])
        issn = issns[0].get("value") if issns else None
        coden = metadata.get("coden")

        # Extract URLs
        urls = metadata.get("urls", [])
        website = urls[0].get("value") if urls else None

        inspire_url = f"https://inspirehep.net/journals/{inspire_id}"

        return JournalInfo(
            inspire_id=inspire_id,
            name=name,
            short_name=short_name,
            publisher=publisher,
            issn=issn,
            coden=coden,
            inspire_url=inspire_url,
            website=website
        )

    def _parse_journal_search_results(self, data: Dict[str, Any]) -> List[JournalInfo]:
        """Parse journal search response."""
        hits = data.get("hits", {})
        return [self._parse_journal(hit) for hit in hits.get("hits", [])]

    def _parse_experiment(self, data: Dict[str, Any]) -> ExperimentInfo:
        """Parse single experiment response."""
        metadata = data.get("metadata", data)
        inspire_id = str(metadata.get("control_number", data.get("id", "")))

        # Extract name
        legacy_name = metadata.get("legacy_name", "")
        long_name = metadata.get("long_name")

        # Extract collaboration
        collaboration = metadata.get("collaboration", {}).get("value")

        # Extract institutions
        institutions = metadata.get("institutions", [])
        inst_list = [i.get("value", "") for i in institutions]

        # Extract status and dates
        project_type = metadata.get("project_type", [])
        status = project_type[0] if project_type else None
        date_started = metadata.get("date_started")

        # Extract description
        description = metadata.get("description")

        # Extract accelerator
        accelerator = metadata.get("accelerator", {}).get("value")

        # Extract URLs
        urls = metadata.get("urls", [])
        website = urls[0].get("value") if urls else None

        inspire_url = f"https://inspirehep.net/experiments/{inspire_id}"

        return ExperimentInfo(
            inspire_id=inspire_id,
            name=legacy_name,
            long_name=long_name,
            collaboration=collaboration,
            institutions=inst_list,
            status=status,
            description=description,
            inspire_url=inspire_url,
            website=website,
            start_date=date_started,
            accelerator=accelerator
        )

    def _parse_experiment_search_results(self, data: Dict[str, Any]) -> List[ExperimentInfo]:
        """Parse experiment search response."""
        hits = data.get("hits", {})
        return [self._parse_experiment(hit) for hit in hits.get("hits", [])]


# Singleton interface for convenience functions
_interface: Optional[InspireInterface] = None


def get_interface() -> InspireInterface:
    """Get or create singleton INSPIRE interface."""
    global _interface
    if _interface is None:
        _interface = InspireInterface()
    return _interface
