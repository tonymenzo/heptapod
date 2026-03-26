"""
# query_builder.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Natural language to SPIRES query converter.

Transforms human-readable queries into INSPIRE's SPIRES-compatible
search syntax for efficient literature searches.
"""

import re
from typing import List, Tuple, Optional


class QueryBuilder:
    """
    Convert natural language queries to SPIRES-compatible syntax.

    INSPIRE supports SPIRES search syntax for literature:
      - a/author: author search
      - t/title: title search
      - j/journal: journal search
      - k/keyword: keyword search
      - topcite N+: highly cited papers (N+ citations)
      - citedby:recid:X: papers citing record X
      - refersto:recid:X: papers referenced by record X
      - date YYYY or date YYYY->YYYY: date range

    Examples:
        >>> qb = QueryBuilder()
        >>> qb.build("papers by Witten on string theory")
        "a witten and t string theory"
        >>> qb.build("highly cited papers on supersymmetry")
        "t supersymmetry and topcite 100+"
        >>> qb.build("a witten")  # Pass-through
        "a witten"
    """

    # Prefixes that indicate already-formatted SPIRES queries
    SPIRES_PREFIXES = (
        "a ", "author:", "t ", "title:", "j ", "journal:",
        "k ", "keyword:", "citedby:", "refersto:", "topcite",
        "find ", "fin ", "date ", "date:", "exactauthor:",
        "eprint:", "arxiv:", "doi:", "recid:", "collaboration:"
    )

    # Common physicist name patterns for better author detection
    # Note: These patterns must NOT use re.IGNORECASE to properly match capitalized names
    PHYSICIST_PATTERNS = [
        r"(?i:by|author[s]?:?)\s+([A-Z][a-z]+(?:[-'][A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?)?)",
    ]

    def __init__(self):
        """Initialize query builder."""
        # Do NOT use re.IGNORECASE globally - the pattern uses inline (?i:...) for specific parts
        self._compiled_patterns = [
            re.compile(p) for p in self.PHYSICIST_PATTERNS
        ]

    def build(self, natural_query: str) -> str:
        """
        Convert natural language query to SPIRES format.

        Args:
            natural_query: Natural language query

        Returns:
            SPIRES-compatible query string
        """
        query = natural_query.strip()

        # Direct SPIRES syntax detection (already formatted)
        query_lower = query.lower()
        if any(query_lower.startswith(prefix) for prefix in self.SPIRES_PREFIXES):
            return query

        parts = []
        remaining = query

        # Check for author patterns
        author_match = self._extract_author(remaining)
        if author_match:
            author_name, remaining = author_match
            parts.append(f"a {author_name.lower()}")

        # Check for topic/title patterns
        topic_match = self._extract_topic(remaining)
        if topic_match:
            topic, remaining = topic_match
            parts.append(f"t {topic}")

        # Check for highly cited
        if self._has_citation_requirement(remaining):
            citation_threshold = self._extract_citation_threshold(remaining)
            parts.append(f"topcite {citation_threshold}+")
            remaining = self._remove_citation_phrase(remaining)

        # Check for date constraints
        date_constraint = self._extract_date(remaining)
        if date_constraint:
            parts.append(date_constraint)

        # Check for journal
        journal_match = self._extract_journal(remaining)
        if journal_match:
            parts.append(f"j {journal_match}")

        # Combine parts
        if parts:
            return " and ".join(parts)

        # Fallback: treat as title search if it looks like keywords
        # Otherwise return as-is for general search
        if self._looks_like_keywords(query):
            return f"t {query}"

        return query

    def _extract_author(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract author name from text."""
        for pattern in self._compiled_patterns:
            match = pattern.search(text)
            if match:
                author = match.group(1)
                remaining = text[:match.start()] + text[match.end():]
                return author.strip(), remaining.strip()

        # Also check for "X's papers" pattern
        possessive = re.search(
            r"([A-Z][a-z]+(?:[-'][A-Z][a-z]+)?(?:\s+[A-Z][a-z]+)?)'s\s+papers?",
            text, re.IGNORECASE
        )
        if possessive:
            author = possessive.group(1)
            remaining = text[:possessive.start()] + text[possessive.end():]
            return author.strip(), remaining.strip()

        return None

    def _extract_topic(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract topic/title from text."""
        # "on/about X" pattern
        match = re.search(
            r"(?:on|about|regarding|related\s+to)\s+(.+?)(?:\s+(?:since|from|after|before|in)\s+\d{4}|\s*$)",
            text, re.IGNORECASE
        )
        if match:
            topic = match.group(1).strip()
            # Clean up trailing words
            topic = re.sub(r"\s+(and|or|with)\s*$", "", topic, flags=re.IGNORECASE)
            remaining = text[:match.start()] + text[match.end():]
            return topic, remaining.strip()

        # "X papers" where X is a physics topic
        topic_match = re.search(
            r"^(.+?)\s+papers?\b",
            text, re.IGNORECASE
        )
        if topic_match:
            topic = topic_match.group(1).strip()
            # Filter out common non-topic words
            if topic.lower() not in ("recent", "new", "highly cited", "cited", "top"):
                remaining = text[topic_match.end():]
                return topic, remaining.strip()

        return None

    def _has_citation_requirement(self, text: str) -> bool:
        """Check if text mentions citation requirements."""
        patterns = [
            r"highly\s+cited",
            r"top\s+cited",
            r"most\s+cited",
            r"influential",
            r"famous",
            r"seminal",
            r"landmark",
            r"(\d+)\+?\s+citations?",
            r"cited\s+(\d+)\+?\s+times?",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _extract_citation_threshold(self, text: str) -> int:
        """Extract citation threshold from text, default to 100."""
        # Look for explicit numbers
        match = re.search(r"(\d+)\+?\s+citations?", text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        match = re.search(r"cited\s+(\d+)\+?\s+times?", text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Default thresholds based on keywords
        if re.search(r"very\s+highly|extremely|most", text, re.IGNORECASE):
            return 500
        elif re.search(r"highly", text, re.IGNORECASE):
            return 100
        elif re.search(r"influential|famous|seminal|landmark", text, re.IGNORECASE):
            return 250

        return 100

    def _remove_citation_phrase(self, text: str) -> str:
        """Remove citation-related phrases from text."""
        patterns = [
            r"highly\s+cited\s*",
            r"top\s+cited\s*",
            r"most\s+cited\s*",
            r"influential\s*",
            r"famous\s*",
            r"seminal\s*",
            r"landmark\s*",
            r"(\d+)\+?\s+citations?\s*",
            r"cited\s+(\d+)\+?\s+times?\s*",
        ]
        for pattern in patterns:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        return " ".join(text.split())

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date constraint from text."""
        # "since/from/after YYYY"
        match = re.search(r"(?:since|from|after)\s+(\d{4})", text, re.IGNORECASE)
        if match:
            return f"date {match.group(1)}->"

        # "before YYYY"
        match = re.search(r"before\s+(\d{4})", text, re.IGNORECASE)
        if match:
            return f"date ->{match.group(1)}"

        # "in YYYY" or just "YYYY papers"
        match = re.search(r"(?:in|year)\s+(\d{4})\b", text, re.IGNORECASE)
        if match:
            return f"date {match.group(1)}"

        # "between YYYY and YYYY"
        match = re.search(r"between\s+(\d{4})\s+and\s+(\d{4})", text, re.IGNORECASE)
        if match:
            return f"date {match.group(1)}->{match.group(2)}"

        return None

    def _extract_journal(self, text: str) -> Optional[str]:
        """Extract journal name from text."""
        match = re.search(
            r"(?:in|published\s+in|from)\s+(?:the\s+)?([A-Z][a-zA-Z.\s]+(?:Review|Letters|Journal|Physics))",
            text, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        return None

    def _looks_like_keywords(self, text: str) -> bool:
        """Check if text looks like search keywords rather than a question."""
        # Short text without question marks is probably keywords
        if len(text) < 50 and "?" not in text:
            return True
        # Physics-y terms
        physics_terms = [
            "supersymmetry", "susy", "qcd", "higgs", "dark matter",
            "neutrino", "string theory", "black hole", "cosmology",
            "collider", "lhc", "gauge", "symmetry", "boson", "fermion",
            "quark", "lepton", "hadron", "gravitational", "inflation"
        ]
        text_lower = text.lower()
        return any(term in text_lower for term in physics_terms)

    def build_citation_query(self, recid: str) -> str:
        """
        Build query for papers citing a given paper.

        Args:
            recid: INSPIRE record ID

        Returns:
            SPIRES query for citing papers
        """
        return f"citedby:recid:{recid}"

    def build_reference_query(self, recid: str) -> str:
        """
        Build query for papers referenced by a given paper.

        Args:
            recid: INSPIRE record ID

        Returns:
            SPIRES query for referenced papers
        """
        return f"refersto:recid:{recid}"

    def build_author_papers_query(self, author: str) -> str:
        """
        Build query for papers by an author.

        Args:
            author: Author name or BAI

        Returns:
            SPIRES query for author's papers
        """
        return f"a {author}"

    def build_collaboration_query(self, collaboration: str) -> str:
        """
        Build query for papers by a collaboration.

        Args:
            collaboration: Collaboration name (e.g., "CMS", "ATLAS")

        Returns:
            SPIRES query for collaboration papers
        """
        return f"collaboration:{collaboration}"
