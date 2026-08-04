"""
# arxiv_interface.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

arXiv Atom API interface: search, PDF download, and PyMuPDF text extraction.

Complements the INSPIRE tools (which have no arXiv access, PDF retrieval, or
full-text extraction) so an agent can go from a paper hit to the full text
that downstream Lagrangian extraction needs. Uses ``requests`` (HEPTAPOD base
dependency); PyMuPDF is imported lazily so search/download work without it.
"""

import hashlib
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse

import requests

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# arXiv API usage guidance: no more than one request roughly every 3 seconds.
_MIN_REQUEST_INTERVAL = 3.0

# Cap on downloaded PDF size (defensive; arXiv PDFs are rarely > a few tens of MB).
_MAX_PDF_BYTES = 100 * 1024 * 1024

# Only arXiv hosts, over HTTPS, may be fetched (avoids SSRF via arbitrary URLs).
_ALLOWED_PDF_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}

# arXiv identifier forms (version suffix optional):
#   new style: 2103.02708, 2103.02708v2   (YYMM.NNNNN)
#   old style: hep-ph/9905221, math.AG/0309136v1
_ARXIV_NEW_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ARXIV_OLD_RE = re.compile(r"^[a-z][a-z-]*(\.[A-Za-z]{2})?/\d{7}(v\d+)?$")


def _strip_arxiv_prefix(arxiv_id: str) -> str:
    """Drop an optional case-insensitive ``arXiv:`` prefix and surrounding space."""
    return re.sub(r"(?i)^arxiv:", "", (arxiv_id or "").strip())


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Canonical id: strip an ``arXiv:`` prefix and any ``vN`` version suffix.

    e.g. ``arXiv:2101.00001v2`` -> ``2101.00001``. Used for dedup/display, not
    for fetching (fetching preserves an explicit version).
    """
    return re.sub(r"v\d+$", "", _strip_arxiv_prefix(arxiv_id))


def validate_arxiv_id(arxiv_id: str) -> Optional[str]:
    """Return the well-formed id (``arXiv:`` prefix stripped, version preserved),
    or ``None`` if ``arxiv_id`` is not a recognizable arXiv identifier."""
    if not arxiv_id:
        return None
    s = _strip_arxiv_prefix(arxiv_id)
    if _ARXIV_NEW_RE.match(s) or _ARXIV_OLD_RE.match(s):
        return s
    return None


def is_allowed_arxiv_url(url: str) -> bool:
    """True only for HTTPS URLs on a known arXiv host."""
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
        return False
    return parsed.scheme == "https" and (parsed.hostname or "") in _ALLOWED_PDF_HOSTS


# Back-compat alias (original, PDF-specific name).
is_allowed_pdf_url = is_allowed_arxiv_url


def https_arxiv_url(url: Optional[str]) -> Optional[str]:
    """Upgrade an ``http://`` arXiv URL to ``https://`` (arXiv feeds emit http).

    Leaves non-arXiv or already-https URLs unchanged.
    """
    if not url or not url.startswith("http://"):
        return url
    host = (urlparse(url).hostname or "")
    if host in _ALLOWED_PDF_HOSTS or host.endswith(".arxiv.org") or host == "arxiv.org":
        return "https://" + url[len("http://"):]
    return url


class _MinIntervalLimiter:
    """Serialize requests so consecutive calls are >= ``interval`` seconds apart."""

    def __init__(self, interval: float = _MIN_REQUEST_INTERVAL) -> None:
        self.interval = interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> float:
        """Block until at least ``interval`` seconds have passed since the last call."""
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            waited = 0.0
            if self._last and delta < self.interval:
                waited = self.interval - delta
                time.sleep(waited)
            self._last = time.monotonic()
            return waited


# Module-level limiter shared across tool invocations in a process so repeated
# arXiv calls from an agent stay polite even though each tool builds its own
# interface instance.
_LIMITER = _MinIntervalLimiter()


class ArxivInterface:
    """Thin client over the arXiv Atom export API plus PDF retrieval/extraction."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "HEPTAPOD/1.0 (Physics research tool)"}
        )

    # ----------------------------- Search ----------------------------- #

    @staticmethod
    def build_query(
        model_name: str,
        keywords: Optional[List[str]] = None,
        category: Optional[str] = "hep-ph",
    ) -> str:
        """Build an arXiv ``search_query`` from a model name and optional keywords.

        Example: ``cat:hep-ph AND (ti:"scalar leptoquark" OR abs:"scalar leptoquark")
        AND (abs:"BSM" OR abs:"leptoquark")``.
        """
        clauses: List[str] = []
        if category:
            clauses.append(f"cat:{category}")
        clauses.append(f'(ti:"{model_name}" OR abs:"{model_name}")')
        if keywords:
            kw = " OR ".join(f'abs:"{k.strip()}"' for k in keywords if k and k.strip())
            if kw:
                clauses.append(f"({kw})")
        return " AND ".join(clauses)

    def search(
        self,
        query: str,
        max_results: int = 25,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Search arXiv. Returns ``(papers, request_url)``."""
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        url = f"{ARXIV_API_URL}?{urlencode(params)}"
        _LIMITER.wait()
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return self._parse_feed(resp.text), url

    def _parse_feed(self, xml_text: str) -> List[Dict[str, Any]]:
        root = ET.fromstring(xml_text)
        papers: List[Dict[str, Any]] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            parsed = self._parse_entry(entry)
            if parsed is not None:
                papers.append(parsed)
        return papers

    def _parse_entry(self, entry: ET.Element) -> Optional[Dict[str, Any]]:
        title_el = entry.find("atom:title", ATOM_NS)
        if title_el is None or not title_el.text:
            return None
        title = " ".join(title_el.text.split())

        summary_el = entry.find("atom:summary", ATOM_NS)
        abstract = (
            " ".join(summary_el.text.split())
            if summary_el is not None and summary_el.text
            else None
        )

        authors = [
            name_el.text.strip()
            for author_el in entry.findall("atom:author", ATOM_NS)
            if (name_el := author_el.find("atom:name", ATOM_NS)) is not None
            and name_el.text
        ]

        arxiv_id = self._extract_arxiv_id(entry)
        if not arxiv_id:
            return None
        arxiv_id = normalize_arxiv_id(arxiv_id) or arxiv_id

        categories: List[str] = []
        primary = entry.find("arxiv:primary_category", ATOM_NS)
        if primary is not None and primary.get("term"):
            categories.append(primary.get("term", ""))
        for cat in entry.findall("atom:category", ATOM_NS):
            term = cat.get("term")
            if term and term not in categories:
                categories.append(term)

        doi_el = entry.find("arxiv:doi", ATOM_NS)
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

        pdf_url: Optional[str] = None
        abs_url: Optional[str] = None
        for link in entry.findall("atom:link", ATOM_NS):
            if link.get("rel") == "alternate":
                abs_url = link.get("href")
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href")

        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "categories": categories,
            "doi": doi,
            "published": self._parse_date(entry.find("atom:published", ATOM_NS)),
            "updated": self._parse_date(entry.find("atom:updated", ATOM_NS)),
            "pdf_url": https_arxiv_url(pdf_url) or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "abs_url": https_arxiv_url(abs_url) or f"https://arxiv.org/abs/{arxiv_id}",
        }

    @staticmethod
    def _parse_date(element: Optional[ET.Element]) -> Optional[str]:
        if element is None or not element.text:
            return None
        return element.text[:10]

    @staticmethod
    def _extract_arxiv_id(entry: ET.Element) -> Optional[str]:
        id_el = entry.find("atom:id", ATOM_NS)
        if id_el is None or not id_el.text:
            return None
        match = re.search(r"arxiv\.org/abs/(.+)$", id_el.text.strip())
        return match.group(1) if match else None

    # -------------------------- PDF retrieval ------------------------- #

    def download_pdf(
        self,
        dest_path: str,
        arxiv_id: Optional[str] = None,
        pdf_url: Optional[str] = None,
        max_bytes: int = _MAX_PDF_BYTES,
    ) -> Dict[str, Any]:
        """Download a PDF to ``dest_path``, streaming with a size cap.

        Provide either ``arxiv_id`` (the canonical arXiv PDF URL is derived,
        preserving an explicit ``vN`` version) or an explicit ``pdf_url`` (which
        must be an HTTPS arXiv URL). Validates the ``%PDF-`` header, writes to a
        temp file, and atomically renames on success. Returns ``{"bytes",
        "sha256"}``.
        """
        if not pdf_url:
            if not arxiv_id:
                raise ValueError("Either arxiv_id or pdf_url is required")
            valid = validate_arxiv_id(arxiv_id)
            if not valid:
                raise ValueError(f"Invalid arXiv id: {arxiv_id!r}")
            pdf_url = f"https://arxiv.org/pdf/{valid}.pdf"
        if not is_allowed_pdf_url(pdf_url):
            raise ValueError(f"Refusing non-arXiv or non-HTTPS PDF URL: {pdf_url!r}")

        _LIMITER.wait()
        tmp = dest_path + ".part"
        digest = hashlib.sha256()
        total = 0
        header = b""
        validated = False
        with self._session.get(pdf_url, timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            try:
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if not chunk:
                            continue
                        if not validated:
                            # Buffer until we have >= 5 bytes: iter_content does
                            # not guarantee the first chunk contains the full
                            # "%PDF-" magic.
                            header += chunk
                            if len(header) < 5:
                                continue
                            if header[:5] != b"%PDF-":
                                raise ValueError("Downloaded file is not a valid PDF")
                            validated = True
                            chunk = header  # flush the buffered header bytes
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError(
                                f"PDF exceeds max size of {max_bytes} bytes"
                            )
                        digest.update(chunk)
                        fh.write(chunk)
                if not validated:
                    raise ValueError("Downloaded file is not a valid PDF")
            except BaseException:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise

        try:
            os.replace(tmp, dest_path)
        except OSError:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        return {"bytes": total, "sha256": digest.hexdigest()}

    # ------------------------ E-print retrieval ----------------------- #

    def download_eprint(
        self,
        arxiv_id: str,
        max_bytes: int = 50 * 1024 * 1024,
    ) -> Dict[str, Any]:
        """Download the arXiv e-print source payload for ``arxiv_id``.

        Fetches ``https://export.arxiv.org/e-print/<id>`` (gzipped tar of LaTeX
        sources, gzipped single ``.tex``, or a bare PDF when no source exists),
        streaming with a size cap. Returns ``{"data": bytes, "sha256": str,
        "bytes": int}``; the caller classifies via source_archive.detect_payload.
        """
        valid = validate_arxiv_id(arxiv_id)
        if not valid:
            raise ValueError(f"Invalid arXiv id: {arxiv_id!r}")
        url = f"https://export.arxiv.org/e-print/{valid}"
        if not is_allowed_arxiv_url(url):  # defensive; host is ours by construction
            raise ValueError(f"Refusing non-arXiv URL: {url!r}")

        _LIMITER.wait()
        digest = hashlib.sha256()
        buf = bytearray()
        with self._session.get(url, timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    raise ValueError(f"e-print exceeds max size of {max_bytes} bytes")
                digest.update(chunk)
        if not buf:
            raise ValueError("e-print payload is empty")
        return {"data": bytes(buf), "sha256": digest.hexdigest(), "bytes": len(buf)}

    # ------------------------- Text extraction ------------------------ #

    @staticmethod
    def extract_text(pdf_path: str) -> Tuple[str, int]:
        """Extract plain text from a PDF with PyMuPDF. Returns ``(text, n_pages)``."""
        import pymupdf  # lazy: only extraction needs PyMuPDF

        doc = pymupdf.open(pdf_path)
        try:
            pages = [page.get_text("text") for page in doc]
        finally:
            doc.close()
        return "\n\n".join(pages), len(pages)
