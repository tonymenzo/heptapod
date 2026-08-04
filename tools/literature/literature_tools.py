"""
# literature_tools.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

arXiv literature tools for agent use.

Provides BaseTool implementations for searching arXiv, downloading paper PDFs,
and extracting their full text. These complement the INSPIRE tools, which cover
metadata/citation search but neither arXiv nor full-text retrieval — the text an
agent needs to identify and extract a BSM Lagrangian.
"""

import hashlib
import json
import os
from typing import List, Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .arxiv_interface import (
    ArxivInterface,
    is_allowed_pdf_url,
    validate_arxiv_id,
)
from .source_archive import (
    detect_payload,
    gunzip_capped,
    inline_inputs,
    resolve_main_tex,
    safe_extract_tar,
    strip_comments,
)

SCHEMA_VERSION = "literature-1.1"

_SEARCH_SORT_BY = {"relevance", "submittedDate", "lastUpdatedDate"}
_SEARCH_SORT_ORDER = {"ascending", "descending"}


def _safe_join(base_directory: str, rel_or_abs: str) -> Optional[str]:
    """Resolve ``rel_or_abs`` under ``base_directory``; return None if it escapes.

    Uses ``realpath`` on both sides so a symlinked component cannot redirect a
    write/read outside the sandbox (see CONTRIBUTING.md "Path Safety and
    Sandboxing").
    """
    if not rel_or_abs:
        return None
    base = os.path.realpath(base_directory)
    full = os.path.realpath(os.path.join(base, rel_or_abs))
    if full != base and not full.startswith(base + os.sep):
        return None
    return full


def _pdf_stem(arxiv_id: Optional[str], pdf_url: Optional[str]) -> Optional[str]:
    """Deterministic, collision-free filename stem for a PDF.

    Derived from a validated arXiv id (version preserved, path separators
    flattened) or, when only a URL is given, from a hash of the URL — so two
    different sources never map to the same cached file. Returns None if an
    ``arxiv_id`` is supplied but malformed.
    """
    if arxiv_id:
        valid = validate_arxiv_id(arxiv_id)
        if not valid:
            return None
        return valid.replace("/", "_").replace("\\", "_")
    if pdf_url:
        return "url_" + hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()[:16]
    return None


def _sha256_file(path: str) -> str:
    """Chunked SHA-256 of a file (avoids reading large PDFs fully into memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ============================ Search ============================ #


class ArxivSearchTool(BaseTool):
    """
    Search arXiv for physics papers via the Atom export API.

    Use this to discover candidate papers for a BSM scenario when you need the
    arXiv preprint (and, via the other literature tools, its full text). For
    citation-ranked metadata search, prefer InspireSearchTool; use this when you
    need arXiv ids / PDF links or an abstract-level keyword search.

    Input:
        query: A raw arXiv search_query (e.g. 'cat:hep-ph AND ti:"leptoquark"').
               If omitted, one is built from model_name/keywords/category.
        model_name: BSM scenario or model name (used to build a query when
                    `query` is not given), e.g. "scalar leptoquark".
        keywords: Extra abstract keywords to AND into a built query.
        category: arXiv category filter for a built query (default "hep-ph").
        max_results: Max papers to return (default 25, max 100).
        sort_by: "relevance", "submittedDate", or "lastUpdatedDate".
        sort_order: "descending" or "ascending".

    Returns JSON:
        {"status": "ok", "schema": "literature-1.0", "query": "...",
         "count": N, "papers": [{"arxiv_id","title","authors","abstract",
         "categories","doi","published","updated","pdf_url","abs_url"}, ...]}
    """

    # ======================== Runtime fields ======================== #
    query: Optional[str] = RuntimeField(
        default=None,
        description="Raw arXiv search_query; if omitted it is built from model_name/keywords/category",
    )
    model_name: Optional[str] = RuntimeField(
        default=None,
        description="BSM model/scenario name used to build a query when `query` is not provided",
    )
    keywords: Optional[List[str]] = RuntimeField(
        default=None,
        description="Extra abstract keywords to AND into a built query",
    )
    category: Optional[str] = RuntimeField(
        default="hep-ph",
        description="arXiv category filter for a built query (default hep-ph)",
    )
    max_results: Optional[int] = RuntimeField(
        default=25, description="Max papers to return (default 25, max 100)"
    )
    sort_by: Optional[str] = RuntimeField(
        default="relevance",
        description="Sort field: 'relevance', 'submittedDate', or 'lastUpdatedDate'",
    )
    sort_order: Optional[str] = RuntimeField(
        default="descending", description="'descending' or 'ascending'"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations",
        default=".",
    )
    # ================================================================ #

    def _run(self) -> str:
        query = self.query
        if not query:
            if not self.model_name:
                return self.format_error(
                    error="Missing Parameter",
                    reason="Provide either `query` or `model_name`",
                    suggestion="Pass a raw arXiv query, or a model_name to build one",
                )
            query = ArxivInterface.build_query(
                self.model_name, self.keywords, self.category
            )

        sort_by = self.sort_by or "relevance"
        sort_order = self.sort_order or "descending"
        if sort_by not in _SEARCH_SORT_BY:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"sort_by must be one of {sorted(_SEARCH_SORT_BY)}",
                context=f"sort_by={self.sort_by}",
            )
        if sort_order not in _SEARCH_SORT_ORDER:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"sort_order must be one of {sorted(_SEARCH_SORT_ORDER)}",
                context=f"sort_order={self.sort_order}",
            )
        max_results = max(1, min(self.max_results or 25, 100))

        try:
            interface = ArxivInterface()
            papers, url = interface.search(
                query,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        except Exception as e:  # noqa: BLE001 - surface network/parse errors to the agent
            return self.format_error(
                error="arXiv Search Failed",
                reason=str(e),
                context=f"query={query}",
                suggestion="Check the query syntax or retry (arXiv may be rate-limiting)",
            )

        return json.dumps(
            {
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "query": query,
                "request_url": url,
                "count": len(papers),
                "papers": papers,
            },
            indent=2,
        )


# ========================= PDF retrieval ======================== #


class FetchPaperPDFTool(BaseTool):
    """
    Download an arXiv paper PDF into the sandbox for later text extraction.

    Input:
        arxiv_id: arXiv identifier, e.g. "2103.02708" (version suffix optional;
                  an explicit version is preserved, else the latest is fetched).
        pdf_url: Explicit HTTPS arXiv PDF URL, used only when arxiv_id is not
                 given (must be on an arXiv host). If both are given, arxiv_id
                 wins.
        output_dir: Directory (relative to base_directory) for the PDF
                    (default "pdfs").

    Behavior:
        Downloads to {output_dir}/{arxiv_id}.pdf under base_directory, reusing a
        cached valid PDF if present. Validates the %PDF- header.

    Returns JSON:
        {"status": "ok", "schema": "literature-1.0", "arxiv_id": "...",
         "pdf_path": "pdfs/2103.02708.pdf", "bytes": N, "sha256": "...",
         "cached": false}
    """

    # ======================== Runtime fields ======================== #
    arxiv_id: Optional[str] = RuntimeField(
        default=None, description="arXiv identifier, e.g. '2103.02708'"
    )
    pdf_url: Optional[str] = RuntimeField(
        default=None,
        description="Explicit HTTPS arXiv PDF URL; used only when arxiv_id is not given",
    )
    output_dir: Optional[str] = RuntimeField(
        default="pdfs", description="Directory (relative to base_directory) for the PDF"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    # ================================================================ #

    def _run(self) -> str:
        if not self.arxiv_id and not self.pdf_url:
            return self.format_error(
                error="Missing Parameter",
                reason="Provide either `arxiv_id` or `pdf_url`",
            )
        if self.arxiv_id and validate_arxiv_id(self.arxiv_id) is None:
            return self.format_error(
                error="Invalid Parameter",
                reason="arxiv_id is not a well-formed arXiv identifier",
                context=f"arxiv_id={self.arxiv_id}",
                suggestion="Use e.g. '2103.02708' or 'hep-ph/9905221' (version optional)",
            )
        if self.pdf_url and not is_allowed_pdf_url(self.pdf_url):
            return self.format_error(
                error="Access Denied",
                reason="pdf_url must be an HTTPS URL on an arXiv host",
                context=f"pdf_url={self.pdf_url}",
                suggestion="Pass an arxiv_id, or an https://arxiv.org/... PDF URL",
            )

        out_dir = _safe_join(self.base_directory, self.output_dir or "pdfs")
        if out_dir is None:
            return self.format_error(
                error="Access Denied",
                reason="output_dir escapes base_directory",
                context=self.output_dir,
                suggestion="Use a relative path inside base_directory",
            )

        stem = _pdf_stem(self.arxiv_id, self.pdf_url)
        if stem is None:
            return self.format_error(
                error="Invalid Parameter",
                reason="Could not derive a filename from arxiv_id/pdf_url",
            )

        try:
            os.makedirs(out_dir, exist_ok=True)
            dest = os.path.join(out_dir, f"{stem}.pdf")
            rel_dest = os.path.relpath(dest, os.path.realpath(self.base_directory))

            # Reuse a cached, valid PDF for this exact source. Never follow a
            # symlink at the leaf (a download would os.replace it safely, but a
            # cache read must not escape the sandbox).
            if (
                os.path.exists(dest)
                and not os.path.islink(dest)
                and os.path.getsize(dest) > 0
            ):
                with open(dest, "rb") as fh:
                    is_pdf = fh.read(5) == b"%PDF-"
                if is_pdf:
                    return json.dumps(
                        {
                            "status": "ok",
                            "schema": SCHEMA_VERSION,
                            "arxiv_id": self.arxiv_id,
                            "pdf_path": rel_dest,
                            "bytes": os.path.getsize(dest),
                            "sha256": _sha256_file(dest),
                            "cached": True,
                        },
                        indent=2,
                    )
        except OSError as e:
            return self.format_error(
                error="Filesystem Error",
                reason=str(e),
                context=f"output_dir={self.output_dir}",
            )

        try:
            interface = ArxivInterface()
            # arxiv_id takes precedence when both are given, so the downloaded
            # source matches the cache key derived by _pdf_stem.
            meta = interface.download_pdf(
                dest,
                arxiv_id=self.arxiv_id,
                pdf_url=None if self.arxiv_id else self.pdf_url,
            )
        except Exception as e:  # noqa: BLE001
            return self.format_error(
                error="PDF Download Failed",
                reason=str(e),
                context=f"arxiv_id={self.arxiv_id} pdf_url={self.pdf_url}",
                suggestion="Verify the arXiv id/URL; arXiv may be rate-limiting",
            )

        return json.dumps(
            {
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "arxiv_id": self.arxiv_id,
                "pdf_path": rel_dest,
                "bytes": meta["bytes"],
                "sha256": meta["sha256"],
                "cached": False,
            },
            indent=2,
        )


# ======================= LaTeX source (e-print) ================= #


class ArxivSourceTool(BaseTool):
    """
    Fetch a paper's LaTeX SOURCE from arXiv (e-print) — the preferred input for
    Lagrangian extraction, since PDF text mangles equations while the .tex
    source preserves them exactly.

    Input:
        arxiv_id: arXiv identifier, e.g. "2103.02708" (version suffix optional).
        output_dir: Directory (relative to base_directory) for extracted source
                    files (default "source").
        inline_one_level: Inline \\input{...}/\\include{...} one level into the
                          main file (default true).
        preview_chars: Length of the inline LaTeX preview returned (default 2000).

    Behavior:
        Downloads https://export.arxiv.org/e-print/<id> (size-capped), detects
        the payload (gzipped tar of sources / gzipped single .tex / bare PDF),
        extracts archives with strict safety checks, resolves the main .tex,
        strips % comments, optionally inlines one level of \\input, and writes
        the normalized LaTeX to text/<id>_source.tex.

    Returns JSON:
        {"status": "ok", "schema": "literature-1.1", "arxiv_id": "...",
         "source_type": "tar"|"single_tex"|"pdf_only",
         "source_dir": "source/<id>/"?, "main_tex": "source/<id>/ms.tex"?,
         "tex_path": "text/<id>_source.tex"?, "n_files": N, "chars": M,
         "preview": "...", "cached": bool}

        source_type "pdf_only" means arXiv has no LaTeX source for this paper;
        the result includes a suggestion to use FetchPaperPDFTool +
        ExtractPaperTextTool instead (this is a normal outcome, not an error).
    """

    # ======================== Runtime fields ======================== #
    arxiv_id: str = RuntimeField(
        description="arXiv identifier, e.g. '2103.02708' (version suffix optional)"
    )
    output_dir: Optional[str] = RuntimeField(
        default="source",
        description="Directory (relative to base_directory) for extracted source files",
    )
    inline_one_level: Optional[bool] = RuntimeField(
        default=True,
        description="Inline \\input/\\include one level into the main file (default true)",
    )
    preview_chars: Optional[int] = RuntimeField(
        default=2000, description="Length of the inline LaTeX preview (default 2000)"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    # ================================================================ #

    def _run(self) -> str:
        valid = validate_arxiv_id(self.arxiv_id)
        if not valid:
            return self.format_error(
                error="Invalid Parameter",
                reason="arxiv_id is not a well-formed arXiv identifier",
                context=f"arxiv_id={self.arxiv_id}",
                suggestion="Use e.g. '2103.02708' or 'hep-ph/9905221' (version optional)",
            )
        stem = _pdf_stem(self.arxiv_id, None)

        base = os.path.realpath(self.base_directory)
        tex_rel = os.path.join("text", f"{stem}_source.tex")
        tex_abs = os.path.join(base, tex_rel)

        # Cache: normalized LaTeX already produced for this id.
        if os.path.isfile(tex_abs) and os.path.getsize(tex_abs) > 0:
            try:
                with open(tex_abs, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError as e:
                return self.format_error(error="Filesystem Error", reason=str(e))
            n = self.preview_chars if self.preview_chars is not None else 2000
            return json.dumps(
                {
                    "status": "ok",
                    "schema": SCHEMA_VERSION,
                    "arxiv_id": self.arxiv_id,
                    "source_type": "cached",
                    "tex_path": tex_rel,
                    "chars": len(text),
                    "preview": text[:n],
                    "cached": True,
                },
                indent=2,
            )

        out_dir = _safe_join(self.base_directory, self.output_dir or "source")
        if out_dir is None:
            return self.format_error(
                error="Access Denied",
                reason="output_dir escapes base_directory",
                context=self.output_dir,
            )
        src_dir = os.path.join(out_dir, stem)

        # 1. Download the e-print payload.
        try:
            interface = ArxivInterface()
            payload = interface.download_eprint(self.arxiv_id)
        except Exception as e:  # noqa: BLE001
            return self.format_error(
                error="E-print Download Failed",
                reason=str(e),
                context=f"arxiv_id={self.arxiv_id}",
                suggestion="Verify the arXiv id; arXiv may be rate-limiting",
            )
        data = payload["data"]

        # 2. Classify + unwrap.
        try:
            kind = detect_payload(data)
            if kind == "gzip":
                data = gunzip_capped(data)
                kind = detect_payload(data)
                if kind == "gzip":  # double-wrapped is not a thing; treat as tex
                    kind = "tex"
            if kind == "pdf":
                return json.dumps(
                    {
                        "status": "ok",
                        "schema": SCHEMA_VERSION,
                        "arxiv_id": self.arxiv_id,
                        "source_type": "pdf_only",
                        "cached": False,
                        "suggestion": (
                            "No LaTeX source on arXiv for this paper; use "
                            "FetchPaperPDFTool + ExtractPaperTextTool instead."
                        ),
                    },
                    indent=2,
                )

            os.makedirs(src_dir, exist_ok=True)
            if kind == "tar":
                files = safe_extract_tar(data, src_dir)
                source_type = "tar"
                main_rel = resolve_main_tex(src_dir)
                if main_rel is None:
                    return self.format_error(
                        error="No TeX Found",
                        reason="archive extracted but contains no .tex files",
                        context=f"n_files={len(files)}",
                    )
            else:  # single tex (or unknown-but-texish)
                main_rel = "main.tex"
                files = [main_rel]
                with open(os.path.join(src_dir, main_rel), "wb") as fh:
                    fh.write(data)
                source_type = "single_tex"

            main_abs = os.path.join(src_dir, main_rel)
            with open(main_abs, "r", encoding="utf-8", errors="replace") as fh:
                tex = fh.read()
            if self.inline_one_level is not False:
                tex = inline_inputs(tex, src_dir, main_rel)
            tex = strip_comments(tex)

            os.makedirs(os.path.dirname(tex_abs), exist_ok=True)
            with open(tex_abs, "w", encoding="utf-8") as fh:
                fh.write(tex)
        except ValueError as e:  # archive-safety violations, caps, bombs
            return self.format_error(
                error="Unsafe or Invalid Archive",
                reason=str(e),
                context=f"arxiv_id={self.arxiv_id}",
            )
        except OSError as e:
            return self.format_error(error="Filesystem Error", reason=str(e))

        n = self.preview_chars if self.preview_chars is not None else 2000
        return json.dumps(
            {
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "arxiv_id": self.arxiv_id,
                "source_type": source_type,
                "source_dir": os.path.relpath(src_dir, base),
                "main_tex": os.path.join(os.path.relpath(src_dir, base), main_rel),
                "tex_path": tex_rel,
                "n_files": len(files),
                "chars": len(tex),
                "preview": tex[:n],
                "cached": False,
            },
            indent=2,
        )


# ======================== Text extraction ====================== #


class ExtractPaperTextTool(BaseTool):
    """
    Extract full plain text from a PDF (PyMuPDF) into the sandbox.

    Use after FetchPaperPDFTool to obtain the text a downstream tool/agent reads
    to locate and extract the Lagrangian, conventions, and field content.

    Input:
        pdf_path: Path to the PDF, relative to base_directory
                  (e.g. "pdfs/2103.02708.pdf").
        output_path: Optional .txt output path relative to base_directory
                     (default: same stem under "text/").
        preview_chars: Length of the inline text preview returned (default 2000).

    Behavior:
        Writes the extracted text to {output_path} under base_directory and
        returns its path, page count, character count, and a preview. Read the
        full text from text_path when you need more than the preview.

    Returns JSON:
        {"status": "ok", "schema": "literature-1.0",
         "text_path": "text/2103.02708.txt", "pages": N, "chars": M,
         "preview": "..."}
    """

    # ======================== Runtime fields ======================== #
    pdf_path: str = RuntimeField(
        description="Path to the PDF relative to base_directory"
    )
    output_path: Optional[str] = RuntimeField(
        default=None,
        description="Optional .txt output path (relative to base_directory)",
    )
    preview_chars: Optional[int] = RuntimeField(
        default=2000, description="Length of the inline text preview (default 2000)"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        description="Base sandbox directory for file operations"
    )
    # ================================================================ #

    def _run(self) -> str:
        src = _safe_join(self.base_directory, self.pdf_path)
        if src is None:
            return self.format_error(
                error="Access Denied",
                reason="pdf_path escapes base_directory",
                context=self.pdf_path,
                suggestion="Use a relative path inside base_directory",
            )
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason="PDF does not exist",
                context=self.pdf_path,
            )

        if self.output_path:
            out = _safe_join(self.base_directory, self.output_path)
            if out is None:
                return self.format_error(
                    error="Access Denied",
                    reason="output_path escapes base_directory",
                    context=self.output_path,
                )
        else:
            stem = os.path.splitext(os.path.basename(src))[0]
            out = os.path.join(os.path.realpath(self.base_directory), "text", f"{stem}.txt")

        try:
            text, pages = ArxivInterface.extract_text(src)
        except ImportError as e:
            return self.format_error(
                error="Dependency Missing",
                reason="PyMuPDF is required for text extraction",
                suggestion="Install with: pip install pymupdf",
                context=str(e),
            )
        except Exception as e:  # noqa: BLE001
            return self.format_error(
                error="Extraction Failed",
                reason=str(e),
                context=self.pdf_path,
            )

        try:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as e:
            return self.format_error(
                error="Filesystem Error",
                reason=str(e),
                context=f"output_path={self.output_path or out}",
            )

        n = self.preview_chars if self.preview_chars is not None else 2000
        return json.dumps(
            {
                "status": "ok",
                "schema": SCHEMA_VERSION,
                "text_path": os.path.relpath(out, os.path.realpath(self.base_directory)),
                "pages": pages,
                "chars": len(text),
                "preview": text[:n],
            },
            indent=2,
        )
