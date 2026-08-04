"""
# source_archive.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Safe handling of arXiv e-print source archives.

arXiv's e-print endpoint serves the original submission: usually a gzipped tar
of LaTeX sources, sometimes a gzipped single .tex file, occasionally a bare PDF
(no source available). These helpers detect the payload, extract archives with
defense-in-depth against archive attacks (path traversal, links, bombs), resolve
the main .tex file, and normalize LaTeX text for downstream extraction.

Pure functions — no network, no tool framework — so they are unit-testable and
reusable (the convention-catalog script reuses safe_extract_tar for 2HDM.tar.gz).
"""

from __future__ import annotations

import gzip
import io
import os
import re
import tarfile
from typing import List, Optional, Tuple

# Defense-in-depth caps for archive extraction (arXiv sources are small; a
# legitimate hep-ph submission is a few MB and a few hundred files).
MAX_MEMBERS = 2000
MAX_MEMBER_BYTES = 50 * 1024 * 1024
MAX_TOTAL_BYTES = 200 * 1024 * 1024

_GZIP_MAGIC = b"\x1f\x8b"
_PDF_MAGIC = b"%PDF-"


def detect_payload(data: bytes) -> str:
    """Classify raw e-print bytes: 'gzip' | 'tar' | 'pdf' | 'tex' | 'unknown'.

    arXiv serves gzip (tar.gz or .tex.gz) for sources and raw %PDF- when no
    source exists. A bare tar (rare) has the ustar magic at offset 257.
    """
    if data[:2] == _GZIP_MAGIC:
        return "gzip"
    if data[:5] == _PDF_MAGIC:
        return "pdf"
    if len(data) > 262 and data[257:262] == b"ustar":
        return "tar"
    # Heuristic: LaTeX text payloads start with printable ASCII and contain a backslash command early.
    head = data[:2048]
    if b"\\document" in head or b"\\input" in head or head.lstrip()[:1] in (b"%", b"\\"):
        return "tex"
    return "unknown"


def gunzip_capped(data: bytes, max_bytes: int = MAX_TOTAL_BYTES) -> bytes:
    """Stream-decompress gzip with a byte cap (bomb guard); never whole-buffer."""
    out = io.BytesIO()
    total = 0
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
        while True:
            chunk = gz.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"decompressed payload exceeds {max_bytes} bytes")
            out.write(chunk)
    return out.getvalue()


def _contained(base: str, target: str) -> bool:
    """realpath containment check, mirroring the tools' _safe_join convention."""
    base_r = os.path.realpath(base)
    full = os.path.realpath(target)
    return full == base_r or full.startswith(base_r + os.sep)


def safe_extract_tar(
    tar_bytes: bytes,
    dest_dir: str,
    max_members: int = MAX_MEMBERS,
    max_member_bytes: int = MAX_MEMBER_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> List[str]:
    """Extract a tar archive with belt-and-braces safety.

    Manual member vetting (reject absolute paths, ``..`` components, symlinks/
    hardlinks/devices/FIFOs; enforce per-member, total-size, and member-count
    caps; realpath containment) AND ``filter="data"`` (PEP 706) as a second
    layer. Returns the list of extracted file paths (relative to dest_dir).
    """
    os.makedirs(dest_dir, exist_ok=True)
    extracted: List[str] = []
    total = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
        members = tf.getmembers()
        if len(members) > max_members:
            raise ValueError(f"archive has {len(members)} members (cap {max_members})")
        for m in members:
            name = m.name
            if not name or name.startswith("/") or name.startswith("\\"):
                raise ValueError(f"absolute path in archive: {name!r}")
            parts = name.replace("\\", "/").split("/")
            if ".." in parts:
                raise ValueError(f"path traversal in archive: {name!r}")
            if m.isdir():
                continue  # directories created implicitly by file extraction
            if not m.isreg():
                # symlink/hardlink/device/FIFO — never extract
                raise ValueError(f"non-regular member in archive: {name!r} (type {m.type!r})")
            if m.size > max_member_bytes:
                raise ValueError(f"member {name!r} exceeds {max_member_bytes} bytes")
            total += m.size
            if total > max_total_bytes:
                raise ValueError(f"archive contents exceed {max_total_bytes} bytes")
            dest_path = os.path.join(dest_dir, *parts)
            if not _contained(dest_dir, dest_path):
                raise ValueError(f"member escapes destination: {name!r}")
            # Second layer: stdlib data filter (PEP 706) sanitizes metadata.
            tf.extract(m, path=dest_dir, filter="data")
            extracted.append(os.path.relpath(dest_path, dest_dir))
    return extracted


_DOCCLASS_RE = re.compile(r"^[^%\n]*\\documentclass", re.MULTILINE)
_BEGINDOC_RE = re.compile(r"^[^%\n]*\\begin\{document\}", re.MULTILINE)
_PREFERRED_NAMES = ("main.tex", "ms.tex", "paper.tex", "article.tex")


def resolve_main_tex(source_dir: str) -> Optional[str]:
    """Pick the main .tex file: \\documentclass > \\begin{document} > name > size.

    Returns a path relative to source_dir, or None if no .tex files exist.
    Multi-document sources pick one (documented limitation).
    """
    candidates: List[Tuple[str, str]] = []  # (relpath, text)
    for root, _dirs, files in os.walk(source_dir):
        for fname in files:
            if not fname.lower().endswith(".tex"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            candidates.append((os.path.relpath(path, source_dir), text))
    if not candidates:
        return None

    def rank(item: Tuple[str, str]) -> Tuple[int, int, int, int]:
        rel, text = item
        has_class = 1 if _DOCCLASS_RE.search(text) else 0
        has_begin = 1 if _BEGINDOC_RE.search(text) else 0
        named = 1 if os.path.basename(rel).lower() in _PREFERRED_NAMES else 0
        return (has_class, has_begin, named, len(text))

    candidates.sort(key=rank, reverse=True)
    return candidates[0][0]


def strip_comments(tex: str) -> str:
    """Remove LaTeX ``%`` comments (a ``%`` not preceded by a backslash).

    Skipping comment-stripping inside verbatim environments is a documented
    non-goal (rare in hep-ph model papers).
    """
    out_lines: List[str] = []
    for line in tex.splitlines():
        cut = None
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\":
                i += 2  # skip escaped char (covers \% and any \x)
                continue
            if ch == "%":
                cut = i
                break
            i += 1
        out_lines.append(line if cut is None else line[:cut])
    return "\n".join(out_lines)


_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")


def inline_inputs(tex: str, source_dir: str, main_rel: str) -> str:
    """Inline ``\\input{...}``/``\\include{...}`` one level deep.

    Referenced paths get ``.tex`` appended when missing an extension, must stay
    inside source_dir (containment check), and a missing file is replaced by an
    inline marker comment rather than an error.
    """
    base_dir = os.path.dirname(os.path.join(source_dir, main_rel))

    def _sub(match: "re.Match[str]") -> str:
        target = match.group(1).strip()
        if not os.path.splitext(target)[1]:
            target += ".tex"
        # Resolve relative to the main file's dir, then the source root.
        for root in (base_dir, source_dir):
            path = os.path.normpath(os.path.join(root, target))
            if _contained(source_dir, path) and os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        return "\n" + fh.read() + "\n"
                except OSError:
                    break
        return f"% [heptapod: missing or unsafe input {target!r} not inlined]"

    return _INPUT_RE.sub(_sub, tex)
