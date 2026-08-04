#!/usr/bin/env python3
"""
# test_literature.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Tests for the arXiv literature toolkit.

Offline tests (no network) cover the query builder, PDF text extraction on a
synthetic PDF, path-traversal safety, and error formatting. Network tests
(arXiv search + PDF download) are attempted and skipped gracefully if arXiv is
unreachable, so the suite passes in offline/CI environments.
"""

import argparse
import io
import json
import os
import shutil
import sys
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent
REPO_ROOT = TOOL_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.literature import arxiv_interface
from tools.literature.arxiv_interface import (
    ArxivInterface,
    normalize_arxiv_id,
    validate_arxiv_id,
)
from tools.literature.literature_tools import (
    ArxivSearchTool,
    ExtractPaperTextTool,
    FetchPaperPDFTool,
    _safe_join,
)

TEST_DIR = TOOL_DIR / "test_files"

# Don't sleep between requests in tests (module-level shared limiter).
arxiv_interface._LIMITER.interval = 0.0

# Minimal arXiv Atom feed fixture (namespaces + version suffix + pdf link).
_ATOM_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2103.02708v2</id>
    <title>A scalar leptoquark model</title>
    <summary>We study a scalar leptoquark.</summary>
    <published>2021-03-04T00:00:00Z</published>
    <updated>2021-05-01T00:00:00Z</updated>
    <author><name>Jane Physicist</name></author>
    <author><name>John Theorist</name></author>
    <arxiv:primary_category term="hep-ph"/>
    <category term="hep-ph"/>
    <category term="hep-ex"/>
    <arxiv:doi>10.1000/xyz123</arxiv:doi>
    <link rel="alternate" href="http://arxiv.org/abs/2103.02708v2"/>
    <link title="pdf" type="application/pdf" href="http://arxiv.org/pdf/2103.02708v2"/>
  </entry>
</feed>"""


class _FakeStreamResponse:
    """Minimal stand-in for a streaming ``requests`` Response context manager."""

    def __init__(self, content: bytes, status: int = 200):
        self._content = content
        self.status_code = status

    def raise_for_status(self):
        import requests

        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_synthetic_pdf(path: Path, text: str) -> None:
    """Write a one-page PDF containing ``text`` using PyMuPDF."""
    import pymupdf

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_query_builder() -> bool:
    print(">> Testing arXiv query builder...\n")
    q = ArxivInterface.build_query("scalar leptoquark", ["BSM", "leptoquark"])
    assert 'cat:hep-ph' in q, q
    assert 'ti:"scalar leptoquark"' in q, q
    assert 'abs:"BSM"' in q, q
    assert normalize_arxiv_id("2103.02708v2") == "2103.02708"
    print("[✓] Query builder test passed\n")
    return True


def test_pdf_extraction() -> bool:
    print(">> Testing PDF text extraction...\n")
    marker = "SCALAR LEPTOQUARK LAGRANGIAN MARKER"
    pdf_rel = "input/sample.pdf"
    _make_synthetic_pdf(TEST_DIR / pdf_rel, marker)

    tool = ExtractPaperTextTool(pdf_path=pdf_rel, base_directory=str(TEST_DIR))
    result = json.loads(tool._run())

    assert result.get("status") == "ok", result
    assert result["pages"] == 1, result
    assert marker in result["preview"], result["preview"]
    assert (TEST_DIR / result["text_path"]).exists()
    print("[✓] PDF extraction test passed\n")
    return True


def test_path_traversal() -> bool:
    print(">> Testing path-traversal safety...\n")
    tool = ExtractPaperTextTool(
        pdf_path="../../../../etc/passwd", base_directory=str(TEST_DIR)
    )
    result = tool._run()
    assert "error" in result.lower() or "denied" in result.lower(), result
    print("[✓] Path-traversal test passed\n")
    return True


def test_extract_error_handling() -> bool:
    print(">> Testing extraction error handling...\n")
    tool = ExtractPaperTextTool(pdf_path="input/nonexistent.pdf", base_directory=str(TEST_DIR))
    result = tool._run()
    assert "not found" in result.lower() or "error" in result.lower(), result
    print("[✓] Error-handling test passed\n")
    return True


def test_search_missing_params() -> bool:
    print(">> Testing search parameter validation...\n")
    tool = ArxivSearchTool(base_directory=str(TEST_DIR))
    result = tool._run()
    assert "error" in result.lower() or "missing" in result.lower(), result
    print("[✓] Search parameter-validation test passed\n")
    return True


def test_arxiv_search_live() -> bool:
    print(">> Testing live arXiv search (network)...\n")
    tool = ArxivSearchTool(
        model_name="scalar leptoquark", keywords=["leptoquark"], max_results=3,
        base_directory=str(TEST_DIR),
    )
    try:
        result = json.loads(tool._run())
    except Exception as e:  # noqa: BLE001
        print(f"[i] Skipping live search (error: {e})\n")
        return True
    if result.get("status") != "ok":
        print(f"[i] Skipping live search (network unavailable): {result.get('reason')}\n")
        return True
    assert result["count"] >= 1, result
    assert result["papers"][0].get("arxiv_id"), result
    print(f"[✓] Live search returned {result['count']} papers\n")
    return True


def test_arxiv_id_validation() -> bool:
    print(">> Testing arXiv id validation/normalization...\n")
    assert validate_arxiv_id("2103.02708") == "2103.02708"
    assert validate_arxiv_id("2103.02708v2") == "2103.02708v2"
    assert validate_arxiv_id("arXiv:2103.02708v2") == "2103.02708v2"
    assert validate_arxiv_id("hep-ph/9905221") == "hep-ph/9905221"
    assert validate_arxiv_id("../../etc/passwd") is None
    assert validate_arxiv_id("not an id") is None
    assert validate_arxiv_id("") is None
    assert normalize_arxiv_id("arXiv:2103.02708v2") == "2103.02708"
    print("[✓] arXiv id validation test passed\n")
    return True


def test_atom_parsing() -> bool:
    print(">> Testing Atom feed parsing (namespaces + version)...\n")
    papers = ArxivInterface()._parse_feed(_ATOM_FIXTURE)
    assert len(papers) == 1, papers
    p = papers[0]
    assert p["arxiv_id"] == "2103.02708", p  # version stripped for canonical id
    assert p["title"] == "A scalar leptoquark model", p
    assert p["authors"] == ["Jane Physicist", "John Theorist"], p
    assert p["categories"][0] == "hep-ph", p
    assert p["doi"] == "10.1000/xyz123", p
    assert p["pdf_url"].startswith("https://"), p  # http -> https upgrade
    assert p["pdf_url"].endswith("2103.02708v2"), p
    assert p["abs_url"].startswith("https://"), p
    assert p["published"] == "2021-03-04", p
    print("[✓] Atom parsing test passed\n")
    return True


def test_fetch_rejects_bad_url() -> bool:
    print(">> Testing PDF fetch URL allowlist (SSRF guard)...\n")
    tool = FetchPaperPDFTool(pdf_url="http://evil.example.com/x.pdf", base_directory=str(TEST_DIR))
    result = tool._run()
    assert "denied" in result.lower() or "error" in result.lower(), result
    # http (non-https) arXiv should also be refused
    tool2 = FetchPaperPDFTool(pdf_url="http://arxiv.org/pdf/2103.02708.pdf", base_directory=str(TEST_DIR))
    assert "denied" in tool2._run().lower()
    print("[✓] URL allowlist test passed\n")
    return True


def test_fetch_rejects_invalid_id() -> bool:
    print(">> Testing PDF fetch id validation...\n")
    tool = FetchPaperPDFTool(arxiv_id="../../etc/passwd", base_directory=str(TEST_DIR))
    result = tool._run()
    assert "invalid" in result.lower() or "error" in result.lower(), result
    print("[✓] id validation test passed\n")
    return True


def test_fetch_download_and_cache_mocked() -> bool:
    print(">> Testing PDF download + cache identity (mocked network)...\n")
    fake_pdf = b"%PDF-1.4\n" + b"x" * 512
    with mock.patch.object(
        arxiv_interface.requests.Session, "get",
        return_value=_FakeStreamResponse(fake_pdf),
    ):
        r1 = json.loads(
            FetchPaperPDFTool(arxiv_id="2103.02708", base_directory=str(TEST_DIR))._run()
        )
        assert r1["status"] == "ok" and r1["cached"] is False, r1
        assert (TEST_DIR / r1["pdf_path"]).exists()

        # Same id again -> cache hit, same hash.
        r1b = json.loads(
            FetchPaperPDFTool(arxiv_id="2103.02708", base_directory=str(TEST_DIR))._run()
        )
        assert r1b["cached"] is True and r1b["sha256"] == r1["sha256"], r1b

        # Different id -> different file (no cache collision).
        r2 = json.loads(
            FetchPaperPDFTool(arxiv_id="1811.07920", base_directory=str(TEST_DIR))._run()
        )
        assert r2["status"] == "ok" and r2["pdf_path"] != r1["pdf_path"], r2

    # Non-PDF payload must be rejected.
    with mock.patch.object(
        arxiv_interface.requests.Session, "get",
        return_value=_FakeStreamResponse(b"<html>not a pdf</html>"),
    ):
        r3 = FetchPaperPDFTool(arxiv_id="2000.00001", base_directory=str(TEST_DIR))._run()
        assert "error" in r3.lower() or "not a valid pdf" in r3.lower(), r3
    print("[✓] Download + cache-identity test passed\n")
    return True


def test_fetch_both_args_precedence() -> bool:
    print(">> Testing fetch arxiv_id precedence over pdf_url...\n")
    fake_pdf = b"%PDF-1.4\n" + b"y" * 128
    with mock.patch.object(
        arxiv_interface.requests.Session, "get",
        return_value=_FakeStreamResponse(fake_pdf),
    ):
        r = json.loads(
            FetchPaperPDFTool(
                arxiv_id="2103.02708",
                pdf_url="https://arxiv.org/pdf/1811.07920.pdf",
                base_directory=str(TEST_DIR),
            )._run()
        )
    assert r["status"] == "ok", r
    # Cache filename must key on arxiv_id (the winner), not the URL.
    assert "2103.02708" in r["pdf_path"], r
    assert "1811.07920" not in r["pdf_path"], r
    print("[✓] Both-args precedence test passed\n")
    return True


def test_cache_symlink_not_followed() -> bool:
    print(">> Testing cache does not follow a leaf symlink...\n")
    base = TEST_DIR / "cachebox"
    pdf_dir = base / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    # A malicious symlink at the cache path pointing outside the sandbox.
    outside_pdf = TEST_DIR / "secret.pdf"
    outside_pdf.write_bytes(b"%PDF-1.4\nSECRET")
    dest = pdf_dir / "2103.02708.pdf"
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.symlink(outside_pdf, dest)

    fake_pdf = b"%PDF-1.4\nLEGIT-DOWNLOAD"
    with mock.patch.object(
        arxiv_interface.requests.Session, "get",
        return_value=_FakeStreamResponse(fake_pdf),
    ):
        r = json.loads(
            FetchPaperPDFTool(arxiv_id="2103.02708", base_directory=str(base))._run()
        )
    assert r["status"] == "ok", r
    assert r["cached"] is False, r  # symlink not served as cache
    # The symlink was atomically replaced by the real download, not followed.
    assert not (pdf_dir / "2103.02708.pdf").is_symlink(), "symlink should be replaced"
    assert outside_pdf.read_bytes() == b"%PDF-1.4\nSECRET", "outside file must be untouched"
    print("[✓] Cache symlink-safety test passed\n")
    return True


def _make_tar(members, symlink=None, absolute=None):
    """Build an in-memory tar. members: {name: bytes}."""
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        if symlink:
            info = tarfile.TarInfo(name=symlink[0])
            info.type = tarfile.SYMTYPE
            info.linkname = symlink[1]
            tf.addfile(info)
        if absolute:
            info = tarfile.TarInfo(name=absolute)
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))
    return buf.getvalue()


def _gz(data: bytes) -> bytes:
    import gzip as _gzip

    return _gzip.compress(data)


def test_source_archive_safety() -> bool:
    print(">> Testing source-archive safety (tar-slip, links, caps)...\n")
    from tools.literature.source_archive import safe_extract_tar

    dest = str(TEST_DIR / "srcbox")

    for bad_tar, label in [
        (_make_tar({"../evil.tex": b"x"}), "traversal ../"),
        (_make_tar({}, absolute="/etc/evil.tex"), "absolute path"),
        (_make_tar({"ok.tex": b"x"}, symlink=("link.tex", "/etc/passwd")), "symlink"),
        (_make_tar({f"f{i}.tex": b"x" for i in range(11)}), "member count cap"),
    ]:
        try:
            if label == "member count cap":
                safe_extract_tar(bad_tar, dest, max_members=10)
            else:
                safe_extract_tar(bad_tar, dest)
            raise AssertionError(f"{label}: should have been rejected")
        except ValueError:
            pass

    # Size caps.
    big = _make_tar({"big.tex": b"y" * 2048})
    try:
        safe_extract_tar(big, dest, max_member_bytes=1024)
        raise AssertionError("member size cap not enforced")
    except ValueError:
        pass

    # A benign archive extracts.
    ok = _make_tar({"paper/ms.tex": b"\\documentclass{article}"})
    files = safe_extract_tar(ok, dest)
    assert files == [os.path.join("paper", "ms.tex")], files
    print("[✓] Source-archive safety test passed\n")
    return True


def test_payload_detection_and_tex_processing() -> bool:
    print(">> Testing payload detection, main-tex resolution, comments, inlining...\n")
    from tools.literature.source_archive import (
        detect_payload,
        gunzip_capped,
        inline_inputs,
        resolve_main_tex,
        strip_comments,
    )

    assert detect_payload(_gz(b"anything")) == "gzip"
    assert detect_payload(b"%PDF-1.4 x") == "pdf"
    assert detect_payload(_make_tar({"a.tex": b"x"})) == "tar"
    assert detect_payload(b"\\documentclass{article}") == "tex"
    assert gunzip_capped(_gz(b"hello")) == b"hello"
    try:
        gunzip_capped(_gz(b"z" * 4096), max_bytes=1024)
        raise AssertionError("gunzip cap not enforced")
    except ValueError:
        pass

    # Main-tex resolution with decoys: appendix (no \documentclass) + main.
    src = TEST_DIR / "resolvebox"
    (src / "sub").mkdir(parents=True, exist_ok=True)
    (src / "appendix.tex").write_text("\\section{app}")
    (src / "sub" / "real.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\nBody\n\\end{document}"
    )
    assert resolve_main_tex(str(src)) == os.path.join("sub", "real.tex")

    # Comment stripping preserves escaped \%.
    out = strip_comments("a \\% kept % dropped\nplain % gone")
    assert out == "a \\% kept \nplain ", repr(out)

    # One-level inlining with containment + missing marker.
    (src / "sub" / "part.tex").write_text("INLINED-CONTENT")
    main = "\\input{part}\n\\input{../../../etc/passwd}\n\\input{missing}"
    inlined = inline_inputs(main, str(src), os.path.join("sub", "real.tex"))
    assert "INLINED-CONTENT" in inlined, inlined
    assert "passwd' not inlined" in inlined or "not inlined" in inlined, inlined
    print("[✓] Payload/tex-processing test passed\n")
    return True


def test_arxiv_source_tool_mocked() -> bool:
    print(">> Testing ArxivSourceTool end-to-end (mocked network)...\n")
    from tools.literature.literature_tools import ArxivSourceTool

    tar = _make_tar(
        {
            "ms.tex": b"\\documentclass{article}\n% comment\n\\input{lag}\n\\begin{document}\\end{document}",
            "lag.tex": b"L = y S1 u e % yukawa",
        }
    )
    with mock.patch.object(
        arxiv_interface.requests.Session, "get",
        return_value=_FakeStreamResponse(_gz(tar)),
    ):
        r = json.loads(
            ArxivSourceTool(arxiv_id="2103.02708", base_directory=str(TEST_DIR))._run()
        )
    assert r["status"] == "ok" and r["source_type"] == "tar", r
    assert r["n_files"] == 2, r
    text = (TEST_DIR / r["tex_path"]).read_text()
    assert "L = y S1 u e" in text and "% comment" not in text, text[:200]

    # Cache hit on second call (no network needed).
    r2 = json.loads(
        ArxivSourceTool(arxiv_id="2103.02708", base_directory=str(TEST_DIR))._run()
    )
    assert r2["cached"] is True, r2

    # pdf_only payload.
    with mock.patch.object(
        arxiv_interface.requests.Session, "get",
        return_value=_FakeStreamResponse(b"%PDF-1.4 binary"),
    ):
        r3 = json.loads(
            ArxivSourceTool(arxiv_id="1811.07920", base_directory=str(TEST_DIR))._run()
        )
    assert r3["status"] == "ok" and r3["source_type"] == "pdf_only", r3
    assert "FetchPaperPDFTool" in r3.get("suggestion", ""), r3

    # Malicious archive rejected via the tool path.
    with mock.patch.object(
        arxiv_interface.requests.Session, "get",
        return_value=_FakeStreamResponse(_gz(_make_tar({"../evil.tex": b"x"}))),
    ):
        r4 = ArxivSourceTool(arxiv_id="2000.00001", base_directory=str(TEST_DIR))._run()
    assert "unsafe" in r4.lower() or "error" in r4.lower(), r4

    # Single gzipped .tex payload.
    with mock.patch.object(
        arxiv_interface.requests.Session, "get",
        return_value=_FakeStreamResponse(_gz(b"\\documentclass{a}\nBody % c")),
    ):
        r5 = json.loads(
            ArxivSourceTool(arxiv_id="2005.06475", base_directory=str(TEST_DIR))._run()
        )
    assert r5["status"] == "ok" and r5["source_type"] == "single_tex", r5
    print("[✓] ArxivSourceTool mocked end-to-end test passed\n")
    return True


def test_symlink_escape() -> bool:
    print(">> Testing symlink-escape rejection in sandbox...\n")
    base = TEST_DIR / "sandbox"
    base.mkdir(parents=True, exist_ok=True)
    outside = TEST_DIR / "outside"
    outside.mkdir(parents=True, exist_ok=True)
    link = base / "escape"
    if not link.exists():
        os.symlink(outside, link)
    # A path through the symlinked dir resolves outside base -> rejected.
    assert _safe_join(str(base), "escape/evil.txt") is None
    assert _safe_join(str(base), "../outside/evil.txt") is None
    # A legitimate in-sandbox path is allowed.
    assert _safe_join(str(base), "ok/file.txt") is not None
    print("[✓] Symlink-escape test passed\n")
    return True


def cleanup_test_files() -> None:
    print("\n>> Cleaning up test files...\n")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"[✓] Removed: {TEST_DIR.name}\n")
    else:
        print("[i] No test files to clean up\n")


TESTS = [
    test_query_builder,
    test_arxiv_id_validation,
    test_atom_parsing,
    test_pdf_extraction,
    test_path_traversal,
    test_symlink_escape,
    test_extract_error_handling,
    test_search_missing_params,
    test_fetch_rejects_bad_url,
    test_fetch_rejects_invalid_id,
    test_fetch_download_and_cache_mocked,
    test_fetch_both_args_precedence,
    test_cache_symlink_not_followed,
    test_source_archive_safety,
    test_payload_detection_and_tex_processing,
    test_arxiv_source_tool_mocked,
    test_arxiv_search_live,
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the literature toolkit")
    parser.add_argument("--keep-files", action="store_true", help="Keep test-generated files")
    args = parser.parse_args()

    all_passed = True
    for test in TESTS:
        try:
            if not test():
                all_passed = False
        except Exception as e:  # noqa: BLE001
            print(f"[✗] {test.__name__} failed: {e}\n")
            all_passed = False

    if not args.keep_files:
        cleanup_test_files()
    else:
        print("\n[i] Keeping test files (--keep-files set)\n")

    if all_passed:
        print("[✓] All tests passed!\n")
        sys.exit(0)
    print("[✗] Some tests failed!\n")
    sys.exit(1)
