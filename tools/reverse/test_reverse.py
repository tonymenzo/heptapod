#!/usr/bin/env python3
"""
# test_reverse.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.

Tests for the reverse Lagrangian check.

All offline: the sanitizer/parser tests are pure functions, and the tool
end-to-end tests use a fake shell engine (test_files/fake_agent.sh) as
blank_agent_cmd — no codex, no network.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
TOOL_DIR = SCRIPT_PATH.parent
REPO_ROOT = TOOL_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.frgen.fr_parser import parse_fr, parse_lagrangian_terms  # noqa: E402
from tools.reverse.blank_agent import run_blank_agent  # noqa: E402
from tools.reverse.reverse_tool import ReverseLagrangianTool  # noqa: E402
from tools.reverse.sanitize import sanitize_fr  # noqa: E402

S1_FR = REPO_ROOT / "tools" / "feynrules" / "test_files" / "models" / "S1_LQ_RR.fr"
FAKE = TOOL_DIR / "test_files" / "fake_agent.sh"

_TMPDIRS = []


def _base() -> str:
    d = tempfile.mkdtemp(prefix="reverse_test_")
    _TMPDIRS.append(d)
    return d


def cleanup_test_files() -> None:
    for d in _TMPDIRS:
        shutil.rmtree(d, ignore_errors=True)


def _fake_cmd(mode: str) -> str:
    return f"sh {FAKE} {mode} {{output}} {{prompt}}"


def test_sanitizer_s1() -> bool:
    text = S1_FR.read_text()
    clean, report = sanitize_fr(text)
    ok = True
    ok &= "Tony Menzo" not in clean and "M$Information" not in clean
    ok &= 'M$ModelName = "ANON-MODEL";' in clean
    ok &= "(*" not in clean
    ok &= report["original_model_name"] == "S1_LQ_RR"
    ok &= report["information_blocks_removed"] == 1
    # physics preserved: same classes/params and same Lagrangian term names
    a, b = parse_fr(text), parse_fr(clean)
    ok &= len(a["classes"]) == len(b["classes"])
    ok &= len(a["parameters"]) == len(b["parameters"])
    ok &= [t["name"] for t in parse_lagrangian_terms(text)] == \
          [t["name"] for t in parse_lagrangian_terms(clean)]
    # prose scrubbed but symbols kept
    ok &= "yRR11" in clean and "S1" in clean
    ok &= 'Description -> ""' in clean or "Description" not in clean
    print(f"[{'✓' if ok else '✗'}] test_sanitizer_s1")
    return ok


def test_sanitizer_edge_cases() -> bool:
    tricky = (
        '(* outer (* nested *) still comment *)\n'
        'M$ModelName = "Weird";\n'
        'M$Information = { Authors -> "A", Emails -> "x{y}@z" };\n'
        'M$Parameters = { g == { Description -> "coupling \\"g\\"" } };\n'
        'L1 := g * phi;\n'
    )
    clean, report = sanitize_fr(tricky)
    ok = True
    ok &= "nested" not in clean and "Authors" not in clean
    ok &= "x{y}@z" not in clean
    ok &= 'ANON-MODEL' in clean
    ok &= "coupling" not in clean
    ok &= "L1 := g * phi" in clean
    ok &= report["information_blocks_removed"] == 1
    # idempotent
    clean2, _ = sanitize_fr(clean)
    ok &= clean2 == clean
    print(f"[{'✓' if ok else '✗'}] test_sanitizer_edge_cases")
    return ok


def test_parse_lagrangian_terms() -> bool:
    text = S1_FR.read_text()
    terms = parse_lagrangian_terms(text)
    names = [t["name"] for t in terms]
    ok = names == ["LkinS1", "L1YukRRNonHC", "L1YukRR", "LBSM"]
    ops = {t["name"]: t["op"] for t in terms}
    ok &= ops["L1YukRR"] == ":=" and ops["LBSM"] == "="
    ok &= "HC[L1YukRRNonHC]" in terms[2]["expression"]
    # a commented-out assignment must not appear
    ok &= "LDead" not in [t["name"] for t in
                          parse_lagrangian_terms("(* LDead := x; *)\nLReal = y;")]
    print(f"[{'✓' if ok else '✗'}] test_parse_lagrangian_terms")
    return ok


def test_blank_agent_modes() -> bool:
    ok = True
    wd = _base()
    out = os.path.join(_base(), "recon.md")
    r = run_blank_agent(_fake_cmd("ok"), wd, "PROMPT", out, timeout_sec=10)
    ok &= r["ok"] and "Reconstruction" in r["output_text"] and r["error"] is None
    # stdout fallback (engine ignores {output})
    out2 = os.path.join(_base(), "recon2.md")
    r = run_blank_agent(f"sh {FAKE} stdout", wd, "PROMPT", out2, timeout_sec=10)
    ok &= r["ok"] and "stdout" in r["output_text"]
    # failure is structured
    out3 = os.path.join(_base(), "recon3.md")
    r = run_blank_agent(_fake_cmd("fail"), wd, "PROMPT", out3, timeout_sec=10)
    ok &= (not r["ok"]) and r["exit_code"] == 3 and "not logged in" in r["stderr_tail"]
    # timeout is structured
    out4 = os.path.join(_base(), "recon4.md")
    r = run_blank_agent(_fake_cmd("hang"), wd, "PROMPT", out4, timeout_sec=2)
    ok &= (not r["ok"]) and r["exit_code"] == 124 and "timed out" in r["error"]
    # missing engine is structured
    r = run_blank_agent("/no/such/engine {output}", wd, "P", out4, timeout_sec=2)
    ok &= (not r["ok"]) and "not found" in r["error"]
    print(f"[{'✓' if ok else '✗'}] test_blank_agent_modes")
    return ok


def test_tool_full_with_fake_engine() -> bool:
    base = _base()
    os.makedirs(os.path.join(base, "model"))
    shutil.copyfile(S1_FR, os.path.join(base, "model", "S1.fr"))
    with open(os.path.join(base, "paper.tex"), "w") as fh:
        fh.write("\\section{Model} L = y S ue + h.c.")
    tool = ReverseLagrangianTool(
        base_directory=base, blank_agent_cmd=_fake_cmd("ok"),
        model_path="model/S1.fr", action="full", paper_tex_path="paper.tex",
        timeout_sec=20)
    res = json.loads(tool._run())
    ok = res["status"] == "ok" and res["human_review_required"] is True
    ok &= [t["name"] for t in res["lagrangian_terms"]] == \
          ["LkinS1", "L1YukRRNonHC", "L1YukRR", "LBSM"]
    ok &= len(res["agent_runs"]) == 2
    for rel in ("sanitized_fr", "reconstruction", "crosscheck",
                "review_package", "review_markdown_source"):
        ok &= res[rel] is not None and os.path.isfile(os.path.join(base, res[rel]))
    review = open(os.path.join(base, res["review_markdown_source"])).read()
    ok &= "Physicist sign-off" in review and "LkinS1" in review
    ok &= "S1_LQ_RR" in review  # original name revealed to the human, not the agent
    # deliverable is the compiled PDF when a converter exists, else the md
    if res["review_package"].endswith(".pdf"):
        with open(os.path.join(base, res["review_package"]), "rb") as fh:
            ok &= fh.read(5) == b"%PDF-"
        ok &= res["review_pdf_error"] is None
    else:
        ok &= res["review_pdf_error"] is not None
        print("    [i] no pandoc/xelatex — PDF fallback path exercised")
    san = open(os.path.join(base, res["sanitized_fr"])).read()
    ok &= "ANON-MODEL" in san and "Menzo" not in san
    # audit trail landed
    ledger = json.loads(open(os.path.join(base, "audit.json")).read())
    stages = [e["stage"] for e in ledger["events"]]
    ok &= "reverse_reconstruct" in stages and "reverse_package" in stages
    print(f"[{'✓' if ok else '✗'}] test_tool_full_with_fake_engine")
    return ok


def test_tool_partial_on_engine_failure() -> bool:
    base = _base()
    os.makedirs(os.path.join(base, "model"))
    shutil.copyfile(S1_FR, os.path.join(base, "model", "S1.fr"))
    tool = ReverseLagrangianTool(
        base_directory=base, blank_agent_cmd=_fake_cmd("fail"),
        model_path="model/S1.fr", action="reconstruct", timeout_sec=20)
    res = json.loads(tool._run())
    ok = res["status"] == "partial"
    ok &= res["agent_runs"][0]["error"] is not None
    ok &= res["reconstruction"] is None
    # review package still written, with the failure note
    ok &= os.path.isfile(os.path.join(base, res["review_package"]))
    print(f"[{'✓' if ok else '✗'}] test_tool_partial_on_engine_failure")
    return ok


def test_tool_input_errors() -> bool:
    base = _base()
    ok = True
    r = ReverseLagrangianTool(base_directory=base, model_path="nope.fr")._run()
    ok &= r.startswith("Error:") and "File Not Found" in r
    r = ReverseLagrangianTool(base_directory=base, model_path="../etc/passwd")._run()
    ok &= r.startswith("Error:")
    os.makedirs(os.path.join(base, "m"))
    shutil.copyfile(S1_FR, os.path.join(base, "m", "S1.fr"))
    r = ReverseLagrangianTool(base_directory=base, model_path="m/S1.fr",
                              action="crosscheck")._run()
    ok &= "requires paper_tex_path" in r
    r = ReverseLagrangianTool(base_directory=base, model_path="m/S1.fr",
                              action="bogus")._run()
    ok &= "action must be one of" in r
    print(f"[{'✓' if ok else '✗'}] test_tool_input_errors")
    return ok


def test_crosscheck_dir_isolation() -> bool:
    """The cross-check phase must never see the .fr — verified via the fake
    engine, which lists its working directory into the output."""
    base = _base()
    os.makedirs(os.path.join(base, "model"))
    shutil.copyfile(S1_FR, os.path.join(base, "model", "S1.fr"))
    with open(os.path.join(base, "paper.tex"), "w") as fh:
        fh.write("L = y S ue")
    lister = os.path.join(base, "lister.sh")
    with open(lister, "w") as fh:
        fh.write('#!/bin/sh\nls -1 . > "$1"\n')
    tool = ReverseLagrangianTool(
        base_directory=base, blank_agent_cmd=f"sh {lister} {{output}}",
        model_path="model/S1.fr", action="full", paper_tex_path="paper.tex",
        timeout_sec=20)
    res = json.loads(tool._run())
    cross_listing = open(os.path.join(base, res["crosscheck"])).read() \
        if res["crosscheck"] else ""
    ok = "paper.tex" in cross_listing and "reconstruction.md" in cross_listing
    ok &= ".fr" not in cross_listing
    print(f"[{'✓' if ok else '✗'}] test_crosscheck_dir_isolation")
    return ok


def test_pdf_build() -> bool:
    from tools.reverse.pdf_build import _find_binary, compile_review_pdf
    base = _base()
    md = os.path.join(base, "doc.md")
    with open(md, "w") as fh:
        fh.write(
            "# Review\n\nInline \\(x^2\\) and display math:\n\n"
            "\\[ \\Gamma = \\frac{|y|^2 M}{16\\pi} \\]\n\n"
            "| term | verdict |\n|---|---|\n| \\(P_L\\) coupling | agree |\n\n"
            "```mathematica\n"
            "L1YukRRNonHC := yRR11 * anti[CC[uR]][sp, 1, aa].lR[sp, 1] "
            "* HC[S1][aa];\n"
            "```\n"
        )
    ok = True
    # every failure mode is structured, never an exception
    r = compile_review_pdf(md, pandoc="/no/such/pandoc")
    ok &= (not r["ok"]) and "pandoc" in r["error"]
    r = compile_review_pdf(md, engine="/no/such/xelatex")
    ok &= (not r["ok"]) and "xelatex" in r["error"]
    r = compile_review_pdf(os.path.join(base, "missing.md"))
    ok &= (not r["ok"]) and "not found" in r["error"]
    # real compile when the toolchain is installed
    if _find_binary("pandoc") and _find_binary("xelatex"):
        r = compile_review_pdf(md)
        ok &= r["ok"] and r["error"] is None
        with open(r["pdf_path"], "rb") as fh:
            ok &= fh.read(5) == b"%PDF-"
        ok &= os.path.getsize(r["pdf_path"]) > 5000
    else:
        print("    [i] pandoc/xelatex not installed — live compile skipped")
    print(f"[{'✓' if ok else '✗'}] test_pdf_build")
    return ok


TESTS = [
    test_sanitizer_s1,
    test_sanitizer_edge_cases,
    test_parse_lagrangian_terms,
    test_pdf_build,
    test_blank_agent_modes,
    test_tool_full_with_fake_engine,
    test_tool_partial_on_engine_failure,
    test_tool_input_errors,
    test_crosscheck_dir_isolation,
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the reverse toolkit")
    parser.add_argument("--keep-files", action="store_true")
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

    if all_passed:
        print("[✓] All tests passed!\n")
        sys.exit(0)
    print("[✗] Some tests failed!\n")
    sys.exit(1)
