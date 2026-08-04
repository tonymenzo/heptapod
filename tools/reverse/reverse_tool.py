"""
# reverse_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Reverse Lagrangian check: verify a generated .fr by making a BLANK-SLATE
agent reconstruct the physics from the file alone, then (optionally) letting
a second fresh instance compare the reconstruction against the source paper.

Why: the forward chain (extract -> generate -> compile -> MadGraph) can be
self-consistent yet wrong. Reversing the direction with an instance that has
no memory of the paper or the extraction breaks that circularity: if the .fr
really encodes the paper's model, an independent reader should recover the
paper's Lagrangian from it. The output is a review package a physicist signs
off on — the tool never issues the final verdict.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import List, Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from tools.frgen.fr_parser import parse_lagrangian_terms
from tools.logging.audit import append_event
from tools.reverse.blank_agent import DEFAULT_BLANK_AGENT_CMD, run_blank_agent
from tools.reverse.pdf_build import compile_review_pdf
from tools.reverse.prompts import CROSSCHECK_PROMPT, RECONSTRUCT_PROMPT
from tools.reverse.review_package import build_review_md
from tools.reverse.sanitize import sanitize_fr

SCHEMA_VERSION = "reverse-lagrangian-1.1"

_ACTIONS = ("reconstruct", "crosscheck", "full")


def _safe_join(base_directory: str, rel_or_abs: str) -> Optional[str]:
    if not rel_or_abs:
        return None
    base = os.path.realpath(base_directory)
    full = os.path.realpath(os.path.join(base, rel_or_abs))
    if full != base and not full.startswith(base + os.sep):
        return None
    return full


class ReverseLagrangianTool(BaseTool):
    """
    Blank-slate reverse check of a FeynRules .fr model.

    Sanitizes the .fr (comments, M$Information, model name, prose labels
    removed), hands it to a FRESH sandboxed agent session with no other
    context, and collects a reconstruction of the physics: the Lagrangian in
    LaTeX term by term, a field table, and parameter meanings. With
    paper_tex_path set, a second fresh session compares the reconstruction
    against the paper (term-by-term agree/disagree table). Assembles the
    review package and compiles it to a LaTeX PDF (REVIEW.pdf) for a human
    physicist to sign off; REVIEW.md is kept as the typeset source.

    Runs take minutes — submit via submitjob(tool_name="reverselagrangian")
    and poll jobstatus rather than calling directly.

    Input:
        model_path: .fr file, relative to base_directory.
        action: "reconstruct" (sanitize + reconstruction), "crosscheck"
            (reuse an existing reconstruction, compare against the paper), or
            "full" (both; default).
        paper_tex_path: Paper LaTeX/text file relative to base_directory
            (e.g. from arxivsource). Required for crosscheck; optional for
            full (skipped with a note when absent).
        output_dir: Where artifacts go (default "reverse/<model stem>").
        timeout_sec: Per agent phase (default 900).
        ledger_name: Audit ledger to record events in.

    Returns JSON with artifact paths, per-phase agent_runs, the sanitizer
    report and the parsed Lagrangian term names. status "partial" when an
    agent phase failed but artifacts up to that point were written.
    """

    # ======================== Runtime fields ======================== #
    model_path: str = RuntimeField(
        description="Path to the .fr model file, relative to base_directory"
    )
    action: Optional[str] = RuntimeField(
        default="full", description="reconstruct | crosscheck | full"
    )
    paper_tex_path: Optional[str] = RuntimeField(
        default=None,
        description="Paper .tex/.txt (relative) for the cross-check phase",
    )
    output_dir: Optional[str] = RuntimeField(
        default=None, description="Output dir (default reverse/<model stem>)"
    )
    timeout_sec: Optional[int] = RuntimeField(
        default=900, description="Timeout per agent phase (seconds)"
    )
    ledger_name: Optional[str] = RuntimeField(
        default=None, description="Audit ledger name"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(description="Base sandbox directory")
    blank_agent_cmd: Optional[str] = StateField(
        default=DEFAULT_BLANK_AGENT_CMD,
        description="Command template for the blank-slate agent; tokens "
        "{output} and {prompt} are substituted",
    )
    # ================================================================ #

    def _audit(self, stage: str, status: str, summary: str, data: dict) -> None:
        try:
            append_event(self.base_directory, stage=stage, status=status,
                         summary=summary, data=data,
                         ledger_name=self.ledger_name or "audit.json")
        except Exception:  # noqa: BLE001 — audit is best-effort
            pass

    def _run(self) -> str:
        action = (self.action or "full").lower()
        if action not in _ACTIONS:
            return self.format_error(
                error="Invalid Input",
                reason=f"action must be one of {_ACTIONS}, got '{self.action}'",
            )
        src = _safe_join(self.base_directory, self.model_path)
        if src is None or not os.path.isfile(src):
            return self.format_error(
                error="File Not Found",
                reason=".fr model file does not exist inside base_directory",
                context=str(self.model_path),
            )
        paper_abs = None
        if self.paper_tex_path:
            paper_abs = _safe_join(self.base_directory, self.paper_tex_path)
            if paper_abs is None or not os.path.isfile(paper_abs):
                return self.format_error(
                    error="File Not Found",
                    reason="paper_tex_path does not exist inside base_directory",
                    context=str(self.paper_tex_path),
                )
        if action == "crosscheck" and paper_abs is None:
            return self.format_error(
                error="Invalid Input",
                reason="action=crosscheck requires paper_tex_path",
            )

        stem = os.path.splitext(os.path.basename(src))[0]
        out_rel = self.output_dir or os.path.join("reverse", stem)
        out_abs = _safe_join(self.base_directory, out_rel)
        if out_abs is None:
            return self.format_error(
                error="Access Denied", reason="output_dir escapes base_directory"
            )
        os.makedirs(out_abs, exist_ok=True)

        cmd = self.blank_agent_cmd or DEFAULT_BLANK_AGENT_CMD
        timeout = int(self.timeout_sec or 900)

        original_text = open(src, encoding="utf-8", errors="replace").read()
        sanitized, san_report = sanitize_fr(original_text)
        with open(os.path.join(out_abs, "sanitized.fr"), "w", encoding="utf-8") as fh:
            fh.write(sanitized)
        with open(os.path.join(out_abs, "sanitizer_report.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(san_report, fh, indent=2)

        terms = parse_lagrangian_terms(original_text)
        agent_runs: List[dict] = []
        recon_path = os.path.join(out_abs, "reconstruction.md")
        cross_path = os.path.join(out_abs, "crosscheck.md")

        # -- phase 1: blank-slate reconstruction -------------------------- #
        if action in ("reconstruct", "full"):
            workdir = tempfile.mkdtemp(prefix="reverse_recon_")
            try:
                shutil.copyfile(os.path.join(out_abs, "sanitized.fr"),
                                os.path.join(workdir, "sanitized.fr"))
                res = run_blank_agent(cmd, workdir, RECONSTRUCT_PROMPT,
                                      recon_path, timeout)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
            if res["ok"] and res["output_text"]:
                # the engine may have written {output} itself; fill it from
                # the captured text only when it's missing or empty
                existing = ""
                if os.path.isfile(recon_path):
                    existing = open(recon_path, encoding="utf-8").read().strip()
                if not existing:
                    with open(recon_path, "w", encoding="utf-8") as fh:
                        fh.write(res["output_text"])
            agent_runs.append({"phase": "reconstruct", "exit_code": res["exit_code"],
                               "seconds": res["seconds"], "error": res["error"],
                               "stderr_tail": (res["stderr_tail"] or "")[-300:]
                               if res["error"] else None})
            self._audit("reverse_reconstruct", "ok" if res["ok"] else "fail",
                        f"blank-slate reconstruction of {stem}",
                        {"seconds": res["seconds"], "error": res["error"]})

        # -- phase 2: paper cross-check ----------------------------------- #
        recon_exists = os.path.isfile(recon_path) and \
            open(recon_path, encoding="utf-8").read().strip()
        if action in ("crosscheck", "full") and paper_abs and recon_exists:
            workdir = tempfile.mkdtemp(prefix="reverse_cross_")
            try:
                # ONLY the paper + reconstruction — never the .fr
                shutil.copyfile(paper_abs, os.path.join(workdir, "paper.tex"))
                shutil.copyfile(recon_path, os.path.join(workdir, "reconstruction.md"))
                res = run_blank_agent(cmd, workdir, CROSSCHECK_PROMPT,
                                      cross_path, timeout)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
            if res["ok"] and res["output_text"] and (
                    not os.path.isfile(cross_path)
                    or not open(cross_path, encoding="utf-8").read().strip()):
                with open(cross_path, "w", encoding="utf-8") as fh:
                    fh.write(res["output_text"])
            agent_runs.append({"phase": "crosscheck", "exit_code": res["exit_code"],
                               "seconds": res["seconds"], "error": res["error"],
                               "stderr_tail": (res["stderr_tail"] or "")[-300:]
                               if res["error"] else None})
            self._audit("reverse_crosscheck", "ok" if res["ok"] else "fail",
                        f"paper cross-check of {stem}",
                        {"seconds": res["seconds"], "error": res["error"]})

        # -- assemble the review package ---------------------------------- #
        recon_md = open(recon_path, encoding="utf-8").read() \
            if os.path.isfile(recon_path) else None
        cross_md = open(cross_path, encoding="utf-8").read() \
            if os.path.isfile(cross_path) else None
        review = build_review_md(
            model_path=self.model_path,
            terms=terms,
            sanitizer_report=san_report,
            reconstruction_md=recon_md,
            crosscheck_md=cross_md,
            paper_ref=self.paper_tex_path,
        )
        review_path = os.path.join(out_abs, "REVIEW.md")
        with open(review_path, "w", encoding="utf-8") as fh:
            fh.write(review)

        # The physicist deliverable is a LaTeX-compiled PDF; REVIEW.md stays
        # on disk as the pandoc source and as the fallback when no converter
        # is installed.
        pdf_res = compile_review_pdf(review_path)
        review_pdf_rel = os.path.join(out_rel, "REVIEW.pdf") \
            if pdf_res["ok"] else None
        self._audit("reverse_package", "ok", f"review package for {stem}",
                    {"review": review_pdf_rel or os.path.join(out_rel, "REVIEW.md"),
                     "pdf_error": pdf_res["error"]})

        failed = [r for r in agent_runs if r.get("error")]
        result = {
            "status": "partial" if failed else "ok",
            "schema": SCHEMA_VERSION,
            "action": action,
            "model_path": self.model_path,
            "sanitized_fr": os.path.join(out_rel, "sanitized.fr"),
            "reconstruction": os.path.join(out_rel, "reconstruction.md")
            if recon_md else None,
            "crosscheck": os.path.join(out_rel, "crosscheck.md")
            if cross_md else None,
            "review_package": review_pdf_rel or os.path.join(out_rel, "REVIEW.md"),
            "review_markdown_source": os.path.join(out_rel, "REVIEW.md"),
            "review_pdf_error": pdf_res["error"],
            "lagrangian_terms": [
                {"name": t["name"], "op": t["op"], "n_chars": len(t["expression"])}
                for t in terms
            ],
            "agent_runs": agent_runs,
            "sanitizer_report": san_report,
            "human_review_required": True,
        }
        return json.dumps(result, indent=2)
