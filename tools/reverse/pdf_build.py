"""
# pdf_build.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Compile the reverse-check review package to a LaTeX PDF.

The review content is Markdown with embedded LaTeX math (the blank-slate
agent writes \\( \\) / \\[ \\] inline), so pandoc is the converter and a real
LaTeX engine (xelatex — Unicode-safe for agent-written prose) does the
typesetting. Binaries are resolved to absolute paths with fallbacks beyond
$PATH because the detached job runner inherits a minimal environment where
neither Homebrew's pandoc nor MacTeX's /Library/TeX/texbin may be visible.

compile_review_pdf never raises: a missing converter or a LaTeX failure is
reported in the returned dict and the Markdown source remains authoritative.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from typing import Optional

# Searched after $PATH; covers Homebrew (Apple Silicon + Intel), MacTeX and
# common Linux TeX Live installs.
_FALLBACK_DIRS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/Library/TeX/texbin",
    "/usr/local/texlive/bin",
)


def _find_binary(name: str) -> Optional[str]:
    """Absolute path of ``name`` from $PATH, then the fallback dirs."""
    found = shutil.which(name)
    if found:
        return found
    for d in _FALLBACK_DIRS:
        cand = os.path.join(d, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


# LaTeX control words the agents legitimately use. Anything else written as
# \word is almost always a typo for a plain identifier (e.g. the FeynRules
# parameter name betaL23 written as \betaL23), which xelatex rejects as an
# undefined control sequence. Used only in the failure-retry path.
_KNOWN_TEX = frozenset("""
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa
lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi
psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
frac dfrac tfrac sqrt bar hat tilde vec dot ddot overline underline widehat
widetilde overrightarrow mathrm mathbf mathit mathcal mathbb mathfrak mathsf
mathtt boldsymbol text textrm textbf textit texttt textsc emph operatorname
left right big Big bigg Bigg langle rangle lvert rvert lVert rVert vert Vert
cdot cdots ldots dots dotsb times div pm mp ast star circ bullet oplus ominus
otimes odot dagger ddagger prime partial nabla infty hbar ell Re Im
sum prod int oint iint lim sup inf max min arg det dim ker deg gcd Pr exp log
ln lg sin cos tan cot sec csc arcsin arccos arctan sinh cosh tanh coth
leq geq neq equiv sim simeq approx cong propto ll gg subset supset subseteq
supseteq in notin ni cup cap setminus emptyset forall exists neg land lor
to gets rightarrow leftarrow Rightarrow Leftarrow leftrightarrow
Leftrightarrow mapsto longrightarrow longleftarrow hookrightarrow
uparrow downarrow quad qquad hspace vspace phantom stackrel overset underset
begin end label ref eqref cite item section subsection subsubsection
paragraph caption centering hline cline multicolumn multirow textwidth
linewidth newline par noindent slash backslash textbackslash
usepackage documentclass footnote thanks appendix tableofcontents
mathop limits nolimits binom choose atop over displaystyle textstyle
scriptstyle scriptscriptstyle rm bf it sf tt cal frak bb LaTeX TeX
not among daleth eth gimel beth aleph wp top bot angle measuredangle
triangle square diamond dag ddag S P copyright pounds checkmark
""".split())

_MACRO_RE = re.compile(r"(?<!\\)\\([A-Za-z]+)")

# Legal-looking but fragile agent idioms that break xelatex, with safe
# equivalents. \not applied to a spacing command ("Missing { inserted"):
# \not\!D -> \not D keeps the slashed-D semantics and compiles.
_FRAGILE_FIXES = (
    (re.compile(r"\\not\\!"), r"\\not "),
)


def sanitize_unknown_macros(text: str) -> tuple[str, int]:
    """Repair agent-written TeX that xelatex rejects: drop the backslash from
    ``\\word`` control sequences not in the known LaTeX vocabulary (typos like
    ``\\betaL23`` for the parameter name ``betaL23``) and rewrite known-fragile
    idioms (``\\not\\!``). Fenced code blocks are left untouched. Returns
    ``(sanitized_text, n_replacements)``."""
    n = 0

    def _fix(m: re.Match) -> str:
        nonlocal n
        word = m.group(1)
        if word in _KNOWN_TEX:
            return m.group(0)
        n += 1
        return word

    def _clean(line: str) -> str:
        nonlocal n
        line = _MACRO_RE.sub(_fix, line)
        for pat, rep in _FRAGILE_FIXES:
            line, k = pat.subn(rep, line)
            n += k
        return line

    out, in_fence = [], False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
        else:
            out.append(line if in_fence else _clean(line))
    return "".join(out), n


def compile_review_pdf(
    md_path: str,
    pdf_path: Optional[str] = None,
    timeout_sec: int = 300,
    pandoc: Optional[str] = None,
    engine: Optional[str] = None,
) -> dict:
    """Compile ``md_path`` to ``pdf_path`` (default: same stem, .pdf).

    Returns ``{"ok", "pdf_path", "pandoc", "engine", "seconds", "error"}``.
    ``pandoc``/``engine`` args override binary discovery (used by tests).
    """
    t0 = time.time()
    pdf_path = pdf_path or os.path.splitext(md_path)[0] + ".pdf"

    def _fail(msg: str) -> dict:
        return {"ok": False, "pdf_path": None, "pandoc": pandoc,
                "engine": engine, "seconds": round(time.time() - t0, 1),
                "error": msg}

    if not os.path.isfile(md_path):
        return _fail(f"markdown source not found: {md_path}")
    pandoc = pandoc or _find_binary("pandoc")
    if not pandoc or not os.path.isfile(pandoc):
        return _fail("pandoc not found (install: brew install pandoc)")
    engine = engine or _find_binary("xelatex")
    if not engine or not os.path.isfile(engine):
        return _fail("xelatex not found (install MacTeX / TeX Live)")

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [os.path.dirname(engine), os.path.dirname(pandoc)]
        + env.get("PATH", "").split(os.pathsep)
    )

    def _attempt(src: str) -> Optional[str]:
        """One pandoc/xelatex run; returns None on success, error text on failure."""
        # tex_math_single_backslash: the agents write \( \) / \[ \] math,
        # which must parse as math (not raw text) in table cells and prose.
        cmd = [
            pandoc, src, "-o", pdf_path,
            "--from", "markdown+raw_tex+tex_math_single_backslash+tex_math_dollars",
            "--pdf-engine", engine,
            "-V", "geometry:margin=2.2cm",
            "-V", "fontsize=10pt",
            "-V", "colorlinks=true",
            # Wrap long verbatim lines (FeynRules terms overflow the margin
            # otherwise). fvextra ships with MacTeX/TeX Live.
            "-V", "header-includes=\\usepackage{fvextra}"
                  "\\DefineVerbatimEnvironment{Highlighting}{Verbatim}"
                  "{breaklines,breakanywhere,commandchars=\\\\\\{\\}}"
                  "\\fvset{breaklines,breakanywhere}",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout_sec, env=env)
        except subprocess.TimeoutExpired:
            return f"pandoc/LaTeX timed out after {timeout_sec}s"
        except OSError as e:  # engine binary broken, etc.
            return f"failed to launch pandoc: {e}"
        if proc.returncode != 0 or not os.path.isfile(pdf_path):
            tail = (proc.stderr or proc.stdout or "").strip()[-500:]
            return f"pandoc exit {proc.returncode}: {tail}"
        return None

    err = _attempt(md_path)
    if err is None:
        return {"ok": True, "pdf_path": pdf_path, "pandoc": pandoc,
                "engine": engine, "seconds": round(time.time() - t0, 1),
                "error": None, "sanitized_macros": 0}

    # Agent-written LaTeX occasionally invents control sequences (\betaL23
    # for the parameter name betaL23). Retry once with unknown macros
    # de-backslashed; the original Markdown stays untouched and authoritative.
    with open(md_path, encoding="utf-8", errors="replace") as fh:
        fixed, n = sanitize_unknown_macros(fh.read())
    if n == 0:
        return _fail(err)
    fixed_path = os.path.splitext(md_path)[0] + "_texfixed.md"
    with open(fixed_path, "w", encoding="utf-8") as fh:
        fh.write(fixed)
    retry_err = _attempt(fixed_path)
    if retry_err is None:
        return {"ok": True, "pdf_path": pdf_path, "pandoc": pandoc,
                "engine": engine, "seconds": round(time.time() - t0, 1),
                "error": None, "sanitized_macros": n}
    res = _fail(f"{err} | retry after de-backslashing {n} unknown macros: {retry_err}")
    res["sanitized_macros"] = n
    return res
