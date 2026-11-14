"""
# mg5.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import os, json, datetime, subprocess
import ast, re
from typing import Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from tools.jobs.progress import report_progress

import subprocess

SCHEMA_VERSION = "evtjsonl-1.0"

# ====================================================================== #
# ========================= Helper functions =========================== #
# ====================================================================== #

def _resolve_mg5_exec(mg5_root: str) -> str:
    """
    Given MG5 root dir (e.g. /path/to/MG5_aMC_vX.X.X)
    return absolute path to mg5_aMC executable inside bin/.

    Raise ValueError if not found or not executable.
    """
    if mg5_root is None or mg5_root.strip() == "":
        raise ValueError("mg5_path was not provided")

    mg5_root_abs = os.path.abspath(mg5_root)
    exe_path = os.path.join(mg5_root_abs, "bin", "mg5_aMC")

    if not os.path.isfile(exe_path):
        raise ValueError(f"mg5_aMC not found at {exe_path}")
    if not os.access(exe_path, os.X_OK):
        raise ValueError(f"mg5_aMC is not executable at {exe_path}")

    return exe_path

def _rel_if_inside(base_dir: str, path: str) -> str:
    """Return a relative path if the given path is inside the base directory."""
    base_dir = os.path.abspath(base_dir)
    path_abs = os.path.abspath(path)
    if path_abs.startswith(base_dir + os.sep) or path_abs == base_dir:
        return os.path.relpath(path_abs, base_dir)
    return path_abs  # fallback absolute

def _find_lhe_in_tree(root: str) -> Optional[str]:
    """Search for unweighted_events.lhe file in the directory tree."""
    for r, dnames, fnames in os.walk(root):
        for fn in fnames:
            if fn.startswith("unweighted_events.lhe"):
                return os.path.join(r, fn)
    return None

def _grep_lhe_from_log(log_path: str) -> Optional[str]:
    """Search for LHE file paths in the MadGraph log file."""
    try:
        with open(log_path, "r", encoding="utf-8") as fp:
            for line in fp:
                # Crude heuristic: look for "unweighted_events.lhe".
                if "unweighted_events.lhe" in line:
                    # Extract token ending with ".lhe" or ".lhe.gz".
                    parts = line.strip().split()
                    for tok in parts:
                        if "unweighted_events.lhe" in tok:
                            # Strip trailing punctuation.
                            cand = tok.rstrip(",'\")")
                            if os.path.exists(cand):
                                return cand
    except Exception:
        pass
    return None


_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
_SUMMARY_RE = re.compile(r"=== Results Summary for run:\s*(\S+)")
_WIDTH_RE = re.compile(r"Width\s*:\s*" + _NUM + r"\s*\+-\s*" + _NUM + r"\s*GeV")
_XSEC_RE = re.compile(r"Cross-section\s*:\s*" + _NUM + r"\s*\+-\s*" + _NUM + r"\s*pb")
_NEVENTS_RE = re.compile(r"Nb of events\s*:\s*(\d+)")
_INDEX_HTML_RE = re.compile(r"more information in\s+(\S+?)/index\.html")


def _parse_run_summary(log_path: str) -> Optional[dict]:
    """Return the *final* MG5 results summary in the log.

    MG5 prints a "=== Results Summary for run: ... ===" block after the
    survey (``Nb of events : 0``) and again after refine/combine with the
    final width or cross section and the number of unweighted events.  The
    last block is the one that describes the stored run; the survey block
    must never be reported as the result.

    Returns a dict with ``width_gev``/``width_gev_error`` or
    ``cross_section_pb``/``cross_section_pb_error``, plus ``nevents`` and
    ``run_name`` when printed, or None if the log holds no summary.
    """
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fp:
            content = _ANSI_RE.sub("", fp.read())
    except Exception:
        return None

    # Split into summary blocks; fall back to the whole log when MG5 printed
    # bare "Width :" lines without the header.
    heads = list(_SUMMARY_RE.finditer(content))
    if heads:
        blocks = [(m.group(1), content[m.end():(heads[i + 1].start() if i + 1 < len(heads) else len(content))])
                  for i, m in enumerate(heads)]
    else:
        blocks = [(None, content)]

    for run_name, block in reversed(blocks):
        info: dict = {}
        widths = _WIDTH_RE.findall(block)
        xsecs = _XSEC_RE.findall(block)
        if widths:
            info["width_gev"] = float(widths[-1][0])
            info["width_gev_error"] = float(widths[-1][1])
        elif xsecs:
            info["cross_section_pb"] = float(xsecs[-1][0])
            info["cross_section_pb_error"] = float(xsecs[-1][1])
        else:
            continue
        nev = _NEVENTS_RE.findall(block)
        if nev:
            info["nevents"] = int(nev[-1])
        if run_name:
            info["run_name"] = run_name
        return info
    return None


def _parse_width_from_log(log_path: str) -> Optional[dict[str, float]]:
    """Width/cross section (and error) from the final results summary."""
    info = _parse_run_summary(log_path)
    if info is None:
        return None
    return {k: v for k, v in info.items()
            if k in ("width_gev", "width_gev_error", "cross_section_pb", "cross_section_pb_error")}


def _process_dirs(card_text: str, outdir: str, log_path: str) -> list:
    """Candidate MG5 process directories for this run.

    ``output``/``launch`` may name a directory outside data_dir (an absolute
    path, or one relative to the run cwd), so searching data_dir alone misses
    the LHE.  Collect the directories the card and the log name.
    """
    dirs = []
    for line in card_text.splitlines():
        toks = line.split("#", 1)[0].split()
        if len(toks) >= 2 and toks[0] in ("output", "launch"):
            args = [t for t in toks[1:] if not t.startswith("-")]
            if toks[0] == "output" and args and args[0] in (
                    "madevent", "standalone", "standalone_cpp", "madweight", "pythia8", "aloha", "matchbox"):
                args = args[1:]
            if args:
                dirs.append(args[0])
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fp:
            for m in _INDEX_HTML_RE.finditer(_ANSI_RE.sub("", fp.read())):
                dirs.append(m.group(1))
    except Exception:
        pass
    out = []
    for d in dirs:
        full = os.path.abspath(d if os.path.isabs(d) else os.path.join(outdir, d))
        if os.path.isdir(full) and full not in out:
            out.append(full)
    return out


def _locate_lhe(outdir: str, card_text: str, log_path: str,
                run_name: Optional[str] = None) -> Optional[str]:
    """Find the run's unweighted_events.lhe[.gz], preferring Events/<run_name>."""
    roots = _process_dirs(card_text, outdir, log_path)
    if run_name:
        for root in roots + [outdir]:
            for r, _d, fnames in os.walk(root):
                if os.path.basename(r) == run_name and os.path.basename(os.path.dirname(r)) == "Events":
                    for fn in sorted(fnames):
                        if fn.startswith("unweighted_events.lhe"):
                            return os.path.join(r, fn)
    for root in [outdir] + roots:
        cand = _find_lhe_in_tree(root)
        if cand:
            return cand
    return _grep_lhe_from_log(log_path)


def _safe_join(base_dir: str, rel_path: str) -> Optional[str]:
    """
    Resolve rel_path against base_dir.
    Return absolute path if and only if it is inside base_dir.
    Otherwise return None.
    """
    if rel_path is None:
        return None
    rel_norm = rel_path.lstrip(os.sep)
    full = os.path.abspath(os.path.join(base_dir, rel_norm))
    # Allow base_dir itself as edge case.
    if not (full == base_dir or full.startswith(base_dir + os.sep)):
        return None
    return full

def _now_utc_iso() -> str:
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

def _detect_scan_runs(events_dir: str) -> list:
    """
    Detect if MadGraph performed a parameter scan by looking for run_XX subdirectories.

    Returns:
        List of run directory paths (e.g., ["run_01", "run_02", "run_03"]) if scan detected.
        Empty list if no scan (single run or no runs found).
    """
    if not os.path.isdir(events_dir):
        return []

    # Look for run_XX directories
    run_dirs = []
    for entry in os.listdir(events_dir):
        entry_path = os.path.join(events_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith("run_"):
            run_dirs.append(entry)

    # Sort to ensure consistent ordering (run_01, run_02, ...)
    run_dirs.sort()

    # Only consider it a scan if we have multiple runs
    return run_dirs if len(run_dirs) > 1 else []

def _parse_scan_summary(scan_file: str) -> dict:
    """
    Parse the scan_run_0[1-3].txt file generated by MadGraph parameter scans.

    Expected format:
        #run_name            mass#9000005         cross                error
        run_01               1.000000e+03         1.018011e-01         9.330523e-05
        run_02               1.500000e+03         8.741885e-03         7.437245e-06

    Returns:
        Dictionary mapping run_id to {param_name: value, cross_section: ..., error: ...}
    """
    scan_info = {}

    try:
        with open(scan_file, "r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            return scan_info

        # Parse header to get parameter names
        header = lines[0].strip()
        if not header.startswith("#"):
            return scan_info

        # Extract column names from header
        # Format: #run_name  param1  param2  cross  error
        cols = header[1:].split()  # Remove leading #

        # Find indices for known columns
        param_cols = []
        cross_idx = None
        error_idx = None

        for i, col in enumerate(cols):
            if col == "run_name":
                continue
            elif col == "cross":
                cross_idx = i
            elif col == "error":
                error_idx = i
            else:
                # Assume it's a parameter column
                param_cols.append((i, col))

        # Parse data lines
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < len(cols):
                continue

            run_id = parts[0]
            run_data = {}

            # Extract parameter values
            for idx, param_name in param_cols:
                try:
                    run_data[param_name] = float(parts[idx])
                except (ValueError, IndexError):
                    pass

            # Extract cross-section and error
            if cross_idx is not None:
                try:
                    run_data["cross_section"] = float(parts[cross_idx])
                except (ValueError, IndexError):
                    pass

            if error_idx is not None:
                try:
                    run_data["cross_section_error"] = float(parts[error_idx])
                except (ValueError, IndexError):
                    pass

            scan_info[run_id] = run_data

    except Exception:
        # If parsing fails, return empty dict (non-fatal)
        pass

    return scan_info

def _find_all_lhe_files(events_dir: str, run_dirs: list) -> dict:
    """
    Find LHE files for each run in a scan.

    Args:
        events_dir: Path to the Events directory
        run_dirs: List of run directory names (e.g., ["run_01", "run_02"])

    Returns:
        Dictionary mapping run_id to LHE file path
    """
    lhe_files = {}

    for run_id in run_dirs:
        run_path = os.path.join(events_dir, run_id)
        if not os.path.isdir(run_path):
            continue

        # Look for unweighted_events.lhe or .lhe.gz in this run directory
        for fname in os.listdir(run_path):
            if fname.startswith("unweighted_events.lhe"):
                lhe_files[run_id] = os.path.join(run_path, fname)
                break

    return lhe_files

def _count_scan_points(card_text: str) -> Optional[int]:
    """Number of runs a card's `scan:` settings make MG5 do, or None.

    Parameters with a bare `scan:` each add a dimension (MG5 loops over all
    combinations); parameters sharing a `scanN:` tag move together and add
    one. Values MG5 would have to evaluate (e.g. `range(...)`) give None:
    the count is only for the progress bar, so unknown beats wrong.
    """
    dims = {}
    for i, m in enumerate(re.finditer(r"^\s*set\s+\S+\s+(scan\d*):\s*(\[.*\])",
                                      card_text, re.MULTILINE)):
        tag, values = m.group(1), m.group(2)
        try:
            n = len(ast.literal_eval(values))
        except (ValueError, SyntaxError):
            return None
        dims[tag if tag != "scan" else f"scan#{i}"] = n
    if not dims:
        return None
    total = 1
    for n in dims.values():
        total *= n
    return total


class _MG5Progress:
    """Turns mg5_run.log, as MG5 writes it, into progress reports.

    MG5 reports no event count while it runs, so what is available is the
    stage (building the process, survey, refine, combining events; MG5
    compiles inside the survey, so that is not a stage), the scheduler line
    `Idle: a,  Running: b,  Completed: c` for the current stage, and, for a
    scan, which run (point) it is on. A scan reports points done out of the
    total; a single run reports the current stage's jobs done out of queued.
    """

    _STAGES = (
        ("Generating Helas calls", "building process"),
        ("Running Survey", "survey"),
        ("Refine results", "refine"),
        ("Combining Events", "combining events"),
        ("Running MadSpin", "MadSpin"),
        ("Running Systematics", "systematics"),
    )
    _RUN = re.compile(r"Generating\s+\d+\s+events with run name\s+(\S+)")
    _SCHED = re.compile(r"Idle:\s*(\d+),\s*Running:\s*(\d+),\s*Completed:\s*(\d+)")

    def __init__(self, log_path: str, n_points: Optional[int]):
        self.log_path = log_path
        self.n_points = n_points if n_points and n_points > 1 else None
        self.offset = 0
        self.partial = b""
        self.run = None
        self.stage = None
        self.sched = None
        self.runs = []
        self.last = None

    def _feed(self, line: str) -> None:
        m = self._RUN.search(line)
        if m:
            self.run, self.stage, self.sched = m.group(1), None, None
            if self.run not in self.runs:
                self.runs.append(self.run)
            return
        m = self._SCHED.search(line)
        if m:
            idle, running, completed = (int(x) for x in m.groups())
            self.sched = (completed, idle + running + completed)
            return
        for marker, stage in self._STAGES:
            if marker in line:
                self.stage, self.sched = stage, None
                return

    def poll(self, tool, finished: bool = False) -> None:
        """Report what the log shows so far; ``finished`` once MG5 has exited.

        A scan point counts as done when the next one starts (MG5 prints a
        results summary after the survey as well as at the end, so that line
        cannot mark it), and the last one when MG5 exits.
        """
        if getattr(tool, "_stream_callback", None) is None:
            return
        try:
            with open(self.log_path, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
                self.offset = fh.tell()
        except OSError:
            return
        lines = (self.partial + chunk).split(b"\n")
        self.partial = lines.pop()
        for raw in lines:
            self._feed(raw.decode("utf-8", errors="replace"))

        parts = [p for p in (self.run, self.stage) if p]
        if self.n_points:
            if self.sched:
                parts.append(f"{self.sched[0]}/{self.sched[1]} jobs")
            done = len(self.runs) if finished else max(0, len(self.runs) - 1)
            report = (min(done, self.n_points), self.n_points, "scan points")
        elif self.sched:
            report = (self.sched[0], self.sched[1], "integration jobs")
        else:
            report = (None, None, "")
        state = report + (" · ".join(parts),)
        if state != self.last and (state[3] or state[0] is not None):
            self.last = state
            report_progress(tool, done=state[0], total=state[1],
                            unit=state[2], message=state[3])


# Seconds between looks at mg5_run.log while MG5 runs.
_PROGRESS_POLL_SECONDS = 2.0


def _normalize_masses(masses) -> dict:
    """Coerce a {pdg: GeV} mapping (or [[pdg, gev], ...]) into {int: float}.

    Junk entries (non-numeric ids/values) are dropped rather than raising, so a
    caller's loosely-typed spectrum still yields the well-formed pairs.
    """
    out = {}
    items = masses.items() if isinstance(masses, dict) else (
        masses if isinstance(masses, (list, tuple)) else [])
    for pair in items:
        try:
            k, v = pair
            out[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _edit_mg5_card(card_text: str, *, ufo_path: Optional[str] = None,
                   output_name: Optional[str] = None, nevents: Optional[int] = None,
                   seed: Optional[int] = None, masses: Optional[dict] = None) -> str:
    """
    Edit MG5 command card by replacing specific lines.

    Args:
        card_text: Original card content
        ufo_path: If provided, replaces the 'import model' line
        output_name: If provided, replaces the 'output' line
        nevents: If provided, replaces the 'set nevents' line
        seed: If provided, replaces the 'set iseed' line
        masses: If provided, a {pdg_id: mass_gev} mapping injected as
            'set mass <pdg> <gev>' in the run-card region (after 'launch'),
            replacing any existing 'set mass <pdg>' for the same id. A
            structured spectrum channel so benchmark masses survive card
            regeneration instead of being dropped as free text.

    Returns:
        Modified card text
    """
    mass_map = _normalize_masses(masses) if masses else {}
    replaced_pdgs = set()

    lines = card_text.splitlines()
    output_lines = []

    for line in lines:
        stripped = line.strip()

        # Replace 'import model' line
        if ufo_path is not None and stripped.startswith("import model"):
            output_lines.append(f"import model {ufo_path}")
            continue

        # Replace 'output' line
        if output_name is not None and stripped.startswith("output "):
            output_lines.append(f"output {output_name}")
            continue

        # Replace 'set nevents' line
        if nevents is not None and stripped.startswith("set nevents"):
            output_lines.append(f"set nevents {nevents}")
            continue

        # Replace 'set iseed' line
        if seed is not None and stripped.startswith("set iseed"):
            output_lines.append(f"set iseed {seed}")
            continue

        # Replace an existing 'set mass <pdg>' for a requested id (idempotent),
        # so re-editing a card doesn't stack duplicate spectrum lines.
        if mass_map and stripped.startswith("set mass "):
            parts = stripped.split()
            if len(parts) >= 3 and parts[2].lstrip("-").isdigit():
                pdg = int(parts[2])
                if pdg in mass_map:
                    output_lines.append(f"set mass {pdg} {mass_map[pdg]:g}")
                    replaced_pdgs.add(pdg)
                    continue

        # Keep original line
        output_lines.append(line)

    # Inject any not-yet-present masses into the run-card region, right after
    # 'launch' (where 'set' commands configure the run) so a benchmark spectrum
    # is applied robustly rather than left as free text.
    remaining = [(pdg, gev) for pdg, gev in mass_map.items()
                 if pdg not in replaced_pdgs]
    if remaining:
        injected_lines = []
        did_inject = False
        for line in output_lines:
            injected_lines.append(line)
            if not did_inject and line.strip().startswith("launch"):
                for pdg, gev in remaining:
                    injected_lines.append(f"set mass {pdg} {gev:g}")
                did_inject = True
        if not did_inject:  # no 'launch' line — append at the end
            for pdg, gev in remaining:
                injected_lines.append(f"set mass {pdg} {gev:g}")
        output_lines = injected_lines

    result = "\n".join(output_lines)

    # Preserve trailing newline if original had one
    if card_text.endswith("\n"):
        result += "\n"

    return result


_SETUP_COMMANDS = ("import ", "generate ", "add process ", "define ", "output ", "set group_subprocesses", "set gauge")


def _relaunch_card(card_text: str, process_dir: str, *, nevents: Optional[int] = None,
                   seed: Optional[int] = None) -> str:
    """A card that relaunches an existing process directory.

    Keeps the run-setting lines of the template (those after its 'launch'
    line, or every non-setup line when it has none) and drops model import,
    process generation and output, which the compiled directory already has.
    """
    lines = card_text.splitlines()
    starts = [i for i, l in enumerate(lines) if l.strip().startswith("launch")]
    if starts:
        body = lines[starts[0] + 1:]
    else:
        body = [l for l in lines if l.strip() and not l.strip().startswith(_SETUP_COMMANDS)]
    body = [l for l in body if not l.strip().startswith("launch")]
    text = _edit_mg5_card("\n".join(body) + "\n", nevents=nevents, seed=seed)
    stripped = [l.strip() for l in text.splitlines()]
    if nevents is not None and not any(l.startswith("set nevents") for l in stripped):
        text += f"set nevents {nevents}\n"
    if seed is not None and not any(l.startswith("set iseed") for l in stripped):
        text += f"set iseed {seed}\n"
    return f"launch {process_dir}\n" + text


def _default_nb_core() -> int:
    """Cores an MG5 run may use unless its card says otherwise.

    MG5 defaults to every core, so a few runs at once oversubscribe the
    machine many times over. HEPTAPOD_MG5_NB_CORE overrides the default of 4.
    """
    try:
        return max(1, int(os.environ.get("HEPTAPOD_MG5_NB_CORE", "4")))
    except ValueError:
        return 4


def _cap_cores(card_text: str) -> str:
    """Prepend 'set nb_core N' to a card that sets neither nb_core nor run_mode."""
    stripped = [l.strip() for l in card_text.splitlines()]
    if any(l.startswith(("set nb_core", "set run_mode")) for l in stripped):
        return card_text
    return f"set nb_core {_default_nb_core()}\n" + card_text


# ====================================================================== #
# ========================= MG5_aMC@NLO tool =========================== #
# ====================================================================== #

class MadGraphFromRunCardTool(BaseTool):
    """
    Run MadGraph5_aMC@NLO to generate parton-level events given a .mg5 run card template.

    Inputs (runtime):
      - data_dir: output directory (relative to base_directory) where this run will live
      - command_card: path (relative to base_directory) to MG5 command card template file
      - ufo_path: (optional) path to UFO model directory. If provided, replaces 'import model' line.
      - output_name: (optional) process output directory name. If provided, replaces 'output' command.
      - nevents: (optional) number of events to generate. If provided, replaces 'set nevents' command.
      - seed: (optional) random seed. If provided, replaces 'set iseed' command.
      - process_dir: (optional) existing process directory to relaunch with the
        card's run settings, skipping import/generate/output.

    State:
      - mg5_path: absolute path to MG5_aMC install dir (contains bin/mg5_aMC)
      - base_directory: sandbox root for all file operations

    Behavior:
      1. Read the template command card.
      2. Apply runtime substitutions (UFO path, output name, nevents, seed) if provided.
      3. Write the modified card to data_dir.
      4. Launch mg5_aMC to create process directory and generate events.
      5. Return summary with pointer to produced LHE path.
    """
    # --------------------------- Runtime fields --------------------------- #

    data_dir: str = RuntimeField(description="Relative output directory for dataset, e.g. 'data/mg_run001'")
    command_card: str = RuntimeField(description="Relative path (inside base_directory) to MG5 command card template")
    ufo_path: Optional[str] = RuntimeField(default=None, description="Path to UFO model directory (replaces 'import model' line)")
    output_name: Optional[str] = RuntimeField(default=None, description="Process output directory name (replaces 'output' command)")
    nevents: Optional[int] = RuntimeField(default=None, description="Number of events to generate (replaces 'set nevents')")
    seed: Optional[int] = RuntimeField(default=None, description="Random seed (replaces 'set iseed')")
    process_dir: Optional[str] = RuntimeField(default=None, description=(
        "Relaunch this existing MG5 process directory (relative to base_directory) instead of building one: "
        "the card's run settings (lines after 'launch', e.g. 'set ...') are applied and import/generate/output "
        "are skipped. Use it for another run of a compiled process: more events, another seed, changed cuts."))

    # --------------------------------------------------------------------- #

    # ---------------------------- State fields ---------------------------- #

    mg5_path: str = StateField(description="Absolute path to top-level MG5_aMC install dir containing bin/mg5_aMC")
    base_directory: str = StateField(description="Base sandbox root for all file ops")

    # ---------------------------------------------------------------------- #

    def _setup(self):
        """Set up the tool by validating base directory."""
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.isdir(self.base_directory):
            raise ValueError(f"Base directory does not exist or is not a directory: {self.base_directory}")

    def _run(self) -> str:
        """Run the MadGraph event generation from the command card."""
        # Validate base directory.
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Path Error", reason=str(e))

        outdir = _safe_join(self.base_directory, self.data_dir)
        cmd_src = _safe_join(self.base_directory, self.command_card)

        if outdir is None or cmd_src is None:
            return self.format_error(
                error="Access Denied",
                reason="data_dir or command_card escapes base_directory",
                suggestion="Use sandbox-relative paths"
            )

        if not os.path.exists(cmd_src):
            return self.format_error(
                error="File Not Found",
                reason="MG5 command card not found",
                context=self.command_card
            )

        os.makedirs(outdir, exist_ok=True)

        cmd_local_path = os.path.join(outdir, "mg5_command.txt")
        log_local_path = os.path.join(outdir, "mg5_run.log")

        # Read template card and apply runtime substitutions
        try:
            with open(cmd_src, "r", encoding="utf-8") as f:
                card_text = f.read()
        except Exception as e:
            return self.format_error(
                error="Read Error",
                reason=str(e),
                context=self.command_card
            )

        # Apply edits based on runtime fields
        try:
            # Resolve UFO path if provided (make absolute relative to base_directory)
            ufo_abs = None
            if self.ufo_path is not None:
                ufo_abs = _safe_join(self.base_directory, self.ufo_path)
                if ufo_abs is None:
                    return self.format_error(
                        error="Access Denied",
                        reason="ufo_path escapes base_directory",
                        context=self.ufo_path
                    )
                if not os.path.exists(ufo_abs):
                    return self.format_error(
                        error="Path Not Found",
                        reason="UFO directory not found",
                        context=self.ufo_path
                    )

            if self.process_dir is not None:
                process_abs = _safe_join(self.base_directory, self.process_dir)
                if process_abs is None:
                    return self.format_error(
                        error="Access Denied",
                        reason="process_dir escapes base_directory",
                        context=self.process_dir
                    )
                if not os.path.isfile(os.path.join(process_abs, "bin", "generate_events")):
                    return self.format_error(
                        error="Path Not Found",
                        reason="process_dir is not an MG5 process directory (no bin/generate_events)",
                        context=self.process_dir
                    )
                card_text = _relaunch_card(card_text, process_abs, nevents=self.nevents, seed=self.seed)
            else:
                card_text = _edit_mg5_card(
                    card_text,
                    ufo_path=ufo_abs,
                    output_name=self.output_name,
                    nevents=self.nevents,
                    seed=self.seed
                )
        except Exception as e:
            return self.format_error(
                error="Card Edit Error",
                reason=str(e),
                suggestion="Check template card format"
            )

        card_text = _cap_cores(card_text)

        # Write modified card
        try:
            with open(cmd_local_path, "w", encoding="utf-8") as f:
                f.write(card_text)
        except Exception as e:
            return self.format_error(
                error="File Write Error",
                reason=str(e),
                suggestion="Check permissions"
            )
        
        # Resolve executable.
        try:
            mg5_exec = _resolve_mg5_exec(self.mg5_path)
        except Exception as e:
            return self.format_error(
                error="Dependency Missing",
                reason=str(e),
                suggestion="Pass a valid mg5_path pointing to MG5_aMC_v*/"
            )

        # Generate events, reading the log as it grows for progress reports.
        watcher = _MG5Progress(log_local_path, _count_scan_points(card_text))
        with open(log_local_path, "w", encoding="utf-8") as logfp:
            proc = subprocess.Popen(
                [mg5_exec, cmd_local_path],
                cwd=outdir,               # run inside outdir for reproducibility
                stdout=logfp,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                while True:
                    try:
                        proc.wait(timeout=_PROGRESS_POLL_SECONDS)
                        break
                    except subprocess.TimeoutExpired:
                        watcher.poll(self)
            except BaseException:
                # Interrupted, cancelled or timed out: MG5 must not outlive us.
                proc.kill()
                proc.wait()
                raise
        watcher.poll(self, finished=proc.returncode == 0)
        summary = _parse_run_summary(log_local_path) or {}
        run_name = summary.pop("run_name", None)
        if proc.returncode != 0:
            # MG5 may exit nonzero on nevents=0 (ZeroDivisionError) but the
            # width is still computed and logged.  Check for it before failing.
            if summary:
                result = {
                    "status": "ok",
                    "scan_detected": False,
                    "data_dir": os.path.relpath(outdir, self.base_directory),
                    "log_file": os.path.relpath(log_local_path, self.base_directory),
                    "lhe_file": None,
                    "note": "mg5 exited nonzero but width/cross-section was computed (likely nevents=0)",
                }
                result.update(summary)
                return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

            return self.format_error(
                error="MadGraph Error",
                reason="mg5_aMC exited nonzero",
                suggestion="Inspect mg5_run.log for details",
                context=f"log={os.path.relpath(log_local_path, self.base_directory)}"
            )

        # Locate the LHE: data_dir, then the process directory named by the
        # card's output/launch line or the log (it may sit outside data_dir).
        lhe_candidate = _locate_lhe(outdir, card_text, log_local_path, run_name)

        if lhe_candidate is None:
            # No LHE file — but for nevents=0 runs, the width/cross-section
            # is still printed in the log.  Extract it if available.
            if summary:
                result = {
                    "status": "ok",
                    "scan_detected": False,
                    "data_dir": os.path.relpath(outdir, self.base_directory),
                    "log_file": os.path.relpath(log_local_path, self.base_directory),
                    "lhe_file": None,
                }
                result.update(summary)
                if summary.get("nevents", 0) == 0:
                    result["note"] = "nevents=0: no events generated, width/cross-section extracted from log"
                else:
                    result["status"] = "warning"
                    result["note"] = (
                        f"MG5 reports {summary['nevents']} events but unweighted_events.lhe "
                        f"was not found; check the process directory named in the card/log"
                    )
                return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

            launched = any(ln.strip().split()[:1] == ["launch"] for ln in card_text.splitlines())
            if not launched:
                return self.format_error(
                    error="No Launch",
                    reason="The card never launches a run, so there are no events and no "
                           "integration to report (compute_widths alone writes a param card, "
                           "not events)",
                    suggestion="This tool reports a launched run: for a partial width, "
                               "generate the channel (e.g. 'generate n1 > ve z z'), output and launch it"
                )
            return self.format_error(
                error="Output Missing",
                reason="Could not locate unweighted_events.lhe",
                suggestion="Check mg5_run.log and confirm 'output' + 'launch' ran"
            )

        # Detect if this was a parameter scan by looking for the Events directory
        # The LHE file path structure helps us find it
        # Typical path: outdir/process_name/Events/run_01/unweighted_events.lhe.gz
        events_dir = None
        lhe_parent = os.path.dirname(lhe_candidate)

        # Walk up the directory tree to find "Events" directory
        for _ in range(3):  # Check up to 3 levels up
            parent_name = os.path.basename(lhe_parent)
            if parent_name == "Events":
                events_dir = lhe_parent
                break
            lhe_parent = os.path.dirname(lhe_parent)

        # Detect scan runs
        scan_detected = False
        scan_runs = []
        scan_summary_file = None

        if events_dir:
            scan_runs = _detect_scan_runs(events_dir)
            # Several run_XX directories are a scan only when the card asked for
            # one: a relaunched process directory keeps its earlier runs.
            scan_detected = len(scan_runs) > 0 and re.search(r"\bscan\d*:", card_text) is not None

            if scan_detected:
                # Look for scan summary file (e.g., scan_run_0[1-3].txt)
                for fname in os.listdir(events_dir):
                    if fname.startswith("scan_run_") and fname.endswith(".txt"):
                        scan_summary_file = os.path.join(events_dir, fname)
                        break

        # Build manifest.
        manifest = {
            "schema": SCHEMA_VERSION,
            "created_utc": _now_utc_iso(),
            "inputs": {
                "command_card": "mg5_command.txt",
                "mg5_root": os.path.abspath(self.mg5_path),
            },
            "outputs": {
                "lhe_path": _rel_if_inside(self.base_directory, lhe_candidate),
                "log_path": os.path.relpath(log_local_path, self.base_directory),
            },
            "tool": "MadGraphFromCardTool",
            "scan_detected": scan_detected,
        }
        if summary and not scan_detected:
            manifest["results"] = dict(summary)

        if scan_detected:
            manifest["outputs"]["scan_runs"] = scan_runs
            if scan_summary_file:
                manifest["outputs"]["scan_summary_file"] = _rel_if_inside(self.base_directory, scan_summary_file)

        manifest_path = os.path.join(outdir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)

        # Build result
        result = {
            "status": "ok",
            "scan_detected": scan_detected,
            "data_dir": os.path.relpath(outdir, self.base_directory),
            "lhe_file": _rel_if_inside(self.base_directory, lhe_candidate),  # First run for backward compat
            "manifest_json": os.path.relpath(manifest_path, self.base_directory),
            "log_file": os.path.relpath(log_local_path, self.base_directory),
        }
        # Final (post-refine) width / cross section, error and event count.
        # For a scan the per-point values are in runs[] instead.
        if not scan_detected:
            result.update(summary)

        # If scan detected, add detailed scan information
        if scan_detected and events_dir:
            # Find all LHE files for each run
            lhe_files = _find_all_lhe_files(events_dir, scan_runs)

            # Parse scan summary if available
            scan_info = {}
            if scan_summary_file and os.path.exists(scan_summary_file):
                scan_info = _parse_scan_summary(scan_summary_file)
                result["scan_summary_file"] = _rel_if_inside(self.base_directory, scan_summary_file)

            # Build runs array
            runs = []
            for run_id in scan_runs:
                run_data = {
                    "run_id": run_id,
                    "lhe_file": _rel_if_inside(self.base_directory, lhe_files.get(run_id, ""))
                }

                # Add scan parameters and cross-section if available
                if run_id in scan_info:
                    info = scan_info[run_id]

                    # Separate scan parameters from cross-section info
                    scan_params = {}
                    for key, value in info.items():
                        if key not in ["cross_section", "cross_section_error"]:
                            scan_params[key] = value

                    if scan_params:
                        run_data["scan_params"] = scan_params

                    if "cross_section" in info:
                        run_data["cross_section"] = info["cross_section"]
                    if "cross_section_error" in info:
                        run_data["cross_section_error"] = info["cross_section_error"]

                runs.append(run_data)

            result["n_runs"] = len(runs)
            result["runs"] = runs

        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

# --- log-parsing helpers ported from the June-22 snapshot mg5.py ---
# Required by tools/mg5/validate_process.py (ValidateProcessTool).
import re  # noqa: E402  (idempotent; used by the classifiers below)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _distill_mg5_log(log_path: str, max_lines: int = 8, max_chars: int = 700) -> str:
    """Reduce an mg5_run.log to the lines that explain a failure.

    MG5's own diagnostics are highly actionable but buried in hundreds of
    banner/config lines (and wrapped in ANSI color). The lines that matter
    name the bad UFO model, the rejected sub-command, the missing card, etc.,
    e.g.:
        Error detected in sub-command import model ...
        ... directory is not a valid UFO model:  couplings.py is missing
        command not executed: generate p p > go go
    Surfacing these (instead of a generic "inspect the log") lets a weak agent
    fix the run card directly rather than retry blindly. Task-agnostic: the
    markers are MG5's, not any one process's."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fp:
            raw = fp.read()
    except Exception:
        return ""
    lines = [_ANSI_RE.sub("", l).rstrip() for l in raw.splitlines()]
    lines = [l for l in lines if l.strip()]
    # Root-cause lines (what actually went wrong) take priority over the
    # cascade of "command not executed" notices MG5 prints afterwards; keep one
    # of the latter so the agent sees which step was aborted.
    cause_re = re.compile(
        r"error detected|not a valid|is missing|no such|not found|invalid|"
        r"traceback|exception|interrupt|aborted|fail|no matching|unknown|"
        # PDF / run-card config faults (lhapdf, beam/PDF label mismatches) that
        # otherwise leave only a generic "Output Missing".
        r"not compatible|lhapdf|pdlabel|lpp\d|lhaid|no pdf|pdf set",
        re.I)
    cause = [l for l in lines if cause_re.search(l)]
    notexec = [l for l in lines if re.search(r"command not executed", l, re.I)]
    chosen = (cause[-max_lines:] + notexec[:1]) if cause else (lines[-max_lines:])
    return "\n".join(chosen)[-max_chars:].strip()


# MG5's two dominant, currently-opaque generate/process failures. Both name the
# real cause on their own diagnostic line; surfacing that line structured (which
# particle, why, what to do) lets a weak agent fix its process command instead of
# retrying the same rejected generate. Task-agnostic: these are MG5's markers and
# the guidance is generic generator usage, never a process's physics values.
_NO_PARTICLE_RE = re.compile(r"No particle\s+(\S+)", re.I)
_NO_DIAGRAM_RE = re.compile(r"NoDiagramException|no diagrams? for (?:this )?process", re.I)
# A run-card 'set' issued out of place (e.g. top-level 'set nevents'): MG5 emits
# "InvalidCmd: Possible options for set are [...]".
_BAD_SET_RE = re.compile(r"Possible options for set are|InvalidCmd.*\bset\b", re.I)


def _classify_mg5_log(log_path: str) -> Optional[dict]:
    """Detect a known MG5 generate/process failure class in the log.

    Returns a dict with keys ``error``, ``reason``, ``suggestion`` suitable for
    splatting into ``format_error`` when one of the two dominant failure classes
    is recognized, else None (caller falls back to the generic distilled log).
        1. ``No particle <X>`` — MG5 rejected an unknown particle name in the
           generate/process line.
        2. ``NoDiagramException`` / 'No diagrams for process' — the process is
           empty in this model.
    Task-agnostic: keyed on MG5's own markers, names the offending particle, and
    points at generic recovery (display particles / decay in the shower)."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fp:
            raw = _ANSI_RE.sub("", fp.read())
    except Exception:
        return None

    # 1. Unknown particle name(s) in the generate/process line.
    bad = []
    for m in _NO_PARTICLE_RE.finditer(raw):
        tok = m.group(1).strip().strip(".,:;'\"")
        if tok and tok not in bad:
            bad.append(tok)
    if bad:
        names = ", ".join(f"'{b}'" for b in bad)
        return {
            "error": "Unknown Particle in Process",
            "reason": f"MadGraph rejected particle name(s) {names} in the "
                      f"generate/process line — not defined in the model.",
            "suggestion": (
                f"Particle(s) {names} are not defined in this model. Check the "
                "model's particle names (in an MG5 session: `display particles`). "
                "Two common causes: (a) wrong name — e.g. squarks are "
                "ul/ur/dl/dr/cl/cr/sl/sr/t1/t2/b1/b2 (not 'sq'), the jet "
                "multiparticle is 'j'; (b) the particle is genuinely absent from "
                "this model — if it is a decay product not in the model (e.g. a "
                "gravitino in MSSM_SLHA2), do NOT force it into the generate "
                "command: produce the event down to the state the model contains "
                "and perform that decay in the parton shower (Pythia) via the "
                "spectrum's decay table."),
        }

    # 2. No diagrams found for the requested process.
    if _NO_DIAGRAM_RE.search(raw):
        return {
            "error": "No Diagrams for Process",
            "reason": "MadGraph found no diagrams for this process in the model.",
            "suggestion": (
                "MadGraph found no diagrams for this process. Causes: a particle "
                "or coupling in the process is not in this model, the coupling "
                "orders forbid it, or the decay is not allowed. Verify each "
                "particle exists in the model and the process is physical; do not "
                "retry the same process repeatedly."),
        }

    # 3. A 'set' command issued where MG5 does not accept it (e.g. a top-level
    #    'set nevents'/'set iseed' written as an MG5 command before/without
    #    launch). MG5 responds with "InvalidCmd: Possible options for set are
    #    [...]". The fix is to let the tool place run-card sets after launch.
    if _BAD_SET_RE.search(raw):
        return {
            "error": "Invalid 'set' Command",
            "reason": "MadGraph rejected a 'set' command (InvalidCmd: Possible "
                      "options for set are [...]) — a run-card parameter such as "
                      "nevents/iseed was issued where MG5 does not accept it.",
            "suggestion": (
                "Don't hand-write `set nevents`/`set iseed` as MG5 commands — "
                "pass the tool's `nevents`/`seed` arguments instead; the tool "
                "places them in the run-card region for you. Any `set` for a "
                "run-card parameter must come AFTER `launch`, not at the "
                "MG5-command level."),
        }

    return None

