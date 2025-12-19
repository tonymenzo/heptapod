"""
# pythia.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

import json, os, datetime, math
from pathlib import Path
from typing import Any, Dict, Optional, List

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from tqdm import tqdm
import numpy as np
import sys

SCHEMA_VERSION = "evtjsonl-1.0"

# Configure tqdm to prevent multiple line printing
TQDM_CONFIG = {
    'file': sys.stderr,
    'ncols': 80,
    'leave': True,
    'dynamic_ncols': False,
    'mininterval': 0.1,  # Default update interval
    'ascii': False       # Use Unicode characters for better appearance
}

# ====================================================================== #
# =========================== Helper functions ========================= #
# ====================================================================== #

def _require_pythia() -> Any:
    """Ensure pythia8mc is available, or raise ImportError."""
    try:
        import pythia8mc as pythia8
        return pythia8
    except Exception as e:
        raise ImportError("pythia8mc is not available, install to use this tool (e.g. `pip install pythia8mc`).") from e


def _event_to_dict(evt: Any, finals_only: bool, full_history: bool) -> Dict[str, Any]:
    """Convert a Pythia event to dict with fixed keys."""
    parts = []
    n = evt.size()
    # Process each particle in the event.
    for i in range(n):
        p = evt[i]
        status = int(p.status())
        is_final = bool(p.isFinal() if hasattr(p, "isFinal") else (status > 0))
        # Skip non-final particles if requested
        if finals_only and not is_final:
            continue
        # Build particle event record.
        rec = {
            "i": i,
            "id": int(p.id()),
            "status": status,
            "px": float(p.px()),
            "py": float(p.py()),
            "pz": float(p.pz()),
            "E": float(p.e() if hasattr(p, "e") else p.eCalc() if hasattr(p, "eCalc") else float(p.E())),
            "m": float(p.m() if hasattr(p, "m") else p.mCalc() if hasattr(p, "mCalc") else 0.0),
        }
        # Include full history if requested.
        if full_history:
            for key in ("mother1", "mother2", "daughter1", "daughter2"):
                if hasattr(p, key):
                    rec[key] = int(getattr(p, key)())
        parts.append(rec)
    return {"n": len(parts), "particles": parts}


def _edit_pythia_card(
    card_text: str,
    *,
    lhe_path: Optional[str] = None
) -> str:
    """
    Edit Pythia command card by replacing specific lines.

    Parameters:
        card_text: Original .cmnd file content
        lhe_path: Path to LHE file (replaces line starting with 'Beams:LHEF =')

    Returns:
        Modified card text with replacements applied
    """
    lines = card_text.splitlines()
    output_lines = []

    for line in lines:
        stripped = line.strip()

        # Replace line starting with 'Beams:LHEF' if lhe_path is provided
        # Handle various formats: "Beams:LHEF = value", "Beams:LHEF=value", "Beams:LHEF      = value"
        if lhe_path is not None and stripped.split()[0] == "Beams:LHEF" if stripped.split() else False:
            output_lines.append(f"Beams:LHEF = {lhe_path}")
            continue
        # Keep original line
        output_lines.append(line)

    result = "\n".join(output_lines)
    # Preserve trailing newline if original had one
    if card_text.endswith("\n"):
        result += "\n"
    return result

# ====================================================================== #
# =================== Pythia event generation tool ===================== #
# ====================================================================== #

class PythiaFromRunCardTool(BaseTool):
    """
    Generate hadron-level events using Pythia8 driven by a provided .cmnd run card.

    Inputs (runtime):
      - data_dir: relative output directory under base_directory where run artifacts will be stored
      - cmnd_path: relative path to a valid Pythia8 .cmnd configuration file
      - n_events: number of events to generate
      - seed: optional integer random seed (if omitted, Pythia's internal RNG is used)
      - finals_only: if True, record only final-state particles (status==1)
      - full_history: if True, include intermediate particles and mother indices in the JSONL output
      - shower_lhe: if True, use Pythia for showering/hadronization of LHE events (requires lhe_path)
      - lhe_path: path to LHE file (required when shower_lhe=True, optional otherwise)
      - base_directory: sandbox root for all file operations

    Behavior:
      1. Copy the provided .cmnd file into the output directory for provenance.
      2. If lhe_path is provided, automatically edit the run card to inject the LHE path.
      3. Initialize Pythia8 with optional fixed seed.
      4. Generate n_events events; for each successful event, record a structured event record.
      5. Write results to events.jsonl using schema "evtjsonl-1.0", with one JSON object per event.
      6. Report a summary containing output paths, number of accepted events, and cross-section data.

    LHE Showering Mode:
      When shower_lhe=True, Pythia is used for showering/hadronization of pre-generated
      LHE events rather than standalone event generation. In this mode:
        - lhe_path is REQUIRED and specifies the path to the input LHE file
        - The tool automatically edits the run card to inject the LHE path (replaces 'Beams:LHEF' line)
        - Pythia adds parton showering and hadronization to matrix-element level events

    Output (JSON):
      {
        "status": "ok",
        "data_dir": "<relative output directory>",
        "events_jsonl": "<relative path to events.jsonl>",
        "n_events": <int>,
        "accepted": <int>,
        "failed": <int>,
        "cross_section": {
          "sigmaGen_mb": <float>,
          "sigmaErr_mb": <float>,
          "weightSum": <float>
        }
      }

    Errors:
      Returns BaseTool.format_error JSON on any failure including:
        - invalid or missing cmnd_path
        - pythia8mc import errors
        - initialization or event generation failures
        - file read/write or permission issues

    Notes:
      - All paths must remain inside base_directory for safety.
      - Output files are always written under data_dir, including run.cmnd and events.jsonl.
      - The JSONL schema is compatible with downstream jet-clustering and analysis tools.
    """
    # --------------------------- Runtime fields --------------------------- #
    data_dir: str = RuntimeField(description="Relative output directory for dataset, e.g. 'data/run001'")
    cmnd_path: str = RuntimeField(description="Relative path to Pythia .cmnd run card template")
    n_events: int = RuntimeField(description="Number of events to generate")
    seed: Optional[int] = RuntimeField(default=None, description="Random seed (optional)")
    finals_only: bool = RuntimeField(default=True, description="Keep only final-state particles if true")
    full_history: bool = RuntimeField(default=False, description="Include lineage indices if true")
    shower_lhe: bool = RuntimeField(default=False, description="If True, use Pythia for showering/hadronization of LHE events (requires lhe_path)")
    lhe_path: Optional[str] = RuntimeField(default=None, description="Path to LHE file for showering/hadronization (required when shower_lhe=True)")
    # ---------------------------------------------------------------------- #

    # ---------------------------- State fields ---------------------------- #
    base_directory: str = StateField(description="Base directory for safe path resolution")
    # ---------------------------------------------------------------------- #

    def _setup(self):
        """Setup base directory and validate it exists."""
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.exists(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _safe_path(self, rel: str) -> Optional[str]:
        """Ensures that the path is within the allowed base directory."""
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        if not full.startswith(self.base_directory):
            return None
        return full

    def _run(self) -> str:
        """Run Pythia event generation and return JSON summary."""
        # Check required parameters.
        for key in ("data_dir", "cmnd_path", "n_events"):
            if getattr(self, key, None) in (None, ""):
                return self.format_error(
                    error="Missing Parameter",
                    reason=f"{key} is required",
                    suggestion="Provide required runtime fields"
                )

        # Validate shower_lhe mode requirements.
        if self.shower_lhe and not self.lhe_path:
            return self.format_error(
                error="Missing Parameter",
                reason="lhe_path is required when shower_lhe=True",
                suggestion="Provide lhe_path for LHE showering mode or set shower_lhe=False"
            )

        outdir = self._safe_path(self.data_dir)
        cmnd_src = self._safe_path(self.cmnd_path)

        # Check for safe paths.
        if not outdir or not cmnd_src:
            return self.format_error(
                error="Access Denied",
                reason="Path escapes base_directory",
                context=f"data_dir={self.data_dir}, cmnd_path={self.cmnd_path}",
                suggestion="Use paths inside the allowed base directory"
            )

        # Check if run card exists.
        if not os.path.exists(cmnd_src):
            return self.format_error(
                error="File Not Found",
                reason="Run card does not exist",
                context=f"path={self.cmnd_path}",
                suggestion="Provide a valid .cmnd file path"
            )

        # Validate and resolve LHE path if provided.
        lhe_path_abs = None
        if self.lhe_path:
            lhe_path_abs = self._safe_path(self.lhe_path)
            if not lhe_path_abs:
                return self.format_error(
                    error="Access Denied",
                    reason="lhe_path escapes base_directory",
                    context=f"lhe_path={self.lhe_path}",
                    suggestion="Use paths inside the allowed base directory"
                )
            if not os.path.exists(lhe_path_abs):
                return self.format_error(
                    error="File Not Found",
                    reason="LHE file does not exist",
                    context=f"path={self.lhe_path}",
                    suggestion="Provide a valid LHE file path"
                )

        # Create output directory.
        os.makedirs(outdir, exist_ok=True)
        cmnd_dst = os.path.join(outdir, "run.cmnd")

        # Read template card, apply runtime edits if provided, then write to output.
        try:
            with open(cmnd_src, "r", encoding="utf-8") as f:
                card_text = f.read()
        except Exception as e:
            return self.format_error(
                error="Read Error",
                reason=str(e),
                context=f"path={self.cmnd_path}",
                suggestion="Verify file exists and is readable"
            )

        # Apply card edits if any RuntimeFields are provided.
        # Note: n_events doesn't go in the Pythia card; it's controlled by the loop.
        # The seed parameter passed to pythia.readString() below takes precedence.
        # Pass absolute path to _edit_pythia_card so Pythia can find the LHE file.
        card_text = _edit_pythia_card(card_text, lhe_path=lhe_path_abs)

        # Write modified card to output directory.
        try:
            with open(cmnd_dst, "w", encoding="utf-8") as f:
                f.write(card_text)
        except Exception as e:
            return self.format_error(
                error="Write Error",
                reason=str(e),
                context=f"dst={self.data_dir}/run.cmnd",
                suggestion="Verify permissions and disk space"
            )

        # Check for pythia8mc dependency.
        try:
            pythia8 = _require_pythia()
        except Exception as e:
            return self.format_error(
                error="Dependency Missing",
                reason=str(e),
                suggestion="Install pythia8mc in the current runtime"
            )

        # Initialize Pythia.
        try:
            pythia = pythia8.Pythia("", printBanner=False)
            # Suppress all Pythia output to prevent interference with tqdm
            pythia.readString("Print:quiet = on")
            pythia.readString("Init:showProcesses = off")
            pythia.readString("Init:showMultipartonInteractions = off")
            pythia.readString("Init:showChangedSettings = off")
            pythia.readString("Init:showChangedParticleData = off")
            pythia.readString("Next:numberShowInfo = 0")
            pythia.readString("Next:numberShowProcess = 0")
            pythia.readString("Next:numberShowEvent = 0")
            if self.seed is not None:
                pythia.readString("Random:setSeed = on")
                pythia.readString(f"Random:seed = {int(self.seed)}")
            pythia.readFile(str(cmnd_dst))
            if not pythia.init():
                return self.format_error(
                    error="Pythia Init Failed",
                    reason="Initialization returned false",
                    context="Check run.cmnd settings",
                    suggestion="Validate beams, processes, and energy"
                )
        except Exception as e:
            return self.format_error(
                error="Pythia Error",
                reason=str(e),
                suggestion="Check Pythia installation and run card syntax"
            )

        # Check for events file.
        events_path = os.path.join(outdir, "events.jsonl")
        # Initialize counters.
        accepted = 0
        failed = 0

        # Generate events and write to JSONL.
        try:
            # Pre-compute schema metadata (optimization: avoid recreating constants in loop)
            schema_meta = {
                "schema": SCHEMA_VERSION,
                "finals_only": bool(self.finals_only),
                "full_history": bool(self.full_history),
            }

            # Use larger buffer (256KB) for better I/O performance
            with open(events_path, "w", encoding="utf-8", buffering=262144) as fp:
                for ev_id in tqdm(range(int(self.n_events)), desc="Generating events", unit="evt", **TQDM_CONFIG):
                    if not pythia.next():
                        failed += 1
                        continue
                    accepted += 1
                    edict = _event_to_dict(pythia.event, self.finals_only, self.full_history)
                    row = {**schema_meta, "event_id": ev_id, "data": edict}
                    fp.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        except Exception as e:
            return self.format_error(
                error="Write Error",
                reason=str(e),
                context=f"path={events_path}",
                suggestion="Verify disk space and permissions"
            )

        # Extract cross-section information.
        xsec = {}
        try:
            info = pythia.infoPython() if hasattr(pythia, "infoPython") else pythia.info()
            if hasattr(info, "sigmaGen"):
                xsec["sigmaGen_mb"] = float(info.sigmaGen())
            if hasattr(info, "sigmaErr"):
                xsec["sigmaErr_mb"] = float(info.sigmaErr())
            if hasattr(info, "weightSum"):
                xsec["weightSum"] = float(info.weightSum())
        except Exception:
            pass

        # Create manifest file.
        manifest = {
            "schema": SCHEMA_VERSION,
            "created_utc": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
            "inputs": {
                "run_card": "run.cmnd",
                "n_events_requested": int(self.n_events),
                "seed": int(self.seed) if self.seed is not None else None,
                "finals_only": bool(self.finals_only),
                "full_history": bool(self.full_history),
                "shower_lhe": bool(self.shower_lhe),
                **({"lhe_path": self.lhe_path} if self.lhe_path else {}),
            },
            "outputs": {
                "events_jsonl": "events.jsonl",
                "n_events_written": int(accepted),
                "n_events_failed": int(failed),
                **({"xsec": xsec} if xsec else {}),
            },
        }
        manifest_path = os.path.join(outdir, "manifest.json")
        
        # Write manifest file.
        try:
            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2)
        except Exception as e:
            return self.format_error(
                error="Write Error",
                reason=str(e),
                context=f"path={manifest_path}",
                suggestion="Verify disk space and permissions"
            )

        # Create result object.
        result = {
            "status": "ok",
            "data_dir": os.path.relpath(outdir, self.base_directory),
            "events_jsonl": os.path.relpath(events_path, self.base_directory),
            "manifest_json": os.path.relpath(manifest_path, self.base_directory),
            "accepted": int(accepted),
            "failed": int(failed),
        }
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

# ..................................................................... #
# ..................... SlowJet clustering tool ....................... #
# ..................................................................... #

import os

class FdSilence:
    def __init__(self, *fds):
        """Context manager to silence specified file descriptors (e.g., STDOUT, STDERR)."""
        self.fds = fds
        self.saved = []

    def __enter__(self):
        # open null once
        self.null = os.open(os.devnull, os.O_WRONLY)
        for fd in self.fds:
            # dup original fd so we can restore later
            saved_fd = os.dup(fd)
            self.saved.append((fd, saved_fd))
            # redirect fd -> /dev/null
            os.dup2(self.null, fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore each fd
        for (fd, saved_fd) in self.saved:
            os.dup2(saved_fd, fd)
            os.close(saved_fd)
        os.close(self.null)

STDIN, STDOUT, STDERR = 0, 1, 2

_ALGO_TO_POWER = {
    "kt": 1,        # kT
    "ca": 0,        # Cambridge/Aachen
    "antikt": -1,   # anti-kT
}

class JetClusterSlowJetTool(BaseTool):
    """
    Cluster jets with Pythia8.SlowJet for a single event.

    Input (choose ONE):
      - jsonl_path + event_index   (expects schema from PythiaFromRunCardTool)
      - npy_path  + event_index    (zero-padded array shape [N_ev, N_max, 5] with [px,py,pz,E,pid])

    Params:
      - algorithm: 'antikt' | 'kt' | 'ca'
      - R: radius parameter (e.g. 0.4)
      - ptmin: minimum jet pT (GeV)
      - etamax: |eta| max for constituents
      - mass_option: 1 = E-scheme (typical), see Pythia8 doc
      - select: 2 selects |eta| < etamax and pT > ptmin (typical)
    Output:
      JSON with jets [{px, py, pz, E, m, pT, eta, phi, n_const, constituents:[indices]}]
      Indices refer to the event-particle indices used internally here.
    """

    # Tool arguments
    jsonl_path: Optional[str] = RuntimeField(default=None, description="Path to events.jsonl")
    npy_path: Optional[str]   = RuntimeField(default=None, description="Path to padded events .npy")
    output_path: Optional[str] = RuntimeField(default=None, description="Path to save output .jsonl (if cluster_all=True)")
    event_index: Optional[int] = RuntimeField(default=None, description="0-based event index to cluster (not needed if cluster_all=True)")
    cluster_all: bool = RuntimeField(default=False, description="Cluster all events in dataset")

    # SlowJet arguments
    algorithm: str   = RuntimeField(default="antikt", description="antikt | kt | ca")
    R: float         = RuntimeField(default=0.4, description="Jet radius")
    ptmin: float     = RuntimeField(default=20.0, description="Min jet pT [GeV]")
    etamax: float    = RuntimeField(default=5.0, description="Max |eta| for constituents")
    mass_option: int = RuntimeField(default=1, description="Recombination scheme (SlowJet massOption)")

    # Sandbox root
    base_directory: str = StateField(default=".", description="Base directory for safe path resolution")

    def _get_pythia8(self):
        """Get or create a shared Pythia8 module reference."""
        if not hasattr(self, "_pythia8_module"):
            self._pythia8_module = _require_pythia()
        return self._pythia8_module

    def _setup(self):
        """Setup base directory and validate it exists."""
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.isdir(self.base_directory):
            raise ValueError(f"Base directory does not exist or is not a directory: {self.base_directory}")

    def _safe_path(self, rel: Optional[str]) -> Optional[str]:
        """
        Resolve rel against base_directory.
        Return absolute path if and only if it is inside base_directory.
        Otherwise return None.
        """
        if not rel:
            return None
        # If user passed an absolute path, interpret it relative to base_directory anyway
        # so that "/tmp/x" cannot escape.
        rel_norm = rel.lstrip(os.sep)
        full = os.path.abspath(os.path.join(self.base_directory, rel_norm))
        if not full.startswith(self.base_directory + os.sep) and full != self.base_directory:
            return None
        return full

    def _load_event_from_jsonl(self, src: str, idx: int) -> List[Dict[str, float]]:
        """Loads a single event from a JSONL file."""
        # Returns list of particles with keys: id, px, py, pz, E, (optional m), (optional i).
        with open(src, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == idx:
                    ev = json.loads(line)
                    parts = ev["data"]["particles"]
                    # Ensure ordering by original index if provided.
                    parts = sorted(parts, key=lambda p: p.get("i", 0))
                    return parts
        raise IndexError(f"event_index {idx} out of range for {src}")

    def _load_event_from_npy(self, src: str, idx: int) -> List[Dict[str, float]]:
        """Loads a single event from a padded NumPy .npy file."""
        arr = np.load(src, allow_pickle=False, mmap_mode=None)
        if idx < 0 or idx >= arr.shape[0]:
            raise IndexError(f"event_index {idx} out of range for {src}")
        rows = arr[idx]
        parts = []
        for j, row in enumerate(rows):
            px, py, pz, E, pid = row.tolist()
            # Skip pad row.
            if px == 0 and py == 0 and pz == 0 and E == 0 and pid == 0:
                continue
            # Infer mass if not present.
            m2 = max(E*E - (px*px + py*py + pz*pz), 0.0)
            parts.append({"id": int(pid), "px": float(px), "py": float(py), "pz": float(pz), "E": float(E), "m": math.sqrt(m2), "i": j})
        return parts

    def _build_pythia_event(self, pythia8: Any, particles: List[Dict[str, float]]) -> Any:
        """
        Build a minimal Pythia8.Event with final-state particles only.
        Uses Event.append signature available in Pythia8; if unavailable, error out.
        """
        if not hasattr(pythia8, "Event"):
            raise RuntimeError("pythia8mc binding does not expose Event class.")
        
        # Initialize empty event.
        evt = pythia8.Event()

        # Clear existing event content.
        try:
            evt.reset()
        except Exception:
            # Some bindings use clear().
            if hasattr(evt, "clear"):
                evt.clear()
    
        # Add particles.
        for p in particles:
            px, py, pz, E = p["px"], p["py"], p["pz"], p["E"]
            m = float(p.get("m", 0.0))
            pid = int(p.get("id", 0))
            
            # Event.append(id, status, mother1, mother2, daughter1, daughter2, px, py, pz, E, m).
            if hasattr(evt, "append"):
                evt.append(pid, 1, 0, 0, px, py, pz, E, m)
            else:
                raise RuntimeError("pythia8mc Event.append not available in this binding.")
        return evt

    def _cluster(self, event_particles: List[Dict[str, float]], pythia8) -> Dict[str, Any]:
        """Perform jet clustering on the provided event particles."""
        #pythia8 = _require_pythia()
        power = _ALGO_TO_POWER.get(self.algorithm.lower())
        if power is None:
            raise ValueError(f"Unsupported algorithm '{self.algorithm}'. Use one of {list(_ALGO_TO_POWER)}")

        # Build Pythia event.
        evt = self._build_pythia_event(pythia8, event_particles)

        # Initialize SlowJet. Try 6-arg signature first, then 4-arg fallback.
        sj = None
        # Based on the way Pythia events are built, select must be hardcoded to 1.
        SELECT = 1
        try:
            sj = pythia8.SlowJet(
                int(power),
                float(self.R),
                float(self.ptmin),
                float(self.etamax),   # etaMax
                SELECT,               # select
                int(self.mass_option) # massOption
            )
        except Exception:
            sj = pythia8.SlowJet(int(power), float(self.R), float(self.ptmin), float(self.etamax()))

        # Quietly (supress FastJet banner) cluster with SlowJet.
        with FdSilence(STDOUT, STDERR):
            sj.analyze(evt)

        # Collect jet multiplicity.
        n_jets = int(sj.sizeJet())
        jets = []

        # Extract jet constituents.
        for j in range(n_jets):
            cons = []
                # Most recent bindings: constituents(j) -> list[int]
            if hasattr(sj, "constituents") and callable(getattr(sj, "constituents")):
                try:
                    result = sj.constituents(j)
                    # If result is already iterable of ints.
                    if isinstance(result, (list, tuple)):
                        cons = [int(ix) for ix in result]
                except TypeError:
                    # Older SWIG form: constituents(j, PVectorInt)
                    if hasattr(pythia8, "PVectorInt"):
                        tmp = pythia8.PVectorInt()
                        sj.constituents(j, tmp)
                        cons = [int(ix) for ix in tmp]
        
            # Legacy bindings: sizeConstituents/constituent
            if not cons and hasattr(sj, "sizeConstituents") and hasattr(sj, "constituent"):
                ncons = int(sj.sizeConstituents(j))
                cons = [int(sj.constituent(j, k)) for k in range(ncons)]
        
            # Optional index shift (only if confirmed 1-based).
            if cons and min(cons) >= 1 and 0 not in cons:
                cons = [c - 1 for c in cons]

            constituents = []
            for idx in cons:
                p_i = evt[idx]
                cx = float(p_i.px())
                cy = float(p_i.py())
                cz = float(p_i.pz())
                ce = float(p_i.e())
                if hasattr(p_i, "mCalc"):
                    cmass = float(p_i.mCalc())
                else:
                    cmass = float(p_i.m())

                constituents.append({
                    "event_index": idx,
                    "px": cx,
                    "py": cy,
                    "pz": cz,
                    "E":  ce,
                    "m":  cmass,
                })
        
            jets.append({
                "index": j,
                "px":  float(sj.p(j).px()),
                "py":  float(sj.p(j).py()),
                "pz":  float(sj.p(j).pz()),
                "E":   float(sj.p(j).e()),
                "m":   float(sj.m(j)),
                "pT":  float(sj.pT(j)),
                "eta": float(sj.y(j)),
                "phi": float(sj.phi(j)),
                "n_const": len(cons),
                "constituents": constituents,
            })

        result = {
            "algorithm": self.algorithm.lower(),
            "R": float(self.R),
            "ptmin": float(self.ptmin),
            "etamax": float(self.etamax),
            "mass_option": int(self.mass_option),
            "data": {
                "n_jets": n_jets,
                "jets": jets,
            }
        }
        return result

    def _run(self) -> str:
        """Run jet clustering and return JSON summary."""
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Path Error", reason=str(e))

        has_jsonl_arg = bool(self.jsonl_path)
        has_npy_arg   = bool(self.npy_path)

        src_jsonl = self._safe_path(self.jsonl_path) if has_jsonl_arg else None
        src_npy   = self._safe_path(self.npy_path)   if has_npy_arg   else None

        if has_jsonl_arg and src_jsonl is None:
            return self.format_error(error="Path Error", reason="jsonl_path is outside allowed base_directory")
        if has_npy_arg and src_npy is None:
            return self.format_error(error="Path Error", reason="npy_path is outside allowed base_directory")
        if not has_jsonl_arg and not has_npy_arg:
            return self.format_error(error="Invalid Parameters", reason="No input source provided")

        use_jsonl = has_jsonl_arg
        src = src_jsonl if use_jsonl else src_npy
        p = Path(src)
        if not p.exists():
            return self.format_error(error="File Not Found", reason=f"{src} not found")

        # Cluster either one event or all events
        n_events = 0
        results = []  # Only used for single-event mode

        try:
            if self.cluster_all:
                # Validate and open output file for streaming
                if not self.output_path:
                    return self.format_error(
                        error="Missing Parameter",
                        reason="output_path is required when cluster_all=True",
                        suggestion="Provide output_path to save clustered jets"
                    )

                outpath = self._safe_path(self.output_path)
                if outpath is None:
                    return self.format_error(
                        error="Path Error",
                        reason="output_path is outside allowed base_directory"
                    )

                # Create output directory
                os.makedirs(os.path.dirname(outpath), exist_ok=True)

                # Stream results directly to file
                pythia8 = self._get_pythia8()

                with open(outpath, "w", encoding="utf-8") as outfile:
                    if use_jsonl:
                        # Count total lines (events)
                        with open(src, "r", encoding="utf-8") as f:
                            total_events = sum(1 for _ in f)

                        with open(src, "r", encoding="utf-8") as f:
                            for idx, line in tqdm(enumerate(f), total=total_events, desc="Clustering events", **TQDM_CONFIG):
                                # Parse the line directly instead of re-reading the file
                                try:
                                    ev = json.loads(line)
                                    parts = ev["data"]["particles"]
                                    # Ensure ordering by original index if provided
                                    parts = sorted(parts, key=lambda p: p.get("i", 0))
                                except (json.JSONDecodeError, KeyError) as e:
                                    # Skip malformed events
                                    continue

                                if not parts:
                                    continue
                                result = self._cluster(parts, pythia8)
                                result["event_index"] = idx
                                # Write immediately to file
                                outfile.write(json.dumps(result, separators=(",", ":"), ensure_ascii=False) + "\n")
                                n_events += 1
                    else:
                        arr = np.load(src, allow_pickle=False, mmap_mode=None)
                        for idx in tqdm(range(arr.shape[0]), desc="Clustering events", **TQDM_CONFIG):
                            parts = self._load_event_from_npy(src, idx)
                            if not parts:
                                continue
                            result = self._cluster(parts, pythia8)
                            result["event_index"] = idx
                            # Write immediately to file
                            outfile.write(json.dumps(result, separators=(",", ":"), ensure_ascii=False) + "\n")
                            n_events += 1
            else:
                # Single event mode - keep in memory for return
                if self.event_index is None:
                    return self.format_error(
                        error="Missing Parameter",
                        reason="event_index is required when cluster_all=False",
                        suggestion="Provide event_index to cluster a single event or set cluster_all=True"
                    )
                idx = int(self.event_index)
                if idx < 0:
                    raise ValueError("event_index must be non-negative")
                parts = (
                    self._load_event_from_jsonl(src, idx)
                    if use_jsonl
                    else self._load_event_from_npy(src, idx)
                )
                if not parts:
                    return self.format_error(error="Empty Event", reason="No particles at event_index")
                pythia8 = self._get_pythia8()
                result = self._cluster(parts, pythia8)
                result["event_index"] = idx
                results = [result]
                n_events = 1
        except Exception as e:
            return self.format_error(error="Clustering Error", reason=str(e))

        # Return summary
        if self.cluster_all:
            # Streaming mode - return summary only
            return json.dumps(
                {
                    "status": "ok",
                    "n_events": n_events,
                    "output_file": os.path.relpath(outpath, self.base_directory)
                },
                separators=(",", ":"),
                ensure_ascii=False
            )
        else:
            # Single event mode - return full result
            return json.dumps(
                {"status": "ok", "n_events": n_events, "results": results},
                separators=(",", ":"),
                ensure_ascii=False
            )