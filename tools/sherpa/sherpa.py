
import json, os, datetime, math
from pathlib import Path
from typing import Any, Dict, Optional, List

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from tqdm import tqdm
import numpy as np

SCHEMA_VERSION = "evtjsonl-1.0"

# ====================================================================== #
# =========================== Helper functions ========================= #
# ====================================================================== #

def _require_sherpa() -> Any:
    """Ensure Sherpa is available, or raise ImportError."""
    try:
        import Sherpa as sherpa3
        return sherpa3
    except Exception as e:
        raise ImportError("Sherpa is not available, install to use this tool (e.g. `pip install sherpa-mc`).") from e


def _event_to_dict(blobs: Any, finals_only: bool, full_history: bool) -> Dict[str, Any]:
    """Convert a Sherpa event to dict with fixed keys."""
    parts = []
    # print("  Weight ",blobs.Weight());
    # print("  Trials ",blobs.GetFirst(1)["Trials"]);
    # Process each particle in the event.
    pid = 0
    for i in range(0,blobs.size()):
        for j in range(0,blobs[i].NOutP()):
            p=blobs[i].OutPart(j)
            status = int(not p.HasDecBlob())
            is_final = bool(status == 1)
            if finals_only and not is_final:
                continue
            mom = p.Momentum()
            # Skip non-final particles if requested
            # Build particle event record.
            rec = {
                "i": pid,
                "id": int(p.Flav().Kfcode()),
                "status": status,
                "px": float(mom[1]),
                "py": float(mom[2]),
                "pz": float(mom[3]),
                "E": float(mom[0]),
                "m": float(mom.Mass()),
            }
            # Include full history if requested.
            if full_history:
                for key in ("mother1", "mother2", "daughter1", "daughter2"):
                    if hasattr(p, key):
                        rec[key] = int(getattr(p, key)())
            parts.append(rec)
            pid += 1
    return {"n": len(parts), "particles": parts}


def _edit_sherpa_card(card_text: str,*,ufo_path: Optional[str] = None) -> str:
    """
    Prepare Sherpa run by converting UFO and editing run card.

    Parameters:
        card_text: Original .yaml file content
        ufo_path: If provided, runs the ufo model converter

    Returns:
        Modified card text with replacements applied
    """
    lines = card_text.splitlines()
    output_lines = []

    for line in lines:
        # Keep original line
        output_lines.append(line)

    import os
    if ufo_path != None:
        model_name = os.path.basename(ufo_path.strip('/'))
        output_lines.append('MODEL: '+model_name)
        output_lines.append('UFO_PARAM_CARD: param_'+model_name+'.dat')
        if not os.path.isdir(ufo_path+'.sherpa'):
            import sys, Sherpa
            sherpa_base_dir = os.path.dirname(Sherpa.__file__)
            sys.path.append(sherpa_base_dir)
            local_argv = []
            local_argv.append('--root_dir')
            local_argv.append(sherpa_base_dir+'/share/SHERPA-MC')
            local_argv.append('--install_dir')
            if os.path.isdir(sherpa_base_dir+'/lib/SHERPA-MC'):
                local_argv.append(sherpa_base_dir+'/lib/SHERPA-MC')
            if os.path.isdir(sherpa_base_dir+'/lib64/SHERPA-MC'):
                local_argv.append(sherpa_base_dir+'/lib64/SHERPA-MC')
            local_argv.append('--auto_convert')
            local_argv.append(ufo_path)
            import ufo_interface.parser
            ufo_interface.parser.main(local_argv)

    result = "\n".join(output_lines)
    # Preserve trailing newline if original had one
    if card_text.endswith("\n"):
        result += "\n"
    return result

# ====================================================================== #
# =================== Sherpa event generation tool ===================== #
# ====================================================================== #

class SherpaFromRunCardTool(BaseTool):
    """
    Generate hadron-level events using Sherpa3 driven by a provided .yaml run card.

    Inputs (runtime):
      - data_dir: relative output directory under base_directory where run artifacts will be stored
      - cmnd_path: relative path to a valid Sherpa3 .yaml configuration file
      - ufo_path: If provided, defines the location of the UFO model
      - n_events: number of events to generate
      - seed: optional integer random seed (if omitted, Sherpa's internal RNG is used)
      - finals_only: if True, record only final-state particles (status==1)
      - full_history: if True, include intermediate particles and mother indices in the JSONL output
      - base_directory: sandbox root for all file operations

    Behavior:
      1. Copy the provided .yaml file into the output directory for provenance.
      2. Initialize Sherpa3 with optional fixed seed.
      3. Generate n_events events; for each successful event, record a structured event record.
      4. Write results to events.jsonl using schema "evtjsonl-1.0", with one JSON object per event.
      5. Report a summary containing output paths, number of accepted events, and cross-section data.

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
        - sherpa3mc import errors
        - initialization or event generation failures
        - file read/write or permission issues

    Notes:
      - All paths must remain inside base_directory for safety.
      - Output files are always written under data_dir, including run.yaml and events.jsonl.
      - The JSONL schema is compatible with downstream jet-clustering and analysis tools.
    """
    # --------------------------- Runtime fields --------------------------- #
    data_dir: str = RuntimeField(description="Relative output directory for dataset, e.g. 'data/run001'")
    cmnd_path: str = RuntimeField(description="Relative path to Sherpa .yaml run card template")
    ufo_path: str = RuntimeField(description="UFO model directory")
    n_events: int = RuntimeField(description="Number of events to generate")
    seed: Optional[int] = RuntimeField(default=None, description="Random seed (optional)")
    finals_only: bool = RuntimeField(default=True, description="Keep only final-state particles if true")
    full_history: bool = RuntimeField(default=False, description="Include lineage indices if true")
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
        """Run Sherpa event generation and return JSON summary."""
        # Check required parameters.
        for key in ("data_dir", "cmnd_path", "n_events"):
            if getattr(self, key, None) in (None, ""):
                return self.format_error(
                    error="Missing Parameter",
                    reason=f"{key} is required",
                    suggestion="Provide required runtime fields"
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

        # Create output directory.
        os.makedirs(outdir, exist_ok=True)
        cmnd_dst = os.path.join(outdir, "Sherpa.yaml")

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
        # Note: n_events doesn't go in the Sherpa card; it's controlled by the loop.
        # The seed parameter passed to sherpa.readString() below takes precedence.
        # Pass absolute path to _edit_sherpa_card so Sherpa can find the LHE file.
        card_text = _edit_sherpa_card(card_text, ufo_path=self.ufo_path)

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

        # Check for sherpa3mc dependency.
        try:
            sherpa3 = _require_sherpa()
        except Exception as e:
            return self.format_error(
                error="Dependency Missing",
                reason=str(e),
                suggestion="Install sherpa3mc in the current runtime"
            )

        # Initialize Sherpa.
        try:
            local_argv = ['Sherpa']
            local_argv.append("-O0")
            local_argv.append(f"-f{str(cmnd_dst)}")
            local_argv.append(f"-e{int(self.n_events)}")
            if self.seed is not None:
                local_argv.append(f"-R{int(self.seed)}")
            sherpa = sherpa3.Sherpa(len(local_argv),local_argv)
            try:
                sherpa.InitializeTheRun()
                sherpa.InitializeTheEventHandler()
            except sherpa3.SherpaException as e:
                sherpa.SummarizeRun()
                del sherpa
                del sherpa3
                return self.format_error(
                    error="Sherpa Init Failed",
                    reason="Initialization returned false",
                    context="Check run.cmnd settings",
                    suggestion="Validate beams, processes, and energy"
                )
        except Exception as e:
            exit(1)
            return self.format_error(
                error="Sherpa Error",
                reason=str(e),
                suggestion="Check Sherpa installation and run card syntax"
            )
        # Check for events file.
        events_path = os.path.join(outdir, "events.jsonl")
        # Initialize counters.
        accepted = 0
        failed = 0
        sumw = 0.
        sumw2 = 0.
        trials = 0.

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
                for ev_id in tqdm(range(int(self.n_events)), desc="Generating events", unit="evt", ncols=80):
                    sherpa.GenerateOneEvent()
                    blobs=sherpa.GetBlobList()
                    sumw += blobs.Weight()
                    sumw2 += blobs.Weight()**2
                    trials += blobs.GetFirst(1)["Trials"]
                    edict = _event_to_dict(blobs, self.finals_only, self.full_history)
                    row = {**schema_meta, "event_id": ev_id, "data": edict}
                    fp.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
                    accepted += 1
        except Exception as e:
            sherpa.SummarizeRun()
            del sherpa
            del sherpa3
            return self.format_error(
                error="Write Error",
                reason=str(e),
                context=f"path={events_path}",
                suggestion="Verify disk space and permissions"
            )
        sherpa.SummarizeRun()
        del sherpa
        del sherpa3

        # Extract cross-section information.
        xsec = {}
        xsec["sigmaGen_mb"] = sumw/trials
        xsec["sigmaErr_mb"] = np.sqrt((sumw2/trials-(sumw/trials)**2)/(trials-1))

        # Create manifest file.
        manifest = {
            "schema": SCHEMA_VERSION,
            "created_utc": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
            "inputs": {
                "run_card": "Sherpa.yaml",
                "n_events_requested": int(self.n_events),
                "seed": int(self.seed) if self.seed is not None else None,
                "finals_only": bool(self.finals_only),
                "full_history": bool(self.full_history),
                **({"ufo_path": self.ufo_path} if self.ufo_path else {}),
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

