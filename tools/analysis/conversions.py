"""
# conversions.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json, os
from typing import Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

import numpy as np
from tqdm import tqdm
import sys

SCHEMA_VERSION = "evtjsonl-1.0"

# Configure tqdm to prevent multiple line printing
TQDM_CONFIG = {
    'file': sys.stderr,
    'ncols': 80,
    'leave': True,
    'dynamic_ncols': False
}

# ====================================================================== #
# ====================== JSONL \to Numpy tool ========================== #
# ====================================================================== #

class EventJSONLToNumpyTool(BaseTool):
    """
    Convert a JSONL event dataset into a zero-padded NumPy array.

    Each event is represented as an array of shape (N_particles, 5),
    with columns [px, py, pz, E, pid].
    The tool zero-pads shorter events to match the longest event.
    """
    # --------------------------- Runtime fields --------------------------- #
    jsonl_path: str = RuntimeField(description="Relative path to events.jsonl file")
    output_path: str = RuntimeField(description="Relative path to save .npy file (e.g. 'data/run/events.npy')")
    # ---------------------------------------------------------------------- #

    # ---------------------------- State fields ---------------------------- #
    base_directory: str = StateField(default=".", description="Base directory for safe paths")
    # ---------------------------------------------------------------------- #

    def _setup(self):
        """Setup base directory and validate it exists."""
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.exists(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _safe_path(self, rel: str) -> Optional[str]:
        """Ensures that the path is within the allowed base directory."""
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    def _run(self) -> str:
        """Run the conversion from JSONL to NumPy array."""
        src = self._safe_path(self.jsonl_path)
        dst = self._safe_path(self.output_path)
        
        # Check for safe paths.
        if not src or not dst:
            return self.format_error(
                error="Access Denied",
                reason="jsonl_path or output_path escapes base_directory",
                suggestion="Use relative paths inside base_directory"
            )
        
        # Check if source JSONL exists.
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason="JSONL file not found",
                context=f"path={self.jsonl_path}",
                suggestion="Provide a valid events.jsonl path"
            )
        
        # Read events from JSONL.
        try:
            with open(src, "r") as f:
                events = [json.loads(line) for line in f]
        except Exception as e:
            return self.format_error(
                error="Read Error",
                reason=str(e),
                context=f"path={src}",
                suggestion="Verify JSONL integrity"
            )

        # Process and convert events to padded NumPy array.
        try:
            # Build event list.
            padded_events = []
            for ev in tqdm(events, desc="Processing events", unit="evt", **TQDM_CONFIG):
                particles = ev["data"]["particles"]
                rows = [[p["px"], p["py"], p["pz"], p["E"], p["id"]] for p in particles]
                padded_events.append(rows)

            # Calculate max length for padding.
            max_len = max(len(ev) for ev in padded_events) if padded_events else 0

            # Handle edge case: all events are empty
            if max_len == 0:
                return self.format_error(
                    error="Empty Dataset",
                    reason="All events have zero particles",
                    context=f"Processed {len(events)} events, all with empty particle lists",
                    suggestion="Check that the input JSONL file contains events with particles"
                )

            # Create final padded array.
            final_array = np.array([
                np.array(ev + [[0]*5]*(max_len - len(ev))) for ev in padded_events
            ])

            # Save to .npy file.
            # Strip .npy extension if present, as np.save() adds it automatically
            save_path = dst[:-4] if dst.endswith('.npy') else dst
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.save(save_path, final_array)

            result = {
                "status": "ok",
                "input_events": len(events),
                "max_particles": int(max_len),
                "output_shape": list(final_array.shape),
                "saved_path": os.path.relpath(dst, self.base_directory),
            }
            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=str(e),
                suggestion="Ensure file format matches schema evtjsonl-1.0"
            )
        
# ====================================================================== #
# ======================== LHE \to JSONL tool ========================== #
# ====================================================================== #

class LHEToJSONLTool(BaseTool):
    """
    Convert a MadGraph LHE file into the JSONL (evtjsonl-1.0) schema.

    Uses pylhe for parsing. More robust than manual string parsing.

    Input (runtime):
      - lhe_path: relative or absolute path to unweighted_events.lhe (.gz ok)
                  If absolute, it must still live under base_directory or we refuse.
      - jsonl_path: output path for events.jsonl (relative to base_directory)
      - finals_only: keep only status==1 particles (default True)
      - full_history: include mother indices and status (default False)

    Output:
      JSON string:
      {
        "status": "ok",
        "events_jsonl": "<relative path>",
        "n_events": <int>
      }

    Schema per line:
      {
        "schema": "evtjsonl-1.0",
        "event_id": <int>,
        "finals_only": <bool>,
        "full_history": <bool>,
        "data": {
          "n": <int>,
          "particles": [
             {
               "i": <int>,         # index within this filtered final-state list
               "id": <pdgId>,
               "status": <status>, # if full_history True
               "px": <float>,
               "py": <float>,
               "pz": <float>,
               "E":  <float>,
               "m":  <float>,
               "mother1": <int>,   # if full_history True and available
               "mother2": <int>    # if full_history True and available
             },
             ...
          ]
        }
      }
    """
    # --------------------------- Runtime fields --------------------------- #

    lhe_path: str = RuntimeField(description="Path to LHE file (.lhe or .lhe.gz)")
    jsonl_path: str = RuntimeField(description="Relative path to write events.jsonl")
    finals_only: bool = RuntimeField(default=True, description="Keep only status==1 particles")
    full_history: bool = RuntimeField(default=False, description="Include mothers/status in output records")

    # ---------------------------------------------------------------------- #

    # ---------------------------- State fields ---------------------------- #
    
    base_directory: str = StateField(default=".", description="Base sandbox root")

    # ---------------------------------------------------------------------- #

    def _setup(self):
        """Set up the tool by resolving paths."""
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.isdir(self.base_directory):
            raise ValueError(f"Base directory does not exist or is not a directory: {self.base_directory}")

    def _safe_path(self, rel_or_abs: str) -> Optional[str]:
        """Ensures that the path is within the allowed base directory."""
        if not rel_or_abs:
            return None
        # Resolve relative to base_directory
        full = os.path.abspath(os.path.join(self.base_directory, rel_or_abs))
        # Allow only if inside base_directory
        if full.startswith(self.base_directory + os.sep) or full == self.base_directory:
            return full
        return None


    def _run(self) -> str:
        """Run the tool."""
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Path Error", reason=str(e))

        # Resolve in/out paths.
        src_lhe_abs = self._safe_path(self.lhe_path)
        dst_jsonl_abs = self._safe_path(self.jsonl_path)

        if src_lhe_abs is None:
            return self.format_error(
                error="Access Denied",
                reason="lhe_path is outside allowed base_directory",
                context=self.lhe_path,
                suggestion="Copy LHE under base_directory or adjust base_directory"
            )
        if dst_jsonl_abs is None:
            return self.format_error(
                error="Access Denied",
                reason="jsonl_path escapes base_directory",
                context=self.jsonl_path,
                suggestion="Use a relative output path inside base_directory"
            )

        if not os.path.exists(src_lhe_abs):
            return self.format_error(
                error="File Not Found",
                reason="LHE file not found",
                context=self.lhe_path
            )
        
        # Import pylhe.
        try:
            import pylhe
        except Exception as e:
            return self.format_error(
                error="Dependency Missing",
                reason="pylhe is not importable",
                suggestion="pip install pylhe",
                context=str(e)
            )

        # Read events with attributes using pylhe (returns an iterator).
        try:
            events_iter = pylhe.read_lhe_with_attributes(src_lhe_abs)
        except Exception as e:
            return self.format_error(
                error="Read Error",
                reason=f"Failed to parse LHE with pylhe: {e}",
                suggestion="Check file integrity"
            )

        # Prepare output directory.
        try:
            os.makedirs(os.path.dirname(dst_jsonl_abs), exist_ok=True)
        except Exception as e:
            return self.format_error(
                error="Write Error",
                reason=str(e),
                context=f"path={dst_jsonl_abs}",
                suggestion="Verify disk space and permissions"
            )

        # Stream convert -> JSONL.
        n_written = 0
        try:
            with open(dst_jsonl_abs, "w", encoding="utf-8") as outfp:
                for ev_id, ev in enumerate(tqdm(events_iter, desc="Converting LHE to JSONL", unit="evt", **TQDM_CONFIG)):
                    # pylhe event format:
                    # ev["particles"] is a list.
                    # Each particle p has:
                    #   p["id"]          PDG ID (int)
                    #   p["status"]      status code (int)
                    #   p["px"],["py"],["pz"],["e"],["m"]
                    #   p["mother1"],["mother2"],["color1"],["color2"], ...

                    raw_parts = ev["particles"]

                    # Filter final-state if requested.
                    if self.finals_only:
                        filt_parts = [p for p in raw_parts if int(p.get("status", 0)) == 1]
                    else:
                        filt_parts = list(raw_parts)

                    # Build particle list for JSONL row.
                    particles_out = []
                    for i_local, p in enumerate(filt_parts):
                        rec = {
                            "i": i_local,
                            "id": int(p.get("id", 0)),
                            "px": float(p.get("px", 0.0)),
                            "py": float(p.get("py", 0.0)),
                            "pz": float(p.get("pz", 0.0)),
                            "E":  float(p.get("e", 0.0)),
                            "m":  float(p.get("m", 0.0)),
                        }

                        if self.full_history:
                            rec["status"] = int(p.get("status", 0))
                            # Mother indices come from LHE (1-based indices of parent particles in this event).
                            rec["mother1"] = int(p.get("mother1", 0))
                            rec["mother2"] = int(p.get("mother2", 0))

                        particles_out.append(rec)

                    row = {
                        "schema": SCHEMA_VERSION,
                        "event_id": ev_id,
                        "finals_only": bool(self.finals_only),
                        "full_history": bool(self.full_history),
                        "data": {
                            "n_particles": len(particles_out),
                            "particles": particles_out,
                        },
                    }

                    outfp.write(
                        json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"
                    )
                    n_written += 1
        except Exception as e:
            return self.format_error(
                error="Write Error",
                reason=str(e),
                context=f"path={dst_jsonl_abs}",
                suggestion="Check for disk space and file permissions"
            )

        result = {
            "status": "ok",
            "events_jsonl": os.path.relpath(dst_jsonl_abs, self.base_directory),
            "n_events": int(n_written),
        }
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

# ====================================================================== #
# ====================== Jets JSONL to Numpy tool ===================== #
# ====================================================================== #

class JetsJSONLToNumpyTool(BaseTool):
    """
    Convert a JSONL jets dataset into a zero-padded NumPy array.

    This tool handles the jets schema produced by JetClusterSlowJetTool.
    It can extract either jet-level four-momenta or constituent-level four-momenta.

    Input (runtime):
      - jsonl_path: relative path to jets.jsonl file
      - output_path: relative path to save .npy file (e.g. 'data/jets.npy')
      - extraction_mode: what to extract from the jets data
          * "jets" (default): extract jet four-momenta [px, py, pz, E, m]
          * "constituents": extract constituent four-momenta [px, py, pz, E, m] (flattened from all jets)
          * "jets_with_metadata": extract jets with additional columns [px, py, pz, E, m, pT, eta, phi]

    Output array shape:
      - (N_events, N_max_objects, N_features)
      - N_features = 5 for "jets" and "constituents" modes
      - N_features = 8 for "jets_with_metadata" mode

    Schema:
      Expected JSONL line format from JetClusterSlowJetTool:
      {
        "algorithm": "antikt",
        "R": 0.4,
        "ptmin": 20.0,
        "etamax": 5.0,
        "mass_option": 1,
        "n_jets": <int>,
        "jets": [
          {
            "index": <int>,
            "px": <float>, "py": <float>, "pz": <float>, "E": <float>, "m": <float>,
            "pT": <float>, "eta": <float>, "phi": <float>,
            "n_const": <int>,
            "constituents": [
              {"event_index": <int>, "px": <float>, "py": <float>, "pz": <float>, "E": <float>, "m": <float>},
              ...
            ]
          },
          ...
        ],
        "event_index": <int>
      }

    Output JSON:
      {
        "status": "ok",
        "extraction_mode": "<mode>",
        "input_events": <int>,
        "max_objects_per_event": <int>,
        "output_shape": [N_events, N_max_objects, N_features],
        "feature_columns": ["px", "py", "pz", "E", "m", ...],
        "saved_path": "<relative path>"
      }
    """
    # --------------------------- Runtime fields --------------------------- #
    jsonl_path: str = RuntimeField(description="Relative path to jets.jsonl file")
    output_path: str = RuntimeField(description="Relative path to save .npy file (e.g. 'data/jets.npy')")
    extraction_mode: str = RuntimeField(
        default="jets",
        description="Extraction mode: 'jets', 'constituents', or 'jets_with_metadata'"
    )
    # ---------------------------------------------------------------------- #

    # ---------------------------- State fields ---------------------------- #
    base_directory: str = StateField(default=".", description="Base directory for safe paths")
    # ---------------------------------------------------------------------- #

    def _setup(self):
        """Setup base directory and validate it exists."""
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.exists(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _safe_path(self, rel: str) -> Optional[str]:
        """Ensures that the path is within the allowed base directory."""
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    def _run(self) -> str:
        """Run the conversion from jets JSONL to NumPy array."""
        src = self._safe_path(self.jsonl_path)
        dst = self._safe_path(self.output_path)

        # Check for safe paths.
        if not src or not dst:
            return self.format_error(
                error="Access Denied",
                reason="jsonl_path or output_path escapes base_directory",
                suggestion="Use relative paths inside base_directory"
            )

        # Check if source JSONL exists.
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason="Jets JSONL file not found",
                context=f"path={self.jsonl_path}",
                suggestion="Provide a valid jets.jsonl path"
            )

        # Validate extraction mode.
        valid_modes = ["jets", "constituents", "jets_with_metadata"]
        if self.extraction_mode not in valid_modes:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"extraction_mode must be one of {valid_modes}",
                context=f"Got: {self.extraction_mode}",
                suggestion=f"Use one of: {', '.join(valid_modes)}"
            )

        # Read jets from JSONL.
        try:
            with open(src, "r") as f:
                events = [json.loads(line) for line in f]
        except Exception as e:
            return self.format_error(
                error="Read Error",
                reason=str(e),
                context=f"path={src}",
                suggestion="Verify JSONL integrity"
            )

        # Verify schema.
        if not events:
            return self.format_error(
                error="Empty File",
                reason="No events found in JSONL file",
                suggestion="Provide a file with at least one event"
            )

        first_event = events[0]
        if "data" not in first_event:
            return self.format_error(
                error="Schema Error",
                reason="JSONL file does not contain 'data' key",
                suggestion="Ensure file was created by JetClusterSlowJetTool with unified evtjsonl-1.0 schema"
            )
        if "jets" not in first_event["data"]:
            return self.format_error(
                error="Schema Error",
                reason="Event data does not contain 'jets' key",
                suggestion="Ensure file was created by JetClusterSlowJetTool"
            )

        # Process and convert jets to padded NumPy array.
        try:
            padded_events = []
            feature_columns = []

            if self.extraction_mode == "jets":
                # Extract jet four-momenta: [px, py, pz, E, m]
                feature_columns = ["px", "py", "pz", "E", "m"]
                for ev in tqdm(events, desc="Extracting jets", unit="evt", **TQDM_CONFIG):
                    jets = ev["data"].get("jets", [])
                    rows = [[j["px"], j["py"], j["pz"], j["E"], j["m"]] for j in jets]
                    padded_events.append(rows)

            elif self.extraction_mode == "constituents":
                # Extract constituent four-momenta: [px, py, pz, E, m]
                feature_columns = ["px", "py", "pz", "E", "m"]
                for ev in tqdm(events, desc="Extracting constituents", unit="evt", **TQDM_CONFIG):
                    # Flatten all constituents from all jets in this event
                    constituents = []
                    for jet in ev["data"].get("jets", []):
                        for const in jet.get("constituents", []):
                            constituents.append([
                                const["px"], const["py"], const["pz"],
                                const["E"], const["m"]
                            ])
                    padded_events.append(constituents)

            elif self.extraction_mode == "jets_with_metadata":
                # Extract jets with kinematic metadata: [px, py, pz, E, m, pT, eta, phi]
                feature_columns = ["px", "py", "pz", "E", "m", "pT", "eta", "phi"]
                for ev in tqdm(events, desc="Extracting jets with metadata", unit="evt", **TQDM_CONFIG):
                    jets = ev["data"].get("jets", [])
                    rows = [[
                        j["px"], j["py"], j["pz"], j["E"], j["m"],
                        j["pT"], j["eta"], j["phi"]
                    ] for j in jets]
                    padded_events.append(rows)

            # Calculate max length for padding.
            max_len = max(len(ev) for ev in padded_events) if padded_events else 0

            if max_len == 0:
                return self.format_error(
                    error="Empty Dataset",
                    reason="No jets or constituents found in any event",
                    suggestion="Check that the JSONL file contains valid jet data"
                )

            # Determine number of features.
            n_features = len(feature_columns)

            # Create final padded array.
            final_array = np.array([
                np.array(ev + [[0]*n_features]*(max_len - len(ev))) for ev in padded_events
            ])

            # Save to .npy file.
            # Strip .npy extension if present, as np.save() adds it automatically
            save_path = dst[:-4] if dst.endswith('.npy') else dst
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.save(save_path, final_array)

            result = {
                "status": "ok",
                "extraction_mode": self.extraction_mode,
                "input_events": len(events),
                "max_objects_per_event": int(max_len),
                "output_shape": list(final_array.shape),
                "feature_columns": feature_columns,
                "saved_path": os.path.relpath(dst, self.base_directory),
            }
            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=str(e),
                suggestion="Ensure file format matches jets JSONL schema"
            )