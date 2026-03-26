"""
Tool 1: CMSDQMFetchTool
-----------------------
Fetches CMS Data Quality Monitoring histograms from the CMS DQM REST API
(https://cmsweb.cern.ch/dqm/offline/jsonfairy/archive/) for a given run range
and subsystem. Saves results as line-delimited JSON (JSONL) for downstream ML tools.

Data source: CMS DQM GUI REST API (public, no auth required for offline data)
Output:      dqm_raw.jsonl  — one JSON object per run, containing histogram data
"""

import json
import os
import time
from typing import Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .._compat import BaseTool, RuntimeField, StateField


# CMS DQM REST API base URL (public offline data)
CMS_DQM_API = "https://cmsweb.cern.ch/dqm/offline/jsonfairy/archive"

# Known subsystems and sample monitor elements for each
SUBSYSTEM_ME_MAP = {
    "Pixel":   ["PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_1"],
    "ECAL":    ["EcalBarrel/EBOccupancyTask/EBOT digi occupancy"],
    "Tracker": ["SiStrip/MechanicalView/TIB/Summary_ClusterCharge_OnTrack__TIB"],
    "Muon":    ["DT/02-Segments/numberOfSegments_W0"],
    "Hcal":    ["Hcal/DigiTask/Occupancy/depth/depth1"],
}


class CMSDQMFetchTool(BaseTool):
    """
    Fetches CMS DQM histograms from the public CMS DQM REST API for a
    specified run range and detector subsystem. Falls back to synthetic
    dummy histograms when the real API is unreachable (e.g. offline dev).

    Outputs a JSONL file where each line is a JSON object:
        {"run_id": int, "subsystem": str, "monitor_element": str,
         "histogram": [float, ...], "status": "ok"|"synthetic"}
    """

    # --- RuntimeFields (set by the LLM agent) ---
    run_start: int = RuntimeField(
        description="First CMS run number to fetch (e.g. 360000)")
    run_end: int = RuntimeField(
        description="Last CMS run number to fetch (inclusive, e.g. 360050)")
    subsystem: str = RuntimeField(
        description="CMS detector subsystem: 'Pixel', 'ECAL', 'Tracker', 'Muon', or 'Hcal'")
    output_dir: str = RuntimeField(
        description="Subdirectory (inside sandbox) where dqm_raw.jsonl will be written")
    max_runs: Optional[int] = RuntimeField(
        default=50,
        description="Cap on number of runs to fetch (avoids very long API calls)")
    dataset: str = RuntimeField(
        default="/Global/Online/ALL",
        description="CMS dataset string, e.g. '/Global/Online/ALL'")

    # --- StateField (injected by orchestrator) ---
    sandbox_dir: str = StateField(
        description="Root sandbox directory for this HEPTAPOD session")

    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        out_dir  = os.path.join(self.sandbox_dir, self.output_dir)
        out_path = os.path.join(out_dir, "dqm_raw.jsonl")
        os.makedirs(out_dir, exist_ok=True)

        subsystem = self.subsystem.strip()
        if subsystem not in SUBSYSTEM_ME_MAP:
            return {
                "status": "error",
                "message": (f"Unknown subsystem '{subsystem}'. "
                            f"Choose from: {list(SUBSYSTEM_ME_MAP.keys())}"),
            }

        monitor_elements = SUBSYSTEM_ME_MAP[subsystem]
        run_ids = list(range(
            self.run_start,
            min(self.run_end + 1, self.run_start + (self.max_runs or 50)),
        ))

        records = []
        api_hits, synthetic_hits = 0, 0

        for run_id in run_ids:
            for me in monitor_elements:
                record = self._fetch_one(run_id, self.dataset, me, subsystem)
                if record["status"] == "ok":
                    api_hits += 1
                else:
                    synthetic_hits += 1
                records.append(record)
            time.sleep(0.05)   # polite rate-limit

        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        return {
            "status":          "ok",
            "runs_requested":  len(run_ids),
            "records_written": len(records),
            "api_hits":        api_hits,
            "synthetic_hits":  synthetic_hits,
            "output_jsonl":    out_path,
            "subsystem":       subsystem,
        }

    # ------------------------------------------------------------------ #
    def _fetch_one(self, run_id: int, dataset: str,
                   monitor_element: str, subsystem: str) -> dict:
        """Try real API; fall back to synthetic on any failure."""
        base = {
            "run_id":          run_id,
            "subsystem":       subsystem,
            "dataset":         dataset,
            "monitor_element": monitor_element,
        }
        if not REQUESTS_AVAILABLE:
            return {**base, **self._synthetic(run_id), "status": "synthetic",
                    "reason": "requests not installed"}
        try:
            # JSON fairy endpoint — the /render endpoint is image-only
            url = (f"{CMS_DQM_API}/{run_id}"
                   f"{dataset}/{monitor_element}")
            resp = requests.get(url, timeout=5,
                                headers={"Accept": "application/json"})
            if resp.status_code == 200:
                data = resp.json()
                hist = self._parse_histogram(data)
                return {**base, "histogram": hist, "status": "ok"}
            return {**base, **self._synthetic(run_id),
                    "status": "synthetic",
                    "reason": f"HTTP {resp.status_code}"}
        except Exception as exc:
            return {**base, **self._synthetic(run_id),
                    "status": "synthetic", "reason": str(exc)}

    @staticmethod
    def _parse_histogram(data: dict) -> list:
        """Extract bin values from CMS DQM JSON response."""
        try:
            bins = data.get("hist", {}).get("bins", {})
            return [float(v) for v in bins.get("content", [])]
        except Exception:
            return []

    @staticmethod
    def _synthetic(run_id: int) -> dict:
        """Generate reproducible dummy histogram for offline testing."""
        import math, hashlib
        seed  = int(hashlib.md5(str(run_id).encode()).hexdigest()[:8], 16)
        n_bins = 64
        hist  = [
            abs(math.sin(seed * 0.001 + i * 0.3) * 100 + (i % 8))
            for i in range(n_bins)
        ]
        # inject anomaly for every 10th run
        if run_id % 10 == 0:
            hist[32] *= 50.0  # large spike guarantees AE error exceeds threshold
        is_anomaly = (run_id % 10 == 0)
        return {"histogram": hist, "n_bins": n_bins, "is_synthetic_anomaly": is_anomaly}
