"""
# reconstruction.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json
import os
from typing import Optional, List, Dict, Any
import numpy as np
from tqdm import tqdm
import sys
from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

# Configure tqdm to prevent multiple line printing
TQDM_CONFIG = {
    'file': sys.stderr,
    'ncols': 80,
    'leave': True,
    'dynamic_ncols': False
}


class ResonanceReconstructionTool(BaseTool):
    """
    Template-based resonance reconstruction tool for combining pre-selected particle arrays.

    This tool reconstructs parent resonances from their decay products by:
    1. Loading multiple pre-selected particle arrays (from tools like GetHardestNTool)
    2. Combining them event-by-event
    3. Testing all possible pairings
    4. Selecting the best combination according to physics templates

    **Key Design Philosophy:**
    - Accepts multiple input arrays (already filtered/selected by upstream tools)
    - No filtering/selection logic - just combines and analyzes
    - Clean separation: upstream tools select objects, this tool reconstructs resonances

    **Input Schema:**

    Required:
    - particle_arrays: List of paths to JSONL files, each containing pre-selected particles/jets
      * Example: ["hardest_2_leptons.jsonl", "hardest_2_jets.jsonl"]
      * Each file must use evtjsonl-1.0 schema with "particles" or "jets" key
      * All files must have the same number of events
    - template: Analysis pattern to use
      * "two_body_symmetric": Pair-produced resonances (X -> ab, minimize |m1-m2|)
      * "n_body_all_pairs": General k-body combinatorics (2 <= k <= max_k)

    Optional:
    - max_k: Maximum k-body multiplicity (for n_body_all_pairs, default: 2)
    - output_prefix: Prefix for output files (default: auto-generated)
    - hist_bins: Number of histogram bins (default: 50)
    - hist_range: [min, max] for histogram range (default: auto)
    - min_delta_r: Minimum Delta R separation constraint (default: None = no constraint)
      * **IMPORTANT:** Only checks Delta R between objects from DIFFERENT arrays
      * Does NOT constrain objects within the same array
      * When specified with multiple arrays, requires at least one cross-array pair
      * Example: For leptoquark reconstruction with [leptons.jsonl, jets.jsonl]:
        - min_delta_r=0.4 ensures each lepton-jet pair has ΔR > 0.4
        - Leptons can be arbitrarily close to each other (same array)
        - Jets can be arbitrarily close to each other (same array)
        - Rejects pairings that only combine same-array objects (e.g., lepton+lepton)

    **Output Schema:**
    {
      "status": "ok",
      "template": "...",
      "n_arrays": ...,
      "n_total_objects": ...,
      "n_events_analyzed": ...,
      "n_events_successful": ...,
      "n_events_failed": ...,
      "observables": [{"name": "m1", "kind": "per_event"}, ...],
      "histograms": [{"observable": "m1", "bins": [...], "counts": [...]}, ...],
      "data_paths": {"m1": "path/to/m1.npy", ...}
    }

    **Units:** All momenta and masses in GeV.

    **Delta R Filtering Behavior:**

    The `min_delta_r` parameter implements physics-aware filtering that ONLY checks
    separation between objects from different input arrays:

    - **Cross-array pairs ARE checked**: If you provide [leptons.jsonl, jets.jsonl],
      the tool checks ΔR between each lepton-jet pair
    - **Same-array pairs are NOT checked**: Leptons near other leptons or jets near
      other jets do NOT trigger the Delta R constraint
    - **Physically incorrect pairings are rejected**: When using multiple arrays with
      min_delta_r specified, pairings that only combine same-array objects (e.g.,
      pairing lepton+lepton and jet+jet separately) are automatically rejected

    This design ensures that:
    1. You can select closely-spaced leptons or jets without rejection
    2. The Delta R constraint only applies where it matters physically (lepton-jet)
    3. Unphysical reconstruction combinations are automatically excluded

    Example scenarios:
    - ✓ Two leptons with ΔR=0.1 + two jets with ΔR=0.1, all lepton-jet pairs ΔR>0.4: ACCEPTED
    - ✗ Leptons and jets all in same cone with ΔR<0.4: REJECTED (all cross-array pairs fail)
    - ✗ Attempting to pair lepton+lepton when min_delta_r is set: REJECTED (no cross-array pairs)

    **Example 1: Leptoquark S1 S1~ -> ljlj (2 leptons + 2 jets)**
    ```python
    # Step 1: Get hardest 2 leptons
    GetHardestNTool(
        events_path="data/pythia_events.jsonl",
        object_type="truth_particles",
        pdg_filter=[11, -11, 13, -13],
        n=2,
        output_path="data/hardest_2_leptons.jsonl"
    )

    # Step 2: Get hardest 2 jets
    GetHardestNTool(
        events_path="data/jets.jsonl",
        object_type="jets",
        n=2,
        output_path="data/hardest_2_jets.jsonl"
    )

    # Step 3: Reconstruct leptoquarks with Delta R constraint
    ResonanceReconstructionTool(
        particle_arrays=[
            "data/hardest_2_leptons.jsonl",
            "data/hardest_2_jets.jsonl"
        ],
        template="two_body_symmetric",
        min_delta_r=0.4,  # Ensure lepton-jet pairs are well-separated
        output_prefix="outputs/S1_ljlj"
    )
    ```

    **Example 2: General n-body analysis (3 photons + 1 jet)**
    ```python
    ResonanceReconstructionTool(
        particle_arrays=[
            "data/hardest_3_photons.jsonl",
            "data/hardest_1_jet.jsonl"
        ],
        template="n_body_all_pairs",
        max_k=3,  # Compute up to 3-body masses
        output_prefix="outputs/photon_jet"
    )
    ```

    **Example 3: Single array analysis**
    ```python
    # Can also work with a single pre-selected array
    ResonanceReconstructionTool(
        particle_arrays=["data/hardest_4_jets.jsonl"],
        template="two_body_symmetric",
        output_prefix="outputs/dijet_pairs"
    )
    ```
    """

    # ========================== Runtime Fields ============================ #
    particle_arrays: List[str] = RuntimeField(
        description="List of paths to JSONL files containing pre-selected particles/jets. Files can use either 'particles' or 'jets' key. All files must have the same number of events."
    )
    template: str = RuntimeField(
        description="Analysis template: 'two_body_symmetric' or 'n_body_all_pairs'"
    )
    max_k: int = RuntimeField(
        default=2,
        description="Maximum k-body multiplicity for n_body_all_pairs template (default: 2)"
    )
    output_prefix: Optional[str] = RuntimeField(
        default=None,
        description="Prefix for output files (default: auto-generated from first array path)"
    )
    hist_bins: int = RuntimeField(
        default=50,
        description="Number of histogram bins"
    )
    hist_range: Optional[List[float]] = RuntimeField(
        default=None,
        description="[min, max] for histogram range (default: auto)"
    )
    min_delta_r: Optional[float] = RuntimeField(
        default=None,
        description="Minimum Delta R separation constraint (default: None = no constraint). ONLY checks Delta R between objects from DIFFERENT input arrays (e.g., lepton-jet separation). Does NOT constrain objects within the same array (e.g., two leptons can be arbitrarily close). When specified with multiple arrays, automatically rejects pairings that don't include any cross-array pairs."
    )
    # ===================================================================== #

    # =========================== State Fields ============================ #
    base_directory: str = StateField(
        default=".",
        description="Base directory for safe paths"
    )
    # ===================================================================== #

    def _setup(self):
        """Setup base directory and validate it exists."""
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.exists(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _safe_path(self, rel: str) -> Optional[str]:
        """Ensures that the path is within the allowed base directory."""
        if not rel:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    # ===================== Core Physics Functions ======================= #

    def _calculate_eta_phi(self, px: float, py: float, pz: float) -> tuple:
        """Calculate pseudorapidity and azimuthal angle."""
        pt = np.sqrt(px**2 + py**2)
        eta = np.arcsinh(pz / pt) if pt > 0 else 0.0
        phi = np.arctan2(py, px)
        return eta, phi

    def _calculate_delta_r(self, obj1: np.ndarray, obj2: np.ndarray) -> float:
        """
        Calculate Delta R between two 4-vectors.

        Args:
            obj1: 4-vector [px, py, pz, E]
            obj2: 4-vector [px, py, pz, E]

        Returns:
            Delta R separation
        """
        eta1, phi1 = self._calculate_eta_phi(obj1[0], obj1[1], obj1[2])
        eta2, phi2 = self._calculate_eta_phi(obj2[0], obj2[1], obj2[2])

        delta_eta = eta1 - eta2
        delta_phi = phi1 - phi2

        # Wrap delta_phi to [-pi, pi]
        while delta_phi > np.pi:
            delta_phi -= 2 * np.pi
        while delta_phi < -np.pi:
            delta_phi += 2 * np.pi

        return float(np.sqrt(delta_eta**2 + delta_phi**2))

    def _calculate_invariant_mass(self, four_vectors: np.ndarray) -> float:
        """
        Calculate invariant mass from 4-vectors.

        Args:
            four_vectors: array of shape (N, 4) where columns are [px, py, pz, E]

        Returns:
            Invariant mass in GeV (non-negative)
        """
        if len(four_vectors) == 0:
            return 0.0

        # Sum 4-momenta
        p_total = np.sum(four_vectors, axis=0)
        px, py, pz, E = p_total[0], p_total[1], p_total[2], p_total[3]

        # M^2 = E^2 - p^2
        m_squared = E**2 - px**2 - py**2 - pz**2

        # Handle numerical errors
        if m_squared < 0 and m_squared > -1e-6:
            m_squared = 0.0

        return float(np.sqrt(max(0, m_squared)))

    # ==================== Event Loading Functions ======================= #

    def _load_and_merge_arrays(self, array_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Load multiple particle arrays and merge them event-by-event.

        Args:
            array_paths: List of paths to JSONL files

        Returns:
            List of event dictionaries:
            {
                "event_id": int,
                "objects": np.ndarray of shape (N, 4) with [px, py, pz, E],
                "array_indices": np.ndarray of shape (N,) indicating which array each object came from
            }
        """
        # Load all arrays
        all_arrays = []
        for path_idx, path in enumerate(array_paths):
            safe_path = self._safe_path(path)
            if not safe_path or not os.path.exists(safe_path):
                raise ValueError(f"Array {path_idx} not found: {path}")

            with open(safe_path, 'r') as f:
                events = [json.loads(line) for line in f]

            all_arrays.append(events)

        # Validate all arrays have same number of events
        n_events = len(all_arrays[0])
        for idx, arr in enumerate(all_arrays):
            if len(arr) != n_events:
                raise ValueError(
                    f"Array {idx} has {len(arr)} events, expected {n_events}. "
                    "All arrays must have the same number of events."
                )

        # Merge arrays event-by-event
        merged_events = []
        for event_idx in range(n_events):
            combined_objects = []
            array_indices = []

            # Collect objects from all arrays for this event
            for array_idx, array in enumerate(all_arrays):
                event = array[event_idx]

                if "data" not in event:
                    raise ValueError(
                        f"Event {event_idx} in array {array_idx} missing 'data' key"
                    )

                # Try both 'particles' and 'jets' keys
                objects = None
                if "particles" in event["data"]:
                    objects = event["data"]["particles"]
                elif "jets" in event["data"]:
                    objects = event["data"]["jets"]
                else:
                    raise ValueError(
                        f"Event {event_idx} in array {array_idx} has neither 'data.particles' nor 'data.jets' key"
                    )

                # Convert to 4-vectors and track array origin
                for obj in objects:
                    combined_objects.append([obj["px"], obj["py"], obj["pz"], obj["E"]])
                    array_indices.append(array_idx)

            # Convert to numpy arrays
            if len(combined_objects) > 0:
                objects = np.array(combined_objects)
                array_indices = np.array(array_indices)
            else:
                objects = np.array([]).reshape(0, 4)
                array_indices = np.array([])

            merged_events.append({
                "event_id": event_idx,
                "objects": objects,
                "array_indices": array_indices
            })

        return merged_events

    # ====================== Template Implementations ==================== #

    def _template_two_body_symmetric(self, events: List[Dict]) -> Dict[str, Any]:
        """
        Two-body symmetric template for pair-produced resonances.

        Example: pp -> S1 S1~ where S1 -> l j

        Algorithm:
        1. Use all objects from combined arrays
        2. Form all disjoint (a,b) pairs
        3. For each pairing, compute m1, m2
        4. Choose pairing that minimizes |m1 - m2|
        5. Output: m1, m2, m_min, m_max per event
        """
        m1_list = []
        m2_list = []
        m_min_list = []
        m_max_list = []

        for ev in tqdm(events, desc="Analyzing events (two_body_symmetric)", unit="evt", **TQDM_CONFIG):
            objects = ev["objects"]
            array_indices = ev["array_indices"]
            n_objects = len(objects)

            # Check if we have enough objects
            if n_objects < 2:
                # Not enough objects - add placeholder values
                m1_list.append(0.0)
                m2_list.append(0.0)
                m_min_list.append(0.0)
                m_max_list.append(0.0)
                continue

            # Generate all possible disjoint pairings
            # For n=4: (0,1)+(2,3), (0,2)+(1,3), (0,3)+(1,2)
            if n_objects == 4:
                pairings = [
                    [(0, 1), (2, 3)],
                    [(0, 2), (1, 3)],
                    [(0, 3), (1, 2)]
                ]
            elif n_objects == 2:
                pairings = [[(0, 1)]]
            else:
                # For general n, this becomes more complex
                # For now, handle n=4 and n=2 explicitly
                raise ValueError(f"two_body_symmetric template requires 2 or 4 objects, got {n_objects}")

            # Evaluate each pairing
            best_pairing = None
            best_asymmetry = float('inf')
            best_avg_mass = 0.0  # Tiebreaker: prefer larger masses

            for pairing in pairings:
                # Check Delta R constraint if specified
                if self.min_delta_r is not None:
                    # Check all pairs in this pairing meet Delta R requirement
                    # ONLY check pairs where objects come from different arrays
                    valid_pairing = True
                    has_cross_array_pair = False

                    for pair_indices in pairing:
                        if len(pair_indices) == 2:
                            i, j = pair_indices
                            # Only apply Delta R constraint if objects are from different arrays
                            if array_indices[i] != array_indices[j]:
                                has_cross_array_pair = True
                                delta_r = self._calculate_delta_r(objects[i], objects[j])
                                if delta_r < self.min_delta_r:
                                    valid_pairing = False
                                    break

                    # If min_delta_r is specified and we have multiple arrays,
                    # require at least one cross-array pair (otherwise we're just
                    # pairing leptons with leptons and jets with jets, which is wrong)
                    if len(np.unique(array_indices)) > 1 and not has_cross_array_pair:
                        valid_pairing = False

                    if not valid_pairing:
                        continue  # Skip this pairing

                # Calculate masses for this pairing
                masses = []
                for pair_indices in pairing:
                    pair_4vecs = objects[list(pair_indices)]
                    mass = self._calculate_invariant_mass(pair_4vecs)
                    masses.append(mass)

                # Calculate asymmetry
                if len(masses) == 2:
                    asymmetry = abs(masses[0] - masses[1])
                    avg_mass = (masses[0] + masses[1]) / 2.0
                elif len(masses) == 1:
                    asymmetry = 0.0
                    avg_mass = masses[0]
                else:
                    asymmetry = float('inf')
                    avg_mass = 0.0

                # Choose best pairing: minimize asymmetry, then maximize average mass (tiebreaker)
                is_better = False
                if asymmetry < best_asymmetry:
                    is_better = True
                elif asymmetry == best_asymmetry and avg_mass > best_avg_mass:
                    is_better = True

                if is_better:
                    best_asymmetry = asymmetry
                    best_avg_mass = avg_mass
                    best_pairing = masses

            # Store best pairing
            if best_pairing is not None:
                if len(best_pairing) == 2:
                    m1_list.append(best_pairing[0])
                    m2_list.append(best_pairing[1])
                    m_min_list.append(min(best_pairing))
                    m_max_list.append(max(best_pairing))
                elif len(best_pairing) == 1:
                    m1_list.append(best_pairing[0])
                    m2_list.append(0.0)
                    m_min_list.append(best_pairing[0])
                    m_max_list.append(best_pairing[0])
            else:
                # No valid pairing found (e.g., all pairings failed Delta R cut)
                m1_list.append(0.0)
                m2_list.append(0.0)
                m_min_list.append(0.0)
                m_max_list.append(0.0)

        # Convert to numpy arrays
        results = {
            "m1": np.array(m1_list),
            "m2": np.array(m2_list),
            "m_min": np.array(m_min_list),
            "m_max": np.array(m_max_list)
        }

        # Define observables metadata
        observables = [
            {"name": "m1", "kind": "per_event"},
            {"name": "m2", "kind": "per_event"},
            {"name": "m_min", "kind": "per_event"},
            {"name": "m_max", "kind": "per_event"}
        ]

        return {
            "results": results,
            "observables": observables
        }

    def _template_n_body_all_pairs(self, events: List[Dict]) -> Dict[str, Any]:
        """
        N-body all-pairs template for general combinatorics.

        Computes all k-body invariant masses for 2 <= k <= max_k.
        """
        max_k = self.max_k

        # Storage for each k
        results = {}
        for k in range(2, max_k + 1):
            results[f"m_k{k}"] = []

        for ev in tqdm(events, desc="Analyzing events (n_body_all_pairs)", unit="evt", **TQDM_CONFIG):
            objects = ev["objects"]

            if len(objects) == 0:
                continue

            # Generate all k-body combinations
            from itertools import combinations
            for k in range(2, max_k + 1):
                if len(objects) < k:
                    continue

                for combo in combinations(range(len(objects)), k):
                    combo_4vecs = objects[list(combo)]
                    mass = self._calculate_invariant_mass(combo_4vecs)
                    results[f"m_k{k}"].append(mass)

        # Convert to numpy arrays
        for key in results:
            results[key] = np.array(results[key])

        # Define observables
        observables = [
            {"name": f"m_k{k}", "kind": "per_combination"}
            for k in range(2, max_k + 1)
        ]

        return {
            "results": results,
            "observables": observables
        }


    # ==================== Histogramming Functions ======================= #

    def _create_histograms(self, results: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Create histograms for all observables."""
        histograms = []

        for name, values in results.items():
            if len(values) == 0:
                # Empty histogram
                histograms.append({
                    "observable": name,
                    "bins": [],
                    "counts": []
                })
                continue

            # Determine range
            if self.hist_range is not None:
                hist_range = tuple(self.hist_range)
            else:
                hist_range = (float(np.min(values)), float(np.max(values)))

            # Create histogram
            counts, bin_edges = np.histogram(values, bins=self.hist_bins, range=hist_range)

            histograms.append({
                "observable": name,
                "bins": bin_edges.tolist(),
                "counts": counts.tolist()
            })

        return histograms

    # ====================== Output Functions ============================ #

    def _save_results(self, results: Dict[str, np.ndarray], output_prefix: str) -> Dict[str, str]:
        """Save result arrays to .npy files."""
        data_paths = {}

        # Resolve output_prefix relative to base_directory
        if not os.path.isabs(output_prefix):
            output_prefix = os.path.join(self.base_directory, output_prefix)

        output_dir = os.path.dirname(output_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for name, values in results.items():
            output_path = f"{output_prefix}_{name}.npy"
            np.save(output_path, values)

            # Store relative path
            rel_path = os.path.relpath(output_path, self.base_directory)
            data_paths[name] = rel_path

        return data_paths

    # ======================== Main Run Function ========================= #

    def _run(self) -> str:
        """Run the resonance reconstruction analysis."""
        try:
            # Validate inputs
            if self.template not in ["two_body_symmetric", "n_body_all_pairs"]:
                return self.format_error(
                    error="Invalid Template",
                    reason=f"Unknown template: {self.template}. Valid options: 'two_body_symmetric', 'n_body_all_pairs'"
                )

            if not self.particle_arrays or len(self.particle_arrays) == 0:
                return self.format_error(
                    error="Invalid Input",
                    reason="particle_arrays is empty. Provide at least one array path."
                )

            # Auto-generate output prefix if not provided
            if self.output_prefix is None:
                basename = os.path.splitext(os.path.basename(self.particle_arrays[0]))[0]
                dirname = os.path.dirname(self.particle_arrays[0])
                if dirname:
                    self.output_prefix = os.path.join(dirname, f"{basename}_{self.template}")
                else:
                    self.output_prefix = f"{basename}_{self.template}"

            # Load and merge arrays
            print(f"Loading {len(self.particle_arrays)} particle array(s)...")
            events = self._load_and_merge_arrays(self.particle_arrays)
            print(f"Loaded {len(events)} events")

            # Calculate total objects per event (for reporting)
            n_total_objects = int(np.mean([len(ev["objects"]) for ev in events]))

            # Run template analysis
            if self.template == "two_body_symmetric":
                analysis_result = self._template_two_body_symmetric(events)
            elif self.template == "n_body_all_pairs":
                analysis_result = self._template_n_body_all_pairs(events)

            results = analysis_result["results"]
            observables = analysis_result["observables"]

            # Create histograms
            print("Creating histograms...")
            histograms = self._create_histograms(results)

            # Save results
            print(f"Saving results to {self.output_prefix}...")
            data_paths = self._save_results(results, self.output_prefix)

            # Calculate reconstruction statistics
            # For per-event observables, count how many events had successful reconstruction (non-zero values)
            n_successful = 0
            n_failed = 0

            # Find a per-event observable to check
            per_event_obs = [obs for obs in observables if obs.get("kind") == "per_event"]
            if per_event_obs and len(per_event_obs) > 0:
                obs_name = per_event_obs[0]["name"]
                values = results[obs_name]
                n_successful = int(np.sum(values > 0))
                n_failed = int(np.sum(values == 0))

            # Build output JSON
            output = {
                "status": "ok",
                "template": self.template,
                "n_arrays": len(self.particle_arrays),
                "n_total_objects": n_total_objects,
                "n_events_analyzed": len(events),
                "n_events_successful": n_successful,
                "n_events_failed": n_failed,
                "observables": observables,
                "histograms": histograms,
                "data_paths": data_paths
            }

            return json.dumps(output, separators=(",", ":"), ensure_ascii=False)

        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=str(e)
            )
