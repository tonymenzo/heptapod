"""
# kinematics.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json, os
from typing import Optional, List, Union
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

# ====================================================================== #
# ====================== Kinematics Tools ============================== #
# ====================================================================== #

class CalculateInvariantMassTool(BaseTool):
    """
    Calculate the invariant mass of a system of particles.

    **Primary usage (file-based):**
    - input_file: Path to .npy or .jsonl file with particle data
    - output_file: Path to save results (.npy format)
    - particle_indices: Which particles to combine (optional, defaults to all)

    **Legacy support:**
    - Can still accept direct particle lists for simple cases (but not recommended for large datasets)

    Input formats:
    1. .npy file with shape (N_particles, 4+) or (N_events, N_particles, 4+)
    2. .jsonl file with event structure

    Output:
    - .npy file with invariant masses (shape: (N_events,) for multi-event, scalar for single event)
    - JSON summary with statistics

    Formula: M^2 = (sum E)^2 - (sum px)^2 - (sum py)^2 - (sum pz)^2
    """
    # --------------------------- Runtime fields --------------------------- #
    input_file: Optional[str] = RuntimeField(
        default=None,
        description="Path to input .npy or .jsonl file with particle data"
    )
    output_file: Optional[str] = RuntimeField(
        default=None,
        description="Path to save output .npy file with invariant masses (auto-generated if not provided)"
    )
    particle_indices: Optional[List[int]] = RuntimeField(
        default=None,
        description="Indices of particles to combine (e.g., [0, 1] for first two particles). If None, use all particles."
    )
    pdgids: Optional[List[int]] = RuntimeField(
        default=None,
        description="Filter particles by PDG IDs (only for jsonl input)"
    )
    event_index: Optional[int] = RuntimeField(
        default=None,
        description="Process only this event index (for multi-event input)"
    )
    # Legacy support (deprecated)
    particles: Optional[List[List[float]]] = RuntimeField(
        default=None,
        description="[DEPRECATED] List of particle 4-vectors [[px, py, pz, E], ...]. Use input_file instead."
    )
    npy_path: Optional[str] = RuntimeField(
        default=None,
        description="[DEPRECATED] Use input_file instead"
    )
    jsonl_path: Optional[str] = RuntimeField(
        default=None,
        description="[DEPRECATED] Use input_file instead"
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
        if not rel:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    def _calculate_mass(self, particles_4vec: np.ndarray) -> float:
        """
        Calculate invariant mass from 4-vectors.
        particles_4vec: array of shape (N, 4+) where columns are [px, py, pz, E, ...]
        """
        if len(particles_4vec) == 0:
            return 0.0

        # Sum 4-momenta
        px_sum = np.sum(particles_4vec[:, 0])
        py_sum = np.sum(particles_4vec[:, 1])
        pz_sum = np.sum(particles_4vec[:, 2])
        E_sum = np.sum(particles_4vec[:, 3])

        # M^2 = E^2 - p^2
        m_squared = E_sum**2 - px_sum**2 - py_sum**2 - pz_sum**2

        # Handle numerical errors that might make m_squared slightly negative
        if m_squared < 0 and m_squared > -1e-6:
            m_squared = 0.0

        return float(np.sqrt(max(0, m_squared)))

    def _run_file_based(self) -> str:
        """Run file-based invariant mass calculation with output to .npy file."""
        src = self._safe_path(self.input_file)
        if not src:
            return self.format_error(
                error="Access Denied",
                reason="input_file escapes base_directory"
            )
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason=f"Input file not found: {self.input_file}"
            )

        # Auto-generate output file if not provided
        if self.output_file is None:
            input_basename = os.path.splitext(os.path.basename(self.input_file))[0]
            self.output_file = os.path.join(
                os.path.dirname(self.input_file) if os.path.dirname(self.input_file) else ".",
                f"{input_basename}_invariant_masses.npy"
            )

        dst = self._safe_path(self.output_file)
        if not dst:
            return self.format_error(
                error="Access Denied",
                reason="output_file escapes base_directory"
            )

        try:
            # Handle NumPy input
            if self.input_file.endswith('.npy'):
                data = np.load(src)

                # Handle different array shapes
                if data.ndim == 2:
                    # Single event: (N_particles, features)
                    mask = np.any(data != 0, axis=1)
                    particles = data[mask]

                    # Apply particle_indices if specified
                    if self.particle_indices is not None:
                        particles = particles[self.particle_indices]

                    mass = self._calculate_mass(particles)
                    masses_array = np.array([mass])
                    n_particles = [len(particles)]

                elif data.ndim == 3:
                    # Multiple events: (N_events, N_particles, features)
                    if self.event_index is not None:
                        if self.event_index >= len(data):
                            return self.format_error(
                                error="Index Error",
                                reason=f"event_index {self.event_index} out of range [0, {len(data)-1}]"
                            )
                        data = data[self.event_index:self.event_index+1]

                    masses = []
                    n_particles = []
                    for event in tqdm(data, desc="Calculating invariant masses", unit="evt", **TQDM_CONFIG):
                        mask = np.any(event != 0, axis=1)
                        particles = event[mask]

                        # Apply particle_indices if specified
                        if self.particle_indices is not None:
                            particles = particles[self.particle_indices]

                        masses.append(self._calculate_mass(particles))
                        n_particles.append(len(particles))

                    masses_array = np.array(masses)
                else:
                    return self.format_error(
                        error="Invalid Shape",
                        reason=f"Expected 2D or 3D array, got shape {data.shape}"
                    )

            # Handle JSONL input
            elif self.input_file.endswith('.jsonl'):
                with open(src, 'r') as f:
                    events = [json.loads(line) for line in f]

                if self.event_index is not None:
                    if self.event_index >= len(events):
                        return self.format_error(
                            error="Index Error",
                            reason=f"event_index {self.event_index} out of range [0, {len(events)-1}]"
                        )
                    events = [events[self.event_index]]

                masses = []
                n_particles = []
                for ev in tqdm(events, desc="Calculating invariant masses", unit="evt", **TQDM_CONFIG):
                    if "data" not in ev or "particles" not in ev["data"]:
                        return self.format_error(
                            error="Invalid Format",
                            reason="Event missing 'data.particles' key"
                        )

                    particles_data = ev["data"]["particles"]

                    # Filter by PDG ID if requested
                    if self.pdgids is not None:
                        particles_data = [p for p in particles_data if p["id"] in self.pdgids]

                    # Apply particle_indices if specified
                    if self.particle_indices is not None:
                        particles_data = [particles_data[i] for i in self.particle_indices if i < len(particles_data)]

                    # Extract 4-vectors
                    particles_array = np.array([
                        [p["px"], p["py"], p["pz"], p["E"]]
                        for p in particles_data
                    ])

                    mass = self._calculate_mass(particles_array)
                    masses.append(mass)
                    n_particles.append(len(particles_array))

                masses_array = np.array(masses)
            else:
                return self.format_error(
                    error="Invalid Format",
                    reason="Input file must be .npy or .jsonl"
                )

            # Save output
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            save_path = dst[:-4] if dst.endswith('.npy') else dst
            np.save(save_path, masses_array)

            # Build result
            result = {
                "status": "ok",
                "output_file": os.path.relpath(dst if dst.endswith('.npy') else dst + '.npy', self.base_directory),
                "n_events": len(masses_array),
                "mean_mass": float(np.mean(masses_array)),
                "std_mass": float(np.std(masses_array)),
                "min_mass": float(np.min(masses_array)),
                "max_mass": float(np.max(masses_array)),
                "masses_shape": list(masses_array.shape),
                "n_particles_per_event": n_particles
            }

            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=str(e)
            )

    def _run(self) -> str:
        """Run the invariant mass calculation."""
        # Priority 1: New file-based I/O (only if input_file is explicitly set)
        if self.input_file is not None:
            return self._run_file_based()

        # Priority 2: Legacy direct particle list
        if self.particles is not None:
            try:
                particles_array = np.array(self.particles)

                # Handle empty list
                if len(self.particles) == 0:
                    result = {
                        "status": "ok",
                        "invariant_mass": 0.0,
                        "n_particles": 0
                    }
                    return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

                if particles_array.ndim != 2 or particles_array.shape[1] < 4:
                    return self.format_error(
                        error="Invalid Input",
                        reason="particles must be a list of 4-vectors with at least 4 components [px, py, pz, E]",
                        suggestion="Ensure each particle has [px, py, pz, E]"
                    )
                mass = self._calculate_mass(particles_array)
                result = {
                    "status": "ok",
                    "invariant_mass": mass,
                    "n_particles": len(particles_array)
                }
                return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
            except Exception as e:
                return self.format_error(
                    error="Calculation Error",
                    reason=str(e)
                )

        # Priority 3: Legacy NumPy file (deprecated but supported)
        if self.npy_path is not None:
            src = self._safe_path(self.npy_path)
            if not src:
                return self.format_error(
                    error="Access Denied",
                    reason="npy_path escapes base_directory"
                )
            if not os.path.exists(src):
                return self.format_error(
                    error="File Not Found",
                    reason=f"NumPy file not found: {self.npy_path}"
                )

            try:
                data = np.load(src)

                # Handle different array shapes
                if data.ndim == 2:
                    # Single event: (N_particles, features)
                    mask = np.any(data != 0, axis=1)
                    particles = data[mask]
                    mass = self._calculate_mass(particles)
                    result = {
                        "status": "ok",
                        "invariant_mass": mass,
                        "n_particles": len(particles)
                    }
                elif data.ndim == 3:
                    # Multiple events: (N_events, N_particles, features)
                    if self.event_index is not None:
                        if self.event_index >= len(data):
                            return self.format_error(
                                error="Index Error",
                                reason=f"event_index {self.event_index} out of range [0, {len(data)-1}]"
                            )
                        # Single event
                        mask = np.any(data[self.event_index] != 0, axis=1)
                        particles = data[self.event_index][mask]
                        mass = self._calculate_mass(particles)
                        result = {
                            "status": "ok",
                            "invariant_mass": mass,
                            "n_particles": len(particles),
                            "event_index": self.event_index
                        }
                    else:
                        # All events
                        masses = []
                        n_particles_list = []
                        for event in data:
                            mask = np.any(event != 0, axis=1)
                            particles = event[mask]
                            masses.append(self._calculate_mass(particles))
                            n_particles_list.append(len(particles))
                        result = {
                            "status": "ok",
                            "invariant_masses": masses,
                            "n_particles_per_event": n_particles_list,
                            "n_events": len(data)
                        }
                else:
                    return self.format_error(
                        error="Invalid Shape",
                        reason=f"Expected 2D or 3D array, got shape {data.shape}"
                    )

                return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
            except Exception as e:
                return self.format_error(
                    error="Processing Error",
                    reason=str(e)
                )

        # Priority 4: Legacy JSONL file (deprecated but supported)
        if self.jsonl_path is not None:
            src = self._safe_path(self.jsonl_path)
            if not src:
                return self.format_error(
                    error="Access Denied",
                    reason="jsonl_path escapes base_directory"
                )
            if not os.path.exists(src):
                return self.format_error(
                    error="File Not Found",
                    reason=f"JSONL file not found: {self.jsonl_path}"
                )

            try:
                with open(src, "r") as f:
                    events = [json.loads(line) for line in f]

                if self.event_index is not None:
                    if self.event_index >= len(events):
                        return self.format_error(
                            error="Index Error",
                            reason=f"event_index {self.event_index} out of range [0, {len(events)-1}]"
                        )
                    events = [events[self.event_index]]

                masses = []
                n_particles_list = []
                for ev in events:
                    if "data" not in ev or "particles" not in ev["data"]:
                        return self.format_error(
                            error="Invalid Format",
                            reason="Event missing 'data.particles' key"
                        )

                    particles_data = ev["data"]["particles"]

                    # Filter by PDG ID if requested
                    if self.pdgids is not None:
                        particles_data = [p for p in particles_data if p["id"] in self.pdgids]

                    # Extract 4-vectors
                    particles_array = np.array([
                        [p["px"], p["py"], p["pz"], p["E"]]
                        for p in particles_data
                    ])

                    mass = self._calculate_mass(particles_array)
                    masses.append(mass)
                    n_particles_list.append(len(particles_array))

                if len(masses) == 1:
                    result = {
                        "status": "ok",
                        "invariant_mass": masses[0],
                        "n_particles": n_particles_list[0]
                    }
                    if self.event_index is not None:
                        result["event_index"] = self.event_index
                else:
                    result = {
                        "status": "ok",
                        "invariant_masses": masses,
                        "n_particles_per_event": n_particles_list,
                        "n_events": len(masses)
                    }

                return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
            except Exception as e:
                return self.format_error(
                    error="Processing Error",
                    reason=str(e)
                )

        return self.format_error(
            error="Missing Input",
            reason="Must provide input_file (recommended) or particles/npy_path/jsonl_path (legacy)"
        )


class CalculateTransverseMomentumTool(BaseTool):
    """
    Calculate transverse momentum (pT) for particles.

    **Primary usage (file-based):**
    - input_file: Path to .npy or .jsonl file with particle data
    - output_file: Path to save results (.npy format)

    **Legacy support:**
    - Can still accept direct particle lists for simple cases

    Returns pT = sqrt(px^2 + py^2) for each particle.
    Output: .npy file with pT values per event
    """
    # --------------------------- Runtime fields --------------------------- #
    input_file: Optional[str] = RuntimeField(
        default=None,
        description="Path to input .npy or .jsonl file with particle data"
    )
    output_file: Optional[str] = RuntimeField(
        default=None,
        description="Path to save output .npy file with pT values (auto-generated if not provided)"
    )
    event_index: Optional[int] = RuntimeField(
        default=None,
        description="Process only this event index (for multi-event input)"
    )
    # Legacy support (deprecated)
    particles: Optional[Union[List[float], List[List[float]]]] = RuntimeField(
        default=None,
        description="[DEPRECATED] Single particle [px, py, pz, E] or list of particles. Use input_file instead."
    )
    npy_path: Optional[str] = RuntimeField(
        default=None,
        description="[DEPRECATED] Use input_file instead"
    )
    jsonl_path: Optional[str] = RuntimeField(
        default=None,
        description="[DEPRECATED] Use input_file instead"
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
        if not rel:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    def _calculate_pt(self, particles_4vec: np.ndarray) -> np.ndarray:
        """
        Calculate pT for particles.
        particles_4vec: array of shape (N, 4+) where columns are [px, py, pz, E, ...]
        Returns: array of shape (N,) with pT values
        """
        px = particles_4vec[:, 0]
        py = particles_4vec[:, 1]
        return np.sqrt(px**2 + py**2)

    def _run_file_based(self) -> str:
        """Run file-based pT calculation with output to .npy file."""
        src = self._safe_path(self.input_file)
        if not src:
            return self.format_error(
                error="Access Denied",
                reason="input_file escapes base_directory"
            )
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason=f"Input file not found: {self.input_file}"
            )

        # Auto-generate output file if not provided
        if self.output_file is None:
            input_basename = os.path.splitext(os.path.basename(self.input_file))[0]
            self.output_file = os.path.join(
                os.path.dirname(self.input_file) if os.path.dirname(self.input_file) else ".",
                f"{input_basename}_pt.npy"
            )

        dst = self._safe_path(self.output_file)
        if not dst:
            return self.format_error(
                error="Access Denied",
                reason="output_file escapes base_directory"
            )

        try:
            # Handle NumPy input
            if self.input_file.endswith('.npy'):
                data = np.load(src)

                if data.ndim == 2:
                    # Single event: (N_particles, features)
                    mask = np.any(data != 0, axis=1)
                    particles = data[mask]
                    pt_values = self._calculate_pt(particles)
                    pt_array = pt_values[np.newaxis, :]  # Shape: (1, N_particles)

                elif data.ndim == 3:
                    # Multiple events: (N_events, N_particles, features)
                    if self.event_index is not None:
                        if self.event_index >= len(data):
                            return self.format_error(
                                error="Index Error",
                                reason=f"event_index {self.event_index} out of range [0, {len(data)-1}]"
                            )
                        data = data[self.event_index:self.event_index+1]

                    pt_list = []
                    for event in tqdm(data, desc="Calculating transverse momenta", unit="evt", **TQDM_CONFIG):
                        mask = np.any(event != 0, axis=1)
                        particles = event[mask]
                        pt_values = self._calculate_pt(particles)
                        pt_list.append(pt_values)

                    # Pad to same length
                    max_len = max(len(pt) for pt in pt_list) if pt_list else 0
                    if max_len > 0:
                        pt_array = np.array([
                            np.pad(pt, (0, max_len - len(pt)), constant_values=0)
                            for pt in pt_list
                        ])
                    else:
                        pt_array = np.array([])
                else:
                    return self.format_error(
                        error="Invalid Shape",
                        reason=f"Expected 2D or 3D array, got shape {data.shape}"
                    )

            # Handle JSONL input
            elif self.input_file.endswith('.jsonl'):
                with open(src, 'r') as f:
                    events = [json.loads(line) for line in f]

                if self.event_index is not None:
                    if self.event_index >= len(events):
                        return self.format_error(
                            error="Index Error",
                            reason=f"event_index {self.event_index} out of range [0, {len(events)-1}]"
                        )
                    events = [events[self.event_index]]

                pt_list = []
                for ev in tqdm(events, desc="Calculating transverse momenta", unit="evt", **TQDM_CONFIG):
                    if "data" not in ev or "particles" not in ev["data"]:
                        return self.format_error(
                            error="Invalid Format",
                            reason="Event missing 'data.particles' key"
                        )

                    particles_data = ev["data"]["particles"]
                    particles_array = np.array([
                        [p["px"], p["py"], p["pz"], p["E"]]
                        for p in particles_data
                    ])
                    pt_values = self._calculate_pt(particles_array)
                    pt_list.append(pt_values)

                # Pad to same length
                max_len = max(len(pt) for pt in pt_list) if pt_list else 0
                if max_len > 0:
                    pt_array = np.array([
                        np.pad(pt, (0, max_len - len(pt)), constant_values=0)
                        for pt in pt_list
                    ])
                else:
                    pt_array = np.array([])
            else:
                return self.format_error(
                    error="Invalid Format",
                    reason="Input file must be .npy or .jsonl"
                )

            # Save output
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            save_path = dst[:-4] if dst.endswith('.npy') else dst
            np.save(save_path, pt_array)

            # Build result
            result = {
                "status": "ok",
                "output_file": os.path.relpath(dst if dst.endswith('.npy') else dst + '.npy', self.base_directory),
                "n_events": len(pt_array),
                "output_shape": list(pt_array.shape)
            }

            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=str(e)
            )

    def _run(self) -> str:
        """Run the pT calculation."""
        # Handle legacy parameter mapping
        if self.npy_path is not None and self.input_file is None:
            self.input_file = self.npy_path
        if self.jsonl_path is not None and self.input_file is None:
            self.input_file = self.jsonl_path

        # Priority 1: File-based I/O (recommended approach)
        if self.input_file is not None:
            return self._run_file_based()

        # Priority 2: Legacy direct particle list
        if self.particles is not None:
            try:
                particles_array = np.array(self.particles)

                # Handle single particle
                if particles_array.ndim == 1:
                    if len(particles_array) < 2:
                        return self.format_error(
                            error="Invalid Input",
                            reason="Particle must have at least [px, py]"
                        )
                    pt = float(np.sqrt(particles_array[0]**2 + particles_array[1]**2))
                    result = {
                        "status": "ok",
                        "pt": pt
                    }
                # Handle multiple particles
                elif particles_array.ndim == 2:
                    if particles_array.shape[1] < 2:
                        return self.format_error(
                            error="Invalid Input",
                            reason="Each particle must have at least [px, py]"
                        )
                    pt_values = self._calculate_pt(particles_array).tolist()
                    result = {
                        "status": "ok",
                        "pt_values": pt_values,
                        "n_particles": len(pt_values)
                    }
                else:
                    return self.format_error(
                        error="Invalid Shape",
                        reason=f"Expected 1D or 2D array, got shape {particles_array.shape}"
                    )

                return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
            except Exception as e:
                return self.format_error(
                    error="Calculation Error",
                    reason=str(e)
                )

        return self.format_error(
            error="Missing Input",
            reason="Must provide input_file or particles (legacy)"
        )


class CalculateDeltaRTool(BaseTool):
    """
    Calculate Delta R between particles.

    Delta R = sqrt(delta_eta^2 + delta_phi^2)
    where eta = pseudorapidity, phi = azimuthal angle

    **Primary usage (file-based):**
    - input_file: Path to .npy or .jsonl file with particle data
    - output_file: Path to save results (.npy format)
    - particle_pairs: List of [i, j] indices to compute Delta R for (e.g., [[0,1], [0,2], [1,2]])

    **Legacy support:**
    - Can still accept two single particles for simple cases

    Output: .npy file with Delta R values (shape depends on number of pairs and events)
    """
    # --------------------------- Runtime fields --------------------------- #
    input_file: Optional[str] = RuntimeField(
        default=None,
        description="Path to input .npy or .jsonl file with particle data"
    )
    output_file: Optional[str] = RuntimeField(
        default=None,
        description="Path to save output .npy file with Delta R values (auto-generated if not provided)"
    )
    particle_pairs: Optional[List[List[int]]] = RuntimeField(
        default=None,
        description="List of particle index pairs [[i,j], ...] to compute Delta R. If None, computes all pairs."
    )
    event_index: Optional[int] = RuntimeField(
        default=None,
        description="Process only this event index (for multi-event input)"
    )
    # Legacy support (deprecated)
    particle1: Optional[List[float]] = RuntimeField(
        default=None,
        description="[DEPRECATED] First particle 4-vector [px, py, pz, E]. Use input_file instead."
    )
    particle2: Optional[List[float]] = RuntimeField(
        default=None,
        description="[DEPRECATED] Second particle 4-vector [px, py, pz, E]. Use input_file instead."
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
        if not rel:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    def _calculate_eta_phi(self, px: float, py: float, pz: float) -> tuple:
        """Calculate pseudorapidity and azimuthal angle."""
        pt = np.sqrt(px**2 + py**2)
        eta = np.arcsinh(pz / pt) if pt > 0 else 0.0
        phi = np.arctan2(py, px)
        return eta, phi

    def _calculate_delta_r(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Delta R between two particles."""
        eta1, phi1 = self._calculate_eta_phi(p1[0], p1[1], p1[2])
        eta2, phi2 = self._calculate_eta_phi(p2[0], p2[1], p2[2])

        delta_eta = eta1 - eta2
        delta_phi = phi1 - phi2

        # Wrap delta_phi to [-pi, pi]
        while delta_phi > np.pi:
            delta_phi -= 2 * np.pi
        while delta_phi < -np.pi:
            delta_phi += 2 * np.pi

        return float(np.sqrt(delta_eta**2 + delta_phi**2))

    def _run_file_based(self) -> str:
        """Run file-based Delta R calculation with output to .npy file."""
        src = self._safe_path(self.input_file)
        if not src:
            return self.format_error(
                error="Access Denied",
                reason="input_file escapes base_directory"
            )
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason=f"Input file not found: {self.input_file}"
            )

        # Auto-generate output file if not provided
        if self.output_file is None:
            input_basename = os.path.splitext(os.path.basename(self.input_file))[0]
            self.output_file = os.path.join(
                os.path.dirname(self.input_file) if os.path.dirname(self.input_file) else ".",
                f"{input_basename}_deltaR.npy"
            )

        dst = self._safe_path(self.output_file)
        if not dst:
            return self.format_error(
                error="Access Denied",
                reason="output_file escapes base_directory"
            )

        try:
            # Handle NumPy input
            if self.input_file.endswith('.npy'):
                data = np.load(src)

                if data.ndim == 2:
                    # Single event: (N_particles, features)
                    mask = np.any(data != 0, axis=1)
                    particles = data[mask]

                    # Generate pairs if not provided
                    if self.particle_pairs is None:
                        # All pairs
                        pairs = [[i, j] for i in range(len(particles)) for j in range(i+1, len(particles))]
                    else:
                        pairs = self.particle_pairs

                    deltaR_values = []
                    for i, j in pairs:
                        if i >= len(particles) or j >= len(particles):
                            continue
                        dr = self._calculate_delta_r(particles[i], particles[j])
                        deltaR_values.append(dr)

                    deltaR_array = np.array([deltaR_values])  # Shape: (1, N_pairs)

                elif data.ndim == 3:
                    # Multiple events: (N_events, N_particles, features)
                    if self.event_index is not None:
                        if self.event_index >= len(data):
                            return self.format_error(
                                error="Index Error",
                                reason=f"event_index {self.event_index} out of range [0, {len(data)-1}]"
                            )
                        data = data[self.event_index:self.event_index+1]

                    deltaR_list = []
                    for event in tqdm(data, desc="Calculating Delta R", unit="evt", **TQDM_CONFIG):
                        mask = np.any(event != 0, axis=1)
                        particles = event[mask]

                        # Generate pairs if not provided
                        if self.particle_pairs is None:
                            pairs = [[i, j] for i in range(len(particles)) for j in range(i+1, len(particles))]
                        else:
                            pairs = self.particle_pairs

                        event_deltaR = []
                        for i, j in pairs:
                            if i >= len(particles) or j >= len(particles):
                                continue
                            dr = self._calculate_delta_r(particles[i], particles[j])
                            event_deltaR.append(dr)
                        deltaR_list.append(event_deltaR)

                    # Pad to same length
                    max_len = max(len(dr) for dr in deltaR_list) if deltaR_list else 0
                    if max_len > 0:
                        deltaR_array = np.array([
                            np.pad(dr, (0, max_len - len(dr)), constant_values=0)
                            for dr in deltaR_list
                        ])
                    else:
                        deltaR_array = np.array([])
                else:
                    return self.format_error(
                        error="Invalid Shape",
                        reason=f"Expected 2D or 3D array, got shape {data.shape}"
                    )

            # Handle JSONL input
            elif self.input_file.endswith('.jsonl'):
                with open(src, 'r') as f:
                    events = [json.loads(line) for line in f]

                if self.event_index is not None:
                    if self.event_index >= len(events):
                        return self.format_error(
                            error="Index Error",
                            reason=f"event_index {self.event_index} out of range [0, {len(events)-1}]"
                        )
                    events = [events[self.event_index]]

                deltaR_list = []
                for ev in tqdm(events, desc="Calculating Delta R", unit="evt", **TQDM_CONFIG):
                    if "data" not in ev or "particles" not in ev["data"]:
                        return self.format_error(
                            error="Invalid Format",
                            reason="Event missing 'data.particles' key"
                        )

                    particles_data = ev["data"]["particles"]
                    particles_array = np.array([
                        [p["px"], p["py"], p["pz"], p["E"]]
                        for p in particles_data
                    ])

                    # Generate pairs if not provided
                    if self.particle_pairs is None:
                        pairs = [[i, j] for i in range(len(particles_array)) for j in range(i+1, len(particles_array))]
                    else:
                        pairs = self.particle_pairs

                    event_deltaR = []
                    for i, j in pairs:
                        if i >= len(particles_array) or j >= len(particles_array):
                            continue
                        dr = self._calculate_delta_r(particles_array[i], particles_array[j])
                        event_deltaR.append(dr)
                    deltaR_list.append(event_deltaR)

                # Pad to same length
                max_len = max(len(dr) for dr in deltaR_list) if deltaR_list else 0
                if max_len > 0:
                    deltaR_array = np.array([
                        np.pad(dr, (0, max_len - len(dr)), constant_values=0)
                        for dr in deltaR_list
                    ])
                else:
                    deltaR_array = np.array([])
            else:
                return self.format_error(
                    error="Invalid Format",
                    reason="Input file must be .npy or .jsonl"
                )

            # Save output
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            save_path = dst[:-4] if dst.endswith('.npy') else dst
            np.save(save_path, deltaR_array)

            # Build result
            result = {
                "status": "ok",
                "output_file": os.path.relpath(dst if dst.endswith('.npy') else dst + '.npy', self.base_directory),
                "n_events": len(deltaR_array),
                "output_shape": list(deltaR_array.shape)
            }

            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=str(e)
            )

    def _run(self) -> str:
        """Run the Delta R calculation."""
        # Priority 1: File-based I/O (recommended approach)
        if self.input_file is not None:
            return self._run_file_based()

        # Priority 2: Legacy single particle pair
        if self.particle1 is not None and self.particle2 is not None:
            try:
                p1 = np.array(self.particle1)
                p2 = np.array(self.particle2)

                if len(p1) < 3 or len(p2) < 3:
                    return self.format_error(
                        error="Invalid Input",
                        reason="Particles must have at least [px, py, pz]"
                    )

                delta_r = self._calculate_delta_r(p1, p2)

                result = {
                    "status": "ok",
                    "delta_r": delta_r
                }
                return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
            except Exception as e:
                return self.format_error(
                    error="Calculation Error",
                    reason=str(e)
                )

        return self.format_error(
            error="Missing Input",
            reason="Must provide input_file or both particle1 and particle2 (legacy)"
        )


class ApplyCutsTool(BaseTool):
    """
    Apply kinematic cuts to particles from numpy array or JSONL.

    Available cuts:
    - pt_min: minimum transverse momentum
    - pt_max: maximum transverse momentum
    - eta_min: minimum pseudorapidity
    - eta_max: maximum pseudorapidity (absolute value)
    - pdgids: list of allowed PDG IDs (JSONL only)

    Saves filtered results to output file.
    """
    # --------------------------- Runtime fields --------------------------- #
    input_path: str = RuntimeField(description="Path to input .npy or .jsonl file")
    output_path: str = RuntimeField(description="Path to save filtered output")
    pt_min: Optional[float] = RuntimeField(default=None, description="Minimum pT cut (GeV)")
    pt_max: Optional[float] = RuntimeField(default=None, description="Maximum pT cut (GeV)")
    eta_max: Optional[float] = RuntimeField(default=None, description="Maximum |eta| cut")
    eta_min: Optional[float] = RuntimeField(default=None, description="Minimum eta cut")
    pdgids: Optional[List[int]] = RuntimeField(default=None, description="Allowed PDG IDs (JSONL only)")
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
        if not rel:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    def _passes_cuts(self, px: float, py: float, pz: float) -> bool:
        """Check if a particle passes kinematic cuts."""
        pt = np.sqrt(px**2 + py**2)

        if self.pt_min is not None and pt < self.pt_min:
            return False
        if self.pt_max is not None and pt > self.pt_max:
            return False

        if self.eta_min is not None or self.eta_max is not None:
            eta = np.arcsinh(pz / pt) if pt > 0 else 0.0
            if self.eta_min is not None and eta < self.eta_min:
                return False
            if self.eta_max is not None and abs(eta) > self.eta_max:
                return False

        return True

    def _run(self) -> str:
        """Run the cuts application."""
        src = self._safe_path(self.input_path)
        dst = self._safe_path(self.output_path)

        if not src or not dst:
            return self.format_error(
                error="Access Denied",
                reason="Input or output path escapes base_directory"
            )

        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason=f"Input file not found: {self.input_path}"
            )

        try:
            # Handle JSONL input
            if self.input_path.endswith('.jsonl'):
                with open(src, 'r') as f:
                    events = [json.loads(line) for line in f]

                filtered_events = []
                total_before = 0
                total_after = 0

                for ev in tqdm(events, desc="Applying cuts", unit="evt", **TQDM_CONFIG):
                    # Check if event has expected structure
                    if "data" not in ev:
                        return self.format_error(
                            error="Invalid Format",
                            reason="Event missing 'data' key. Expected structure: {\"data\": {\"particles\": [...]}}"
                        )
                    if "particles" not in ev["data"]:
                        return self.format_error(
                            error="Invalid Format",
                            reason="Event data missing 'particles' key"
                        )

                    particles = ev["data"]["particles"]
                    total_before += len(particles)

                    filtered_particles = []
                    for p in particles:
                        # PDG ID filter
                        if self.pdgids is not None and p["id"] not in self.pdgids:
                            continue

                        # Kinematic cuts
                        if not self._passes_cuts(p["px"], p["py"], p["pz"]):
                            continue

                        filtered_particles.append(p)

                    total_after += len(filtered_particles)

                    # Update event with filtered particles
                    ev_copy = ev.copy()
                    ev_copy["data"] = ev["data"].copy()
                    ev_copy["data"]["particles"] = filtered_particles
                    ev_copy["data"]["n_particles"] = len(filtered_particles)
                    filtered_events.append(ev_copy)

                # Save filtered events
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, 'w') as f:
                    for ev in filtered_events:
                        f.write(json.dumps(ev, separators=(",", ":")) + "\n")

                result = {
                    "status": "ok",
                    "input_events": len(events),
                    "output_events": len(filtered_events),
                    "particles_before": total_before,
                    "particles_after": total_after,
                    "efficiency": total_after / total_before if total_before > 0 else 0,
                    "output_path": os.path.relpath(dst, self.base_directory)
                }

            # Handle NumPy input
            elif self.input_path.endswith('.npy'):
                data = np.load(src)

                # Apply cuts event-by-event
                filtered_events = []
                total_before = 0
                total_after = 0

                if data.ndim == 2:
                    # Single event
                    data = data[np.newaxis, :, :]

                for event in tqdm(data, desc="Applying cuts", unit="evt", **TQDM_CONFIG):
                    # Remove padding
                    mask = np.any(event != 0, axis=1)
                    particles = event[mask]
                    total_before += len(particles)

                    # Apply cuts
                    kept_mask = np.array([
                        self._passes_cuts(p[0], p[1], p[2])
                        for p in particles
                    ])
                    filtered_particles = particles[kept_mask]
                    total_after += len(filtered_particles)

                    filtered_events.append(filtered_particles)

                # Re-pad to max length
                max_len = max(len(ev) for ev in filtered_events) if filtered_events else 0
                if max_len > 0:
                    n_features = filtered_events[0].shape[1]
                    padded = np.array([
                        np.vstack([ev, np.zeros((max_len - len(ev), n_features))])
                        for ev in filtered_events
                    ])
                else:
                    padded = np.array([])

                # Save
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                save_path = dst[:-4] if dst.endswith('.npy') else dst
                np.save(save_path, padded)

                result = {
                    "status": "ok",
                    "input_events": len(data),
                    "output_shape": list(padded.shape),
                    "particles_before": total_before,
                    "particles_after": total_after,
                    "efficiency": total_after / total_before if total_before > 0 else 0,
                    "output_path": os.path.relpath(dst, self.base_directory)
                }
            else:
                return self.format_error(
                    error="Invalid Format",
                    reason="Input must be .npy or .jsonl file"
                )

            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=str(e)
            )


# ====================================================================== #
# =================== Event Selection Tools ============================ #
# ====================================================================== #

class GetHardestNTool(BaseTool):
    """
    Extract the N hardest (highest pT) particles from events.

    Accepts:
    - Path to .npy or .jsonl file
    - N: number of hardest particles to keep
    - Optional PDG ID filter

    Saves results to output file.
    """
    # --------------------------- Runtime fields --------------------------- #
    input_path: str = RuntimeField(description="Path to input .npy or .jsonl file")
    output_path: str = RuntimeField(description="Path to save output")
    n_hardest: int = RuntimeField(description="Number of hardest particles to keep")
    pdgids: Optional[list] = RuntimeField(default=None, description="Filter by PDG IDs first (JSONL only). If empty list, no filtering is applied.")
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
        if not rel:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    def _run(self) -> str:
        """Run the hardest-N selection."""
        src = self._safe_path(self.input_path)
        dst = self._safe_path(self.output_path)

        if not src or not dst:
            return self.format_error(
                error="Access Denied",
                reason="Input or output path escapes base_directory"
            )

        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason=f"Input file not found: {self.input_path}"
            )

        try:
            # Handle JSONL input
            if self.input_path.endswith('.jsonl'):
                with open(src, 'r') as f:
                    events = [json.loads(line) for line in f]

                selected_events = []

                for ev in tqdm(events, desc="Selecting hardest particles", unit="evt", **TQDM_CONFIG):
                    # Check if event has expected structure
                    if "data" not in ev:
                        return self.format_error(
                            error="Invalid Format",
                            reason="Event missing 'data' key. Expected structure: {\"data\": {\"particles\": [...]}}"
                        )
                    if "particles" not in ev["data"]:
                        return self.format_error(
                            error="Invalid Format",
                            reason="Event data missing 'particles' key"
                        )

                    particles = ev["data"]["particles"]

                    # Filter by PDG ID if requested (only if pdgids is not None and not empty)
                    if self.pdgids is not None and len(self.pdgids) > 0:
                        particles = [p for p in particles if p["id"] in self.pdgids]

                    # Calculate pT for each particle
                    particles_with_pt = [
                        (p, np.sqrt(p["px"]**2 + p["py"]**2))
                        for p in particles
                    ]
                
                    # Sort by pT descending
                    particles_with_pt.sort(key=lambda x: x[1], reverse=True)

                    # Keep top N
                    hardest = [p[0] for p in particles_with_pt[:self.n_hardest]]
                    
                    # Update event
                    ev_copy = ev.copy()
                    ev_copy["data"] = ev["data"].copy()
                    ev_copy["data"]["particles"] = hardest
                    ev_copy["data"]["n_particles"] = len(hardest)
                    selected_events.append(ev_copy)

                # Save
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, 'w') as f:
                    for ev in selected_events:
                        f.write(json.dumps(ev, separators=(",", ":")) + "\n")

                result = {
                    "status": "ok",
                    "input_events": len(events),
                    "n_hardest": self.n_hardest,
                    "output_path": os.path.relpath(dst, self.base_directory)
                }

            # Handle NumPy input
            elif self.input_path.endswith('.npy'):
                data = np.load(src)

                selected_events = []

                if data.ndim == 2:
                    data = data[np.newaxis, :, :]

                for event in tqdm(data, desc="Selecting hardest particles", unit="evt", **TQDM_CONFIG):
                    # Remove padding
                    mask = np.any(event != 0, axis=1)
                    particles = event[mask]

                    # Calculate pT
                    pt = np.sqrt(particles[:, 0]**2 + particles[:, 1]**2)

                    # Get indices of N hardest
                    if len(pt) <= self.n_hardest:
                        hardest = particles
                    else:
                        hardest_indices = np.argpartition(pt, -self.n_hardest)[-self.n_hardest:]
                        # Sort by pT descending
                        hardest_indices = hardest_indices[np.argsort(pt[hardest_indices])[::-1]]
                        hardest = particles[hardest_indices]

                    selected_events.append(hardest)

                # Re-pad
                max_len = min(self.n_hardest, max(len(ev) for ev in selected_events) if selected_events else 0)
                if max_len > 0:
                    n_features = selected_events[0].shape[1]
                    padded = np.array([
                        np.vstack([ev, np.zeros((max_len - len(ev), n_features))])
                        if len(ev) < max_len else ev[:max_len]
                        for ev in selected_events
                    ])
                else:
                    padded = np.array([])

                # Save
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                save_path = dst[:-4] if dst.endswith('.npy') else dst
                np.save(save_path, padded)

                result = {
                    "status": "ok",
                    "input_events": len(data),
                    "n_hardest": self.n_hardest,
                    "output_shape": list(padded.shape),
                    "output_path": os.path.relpath(dst, self.base_directory)
                }
            else:
                return self.format_error(
                    error="Invalid Format",
                    reason="Input must be .npy or .jsonl file"
                )

            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=str(e)
            )


class GetHardestNJetsTool(BaseTool):
    """
    Extract the N hardest (highest pT) jets from jet clustering output.

    Accepts:
    - Path to jets .jsonl file (from JetClusterSlowJetTool)
    - N: number of hardest jets to keep per event

    Saves results to output file.

    NOTE: This tool is for jets JSONL files with structure {"data": {"jets": [...]}}.
    For particle events with structure {"data": {"particles": [...]}}, use GetHardestNTool instead.
    """
    # --------------------------- Runtime fields --------------------------- #
    input_path: str = RuntimeField(description="Path to input jets .jsonl file")
    output_path: str = RuntimeField(description="Path to save output")
    n_hardest: int = RuntimeField(description="Number of hardest jets to keep per event")
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
        if not rel:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    def _run(self) -> str:
        """Run the hardest-N jets selection."""
        src = self._safe_path(self.input_path)
        dst = self._safe_path(self.output_path)

        if not src or not dst:
            return self.format_error(
                error="Access Denied",
                reason="Input or output path escapes base_directory"
            )

        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason=f"Input file not found: {self.input_path}"
            )

        if not self.input_path.endswith('.jsonl'):
            return self.format_error(
                error="Invalid Format",
                reason="Input must be .jsonl file"
            )

        try:
            with open(src, 'r') as f:
                events = [json.loads(line) for line in f]

            selected_events = []

            for ev in tqdm(events, desc="Selecting hardest jets", unit="evt", **TQDM_CONFIG):
                # Check if event has expected jets structure
                if "data" not in ev:
                    return self.format_error(
                        error="Invalid Format",
                        reason="Event missing 'data' key. Expected jets JSONL structure: {\"data\": {\"jets\": [...]}}. For particle events, use GetHardestNTool instead."
                    )
                if "jets" not in ev["data"]:
                    return self.format_error(
                        error="Invalid Format",
                        reason="Event data missing 'jets' key. Expected jets JSONL structure: {\"data\": {\"jets\": [...]}}"
                    )

                jets = ev["data"]["jets"]

                # Jets are already sorted by pT from JetClusterSlowJetTool
                # But let's re-sort to be safe
                jets_with_pt = [(jet, jet["pT"]) for jet in jets]
                jets_with_pt.sort(key=lambda x: x[1], reverse=True)

                # Keep top N
                hardest = [jet for jet, _ in jets_with_pt[:self.n_hardest]]

                # Update event
                ev_copy = ev.copy()
                ev_copy["data"] = ev["data"].copy()
                ev_copy["data"]["jets"] = hardest
                ev_copy["data"]["n_jets"] = len(hardest)
                selected_events.append(ev_copy)

            # Save
            dst_dir = os.path.dirname(dst)
            if dst_dir:
                os.makedirs(dst_dir, exist_ok=True)
            with open(dst, 'w') as f:
                for ev in selected_events:
                    f.write(json.dumps(ev, separators=(",", ":")) + "\n")

            result = {
                "status": "ok",
                "input_events": len(events),
                "n_hardest": self.n_hardest,
                "output_path": os.path.relpath(dst, self.base_directory)
            }

            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=str(e)
            )


class FilterByPDGIDTool(BaseTool):
    """
    Filter particles by PDG ID from JSONL files.

    Keeps only particles with PDG IDs in the specified list.
    """
    # --------------------------- Runtime fields --------------------------- #
    input_path: str = RuntimeField(description="Path to input .jsonl file")
    output_path: str = RuntimeField(description="Path to save filtered output")
    pdgids: List[int] = RuntimeField(description="List of PDG IDs to keep")
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
        if not rel:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    def _run(self) -> str:
        """Run the PDG ID filtering."""
        src = self._safe_path(self.input_path)
        dst = self._safe_path(self.output_path)

        if not src or not dst:
            return self.format_error(
                error="Access Denied",
                reason="Input or output path escapes base_directory"
            )

        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason=f"Input file not found: {self.input_path}"
            )

        try:
            with open(src, 'r') as f:
                events = [json.loads(line) for line in f]

            filtered_events = []
            total_before = 0
            total_after = 0

            for ev in tqdm(events, desc="Filtering by PDG ID", unit="evt", **TQDM_CONFIG):
                # Check if event has expected structure
                if "data" not in ev:
                    return self.format_error(
                        error="Invalid Format",
                        reason="Event missing 'data' key. Expected structure: {\"data\": {\"particles\": [...]}}"
                    )
                if "particles" not in ev["data"]:
                    return self.format_error(
                        error="Invalid Format",
                        reason="Event data missing 'particles' key"
                    )

                particles = ev["data"]["particles"]
                total_before += len(particles)

                filtered_particles = [p for p in particles if p["id"] in self.pdgids]
                total_after += len(filtered_particles)

                ev_copy = ev.copy()
                ev_copy["data"] = ev["data"].copy()
                ev_copy["data"]["particles"] = filtered_particles
                ev_copy["data"]["n_particles"] = len(filtered_particles)
                filtered_events.append(ev_copy)

            # Save
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, 'w') as f:
                for ev in filtered_events:
                    f.write(json.dumps(ev, separators=(",", ":")) + "\n")

            result = {
                "status": "ok",
                "input_events": len(events),
                "particles_before": total_before,
                "particles_after": total_after,
                "kept_pdgids": self.pdgids,
                "output_path": os.path.relpath(dst, self.base_directory)
            }

            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=str(e)
            )


class SortByPtTool(BaseTool):
    """
    Sort particles by transverse momentum (pT) in descending order.

    Modifies events so particles are ordered from highest to lowest pT.
    """
    # --------------------------- Runtime fields --------------------------- #
    input_path: str = RuntimeField(description="Path to input .npy or .jsonl file")
    output_path: str = RuntimeField(description="Path to save sorted output")
    ascending: bool = RuntimeField(default=False, description="Sort ascending instead of descending")
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
        if not rel:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    def _run(self) -> str:
        """Run the pT sorting."""
        src = self._safe_path(self.input_path)
        dst = self._safe_path(self.output_path)

        if not src or not dst:
            return self.format_error(
                error="Access Denied",
                reason="Input or output path escapes base_directory"
            )

        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason=f"Input file not found: {self.input_path}"
            )

        try:
            # Handle JSONL input
            if self.input_path.endswith('.jsonl'):
                with open(src, 'r') as f:
                    events = [json.loads(line) for line in f]

                sorted_events = []

                for ev in tqdm(events, desc="Sorting by pT", unit="evt", **TQDM_CONFIG):
                    # Check if event has expected structure
                    if "data" not in ev:
                        return self.format_error(
                            error="Invalid Format",
                            reason="Event missing 'data' key. Expected structure: {\"data\": {\"particles\": [...]}}"
                        )
                    if "particles" not in ev["data"]:
                        return self.format_error(
                            error="Invalid Format",
                            reason="Event data missing 'particles' key"
                        )

                    particles = ev["data"]["particles"]

                    # Calculate pT and sort
                    particles_with_pt = [
                        (p, np.sqrt(p["px"]**2 + p["py"]**2))
                        for p in particles
                    ]
                    particles_with_pt.sort(key=lambda x: x[1], reverse=not self.ascending)

                    sorted_particles = [p[0] for p in particles_with_pt]

                    ev_copy = ev.copy()
                    ev_copy["data"] = ev["data"].copy()
                    ev_copy["data"]["particles"] = sorted_particles
                    sorted_events.append(ev_copy)

                # Save
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, 'w') as f:
                    for ev in sorted_events:
                        f.write(json.dumps(ev, separators=(",", ":")) + "\n")

                result = {
                    "status": "ok",
                    "input_events": len(events),
                    "sort_order": "ascending" if self.ascending else "descending",
                    "output_path": os.path.relpath(dst, self.base_directory)
                }

            # Handle NumPy input
            elif self.input_path.endswith('.npy'):
                data = np.load(src)

                sorted_events = []

                if data.ndim == 2:
                    data = data[np.newaxis, :, :]

                for event in tqdm(data, desc="Sorting by pT", unit="evt", **TQDM_CONFIG):
                    # Remove padding
                    mask = np.any(event != 0, axis=1)
                    particles = event[mask]

                    if len(particles) > 0:
                        # Calculate pT and sort
                        pt = np.sqrt(particles[:, 0]**2 + particles[:, 1]**2)
                        sorted_indices = np.argsort(pt)
                        if not self.ascending:
                            sorted_indices = sorted_indices[::-1]
                        particles = particles[sorted_indices]

                    sorted_events.append(particles)

                # Re-pad
                max_len = max(len(ev) for ev in sorted_events) if sorted_events else 0
                if max_len > 0:
                    n_features = sorted_events[0].shape[1]
                    padded = np.array([
                        np.vstack([ev, np.zeros((max_len - len(ev), n_features))])
                        for ev in sorted_events
                    ])
                else:
                    padded = np.array([])

                # Save
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                save_path = dst[:-4] if dst.endswith('.npy') else dst
                np.save(save_path, padded)

                result = {
                    "status": "ok",
                    "input_events": len(data),
                    "sort_order": "ascending" if self.ascending else "descending",
                    "output_shape": list(padded.shape),
                    "output_path": os.path.relpath(dst, self.base_directory)
                }
            else:
                return self.format_error(
                    error="Invalid Format",
                    reason="Input must be .npy or .jsonl file"
                )

            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=str(e)
            )


class MergeObjectCollectionsTool(BaseTool):
    """
    Merge multiple object collections (particles, jets) into a single unified JSONL file.

    This is a generalized merging tool that handles arbitrary combinations of:
    - Truth particles (with PDG filtering)
    - Clustered jets (multiple radii/algorithms)
    - Any other physics objects in evtjsonl-1.0 format

    **Usage Examples:**

    **Example 1: Leptons + Jets (simple case)**
    ```python
    MergeObjectCollectionsTool(
        collections=[
            {
                "path": "data/pythia_events.jsonl",
                "type": "particles",
                "pdg_filter": [11, -11, 13, -13],  # electrons, muons
                "pdg_mapping": None  # Keep original PDG IDs
            },
            {
                "path": "data/jets_antikt04.jsonl",
                "type": "jets",
                "pdg_mapping": 0  # Assign PDG 0 to jets
            }
        ],
        output_path="data/leptons_jets.jsonl"
    )
    ```

    **Example 2: Multiple jet collections**
    ```python
    MergeObjectCollectionsTool(
        collections=[
            {
                "path": "data/pythia_events.jsonl",
                "type": "particles",
                "pdg_filter": [11, -11, 13, -13, 22],  # leptons + photons
                "pdg_mapping": None
            },
            {
                "path": "data/jets_R04.jsonl",
                "type": "jets",
                "pdg_mapping": 1000  # Small-R jets
            },
            {
                "path": "data/jets_R10.jsonl",
                "type": "jets",
                "pdg_mapping": 1001  # Large-R jets
            }
        ],
        output_path="data/all_objects.jsonl"
    )
    ```

    **Collection Dictionary Schema:**
    - `path` (str): Path to input JSONL file
    - `type` (str): "particles" or "jets"
    - `pdg_filter` (List[int], optional): For particles, which PDG IDs to keep
    - `pdg_mapping` (int or None):
        - For jets: pseudo-PDG ID to assign (e.g., 0, 1000, 1001)
        - For particles: if provided, remaps all to this PDG (use None to preserve)
    - `label` (str, optional): Human-readable label for this collection

    **Output Format:**
    Standard evtjsonl-1.0 with all objects merged into "particles" list, sorted by pT.

    **Advanced Features:**
    - Automatic event alignment checking
    - pT-sorted output
    - Collection statistics in return value
    - Preserves 4-momentum precision
    """

    # ========================== Runtime Fields ============================ #
    collections: List[dict] = RuntimeField(
        description="List of collection dictionaries to merge. Each must have 'path', 'type', and optional 'pdg_filter', 'pdg_mapping', 'label'."
    )
    output_path: str = RuntimeField(
        description="Path to output merged JSONL file"
    )
    sort_by_pt: bool = RuntimeField(
        default=True,
        description="Sort merged objects by pT descending (default: True)"
    )
    # ---------------------------------------------------------------------- #

    # ========================== State Fields ============================== #
    base_directory: str = StateField(
        default=".",
        description="Base directory for safe path resolution"
    )
    # ---------------------------------------------------------------------- #

    def _setup(self):
        """Setup and validate base directory."""
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.exists(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _safe_path(self, rel: str) -> Optional[str]:
        """Ensures that the path is within the allowed base directory."""
        if not rel:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    def _run(self):
        """Merge multiple object collections into a single JSONL file."""
        try:
            # Validate collections parameter
            if not self.collections or len(self.collections) == 0:
                return self.format_error(
                    error="Invalid Input",
                    reason="collections list is empty",
                    suggestion="Provide at least one collection to merge"
                )

            # Validate output path
            output_full = self._safe_path(self.output_path)
            if not output_full:
                return self.format_error(
                    error="Path Security Violation",
                    reason=f"output_path outside base directory: {self.output_path}"
                )

            # Load all collections
            all_events = []
            collection_info = []

            for coll_idx, coll in enumerate(self.collections):
                # Validate collection structure
                if "path" not in coll:
                    return self.format_error(
                        error="Invalid Collection",
                        reason=f"Collection {coll_idx} missing 'path' key"
                    )
                if "type" not in coll:
                    return self.format_error(
                        error="Invalid Collection",
                        reason=f"Collection {coll_idx} missing 'type' key"
                    )
                if coll["type"] not in ["particles", "jets"]:
                    return self.format_error(
                        error="Invalid Collection Type",
                        reason=f"Collection {coll_idx} has type '{coll['type']}', must be 'particles' or 'jets'"
                    )

                # Validate and load path
                coll_path = self._safe_path(coll["path"])
                if not coll_path or not os.path.exists(coll_path):
                    return self.format_error(
                        error="Invalid Path",
                        reason=f"Collection {coll_idx} path not found: {coll['path']}"
                    )

                # Load events from this collection
                with open(coll_path, 'r') as f:
                    events = [json.loads(line) for line in f]

                all_events.append({
                    "events": events,
                    "type": coll["type"],
                    "pdg_filter": coll.get("pdg_filter", None),
                    "pdg_mapping": coll.get("pdg_mapping", None),
                    "label": coll.get("label", f"{coll['type']}_{coll_idx}")
                })

                collection_info.append({
                    "index": coll_idx,
                    "path": coll["path"],
                    "type": coll["type"],
                    "n_events": len(events),
                    "label": coll.get("label", f"{coll['type']}_{coll_idx}")
                })

            # Check all collections have same number of events
            n_events = len(all_events[0]["events"])
            for coll_idx, coll_data in enumerate(all_events):
                if len(coll_data["events"]) != n_events:
                    return self.format_error(
                        error="Event Count Mismatch",
                        reason=f"Collection {coll_idx} has {len(coll_data['events'])} events, expected {n_events}"
                    )

            # Merge events
            merged_events = []
            stats_per_collection = [[] for _ in range(len(all_events))]

            for event_idx in tqdm(range(n_events), desc="Merging events", unit="evt", **TQDM_CONFIG):
                combined_objects = []

                # Process each collection for this event
                for coll_idx, coll_data in enumerate(all_events):
                    event = coll_data["events"][event_idx]

                    if coll_data["type"] == "particles":
                        # Extract particles
                        if "data" not in event or "particles" not in event["data"]:
                            return self.format_error(
                                error="Invalid Format",
                                reason=f"Event {event_idx} in collection {coll_idx} missing 'data.particles'"
                            )

                        particles = event["data"]["particles"]

                        # Apply PDG filter if specified
                        if coll_data["pdg_filter"] is not None:
                            particles = [p for p in particles if p["id"] in coll_data["pdg_filter"]]

                        # Convert to standard format
                        for p in particles:
                            obj = {
                                "px": p["px"],
                                "py": p["py"],
                                "pz": p["pz"],
                                "E": p["E"],
                                "m": p.get("m", 0.0)
                            }
                            # Apply PDG mapping if specified, otherwise keep original
                            if coll_data["pdg_mapping"] is not None:
                                obj["id"] = coll_data["pdg_mapping"]
                            else:
                                obj["id"] = p["id"]

                            combined_objects.append(obj)

                        stats_per_collection[coll_idx].append(len(particles))

                    elif coll_data["type"] == "jets":
                        # Extract jets
                        if "data" not in event or "jets" not in event["data"]:
                            return self.format_error(
                                error="Invalid Format",
                                reason=f"Event {event_idx} in collection {coll_idx} missing 'data.jets'"
                            )

                        jets = event["data"]["jets"]

                        # Convert jets to particle format
                        for j in jets:
                            obj = {
                                "px": j["px"],
                                "py": j["py"],
                                "pz": j["pz"],
                                "E": j["E"],
                                "m": j.get("m", 0.0),
                                "id": coll_data["pdg_mapping"] if coll_data["pdg_mapping"] is not None else 0
                            }
                            combined_objects.append(obj)

                        stats_per_collection[coll_idx].append(len(jets))

                # Sort by pT if requested
                if self.sort_by_pt:
                    for obj in combined_objects:
                        obj["pt"] = np.sqrt(obj["px"]**2 + obj["py"]**2)
                    combined_objects.sort(key=lambda x: x["pt"], reverse=True)
                    # Remove temporary pt field
                    for obj in combined_objects:
                        del obj["pt"]

                # Reindex
                for i, obj in enumerate(combined_objects):
                    obj["i"] = i

                # Create merged event
                merged_event = {
                    "schema": "evtjsonl-1.0",
                    "event_id": event_idx,
                    "data": {
                        "n_particles": len(combined_objects),
                        "particles": combined_objects
                    }
                }
                merged_events.append(merged_event)

            # Create output directory if needed
            output_dir = os.path.dirname(output_full)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Write output
            with open(output_full, 'w') as f:
                for ev in merged_events:
                    f.write(json.dumps(ev) + "\n")

            # Compute statistics
            collection_stats = []
            for coll_idx, counts in enumerate(stats_per_collection):
                collection_stats.append({
                    "collection_index": coll_idx,
                    "label": collection_info[coll_idx]["label"],
                    "type": collection_info[coll_idx]["type"],
                    "avg_objects_per_event": float(np.mean(counts)) if counts else 0.0,
                    "total_objects": int(np.sum(counts)) if counts else 0
                })

            # Return success
            result = {
                "status": "ok",
                "output_path": self.output_path,
                "n_events": n_events,
                "n_collections": len(all_events),
                "collections": collection_info,
                "stats": {
                    "per_collection": collection_stats,
                    "avg_total_objects_per_event": float(np.mean([
                        len(ev["data"]["particles"]) for ev in merged_events
                    ]))
                }
            }

            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=str(e)
            )


class FilterByDeltaRTool(BaseTool):
    """
    Remove objects from event arrays based on Delta R proximity to other objects.

    This tool performs overlap removal and applies isolation criteria by permanently
    removing objects that are too close (Delta R < threshold) to objects in other arrays.

    **WARNING**: This tool permanently removes objects from your dataset. Do not use
    this tool for resonance reconstruction where you need to pair objects from different
    arrays (e.g., leptoquarks S → ℓj). Removing objects will prevent mass reconstruction.
    For resonance pairing with Delta R constraints, use ResonanceReconstructionTool with
    the min_delta_r parameter instead.

    **Common use cases:**
    - **Jet-lepton overlap removal**: Remove jets near isolated leptons in analyses
      where they originate from different sources (e.g., W+jets: leptons from W decay,
      jets from QCD radiation)
    - **Lepton/photon isolation**: Remove non-isolated leptons or photons that are
      too close to hadronic activity
    - **Double-counting prevention**: Ensure objects from different reconstruction
      algorithms don't overlap

    **Input Schema:**

    Required:
    - particle_arrays: List of paths to JSONL files containing pre-selected objects
      * Example: ["leptons.jsonl", "jets.jsonl"]
      * Each file must use evtjsonl-1.0 schema with "particles" or "jets" key
      * All files must have the same number of events
    - delta_r_threshold: Delta R separation threshold
    - filter_mode: How to apply the filter
      * "remove_second": Remove objects from second array if Delta R < threshold
      * "remove_first": Remove objects from first array if Delta R < threshold
      * "remove_both": Remove pairs from both arrays if Delta R < threshold
      * "keep_only_separated": Keep only pairs where Delta R >= threshold

    Optional:
    - output_paths: List of output paths (default: auto-generated with "_filtered" suffix)
      * Must match length of particle_arrays
    - apply_to_arrays: Which arrays to filter (default: all)
      * Example: [1] to only filter second array
      * Example: [0, 1] to filter both arrays

    **Output Schema:**
    {
      "status": "ok",
      "delta_r_threshold": 0.4,
      "filter_mode": "remove_second",
      "n_events": 1000,
      "arrays": [
        {
          "index": 0,
          "input_path": "leptons.jsonl",
          "output_path": "leptons_filtered.jsonl",
          "objects_before": 2145,
          "objects_after": 2145,
          "efficiency": 1.0
        },
        {
          "index": 1,
          "input_path": "jets.jsonl",
          "output_path": "jets_filtered.jsonl",
          "objects_before": 4523,
          "objects_after": 3012,
          "efficiency": 0.666
        }
      ]
    }

    **Example 1: Lepton-jet overlap removal**
    ```python
    # Remove jets that overlap with leptons (Delta R < 0.4)
    FilterByDeltaRTool(
        particle_arrays=[
            "data/hardest_2_leptons.jsonl",
            "data/hardest_4_jets.jsonl"
        ],
        delta_r_threshold=0.4,
        filter_mode="remove_second",  # Remove jets, keep leptons
        output_paths=[
            "data/hardest_2_leptons_filtered.jsonl",
            "data/hardest_4_jets_filtered.jsonl"
        ]
    )
    ```

    **Example 2: Lepton isolation**
    ```python
    # Remove leptons that are too close to each other
    FilterByDeltaRTool(
        particle_arrays=["data/all_leptons.jsonl"],
        delta_r_threshold=0.3,
        filter_mode="remove_both",  # Remove both leptons in close pairs
        output_paths=["data/isolated_leptons.jsonl"]
    )
    ```

    **Example 3: Photon-jet separation**
    ```python
    # Keep only photons that are separated from jets
    FilterByDeltaRTool(
        particle_arrays=[
            "data/photons.jsonl",
            "data/jets.jsonl"
        ],
        delta_r_threshold=0.4,
        filter_mode="remove_first",  # Remove photons near jets
        output_paths=["data/isolated_photons.jsonl", "data/jets.jsonl"]
    )
    ```
    """

    # ========================== Runtime Fields ============================ #
    particle_arrays: List[str] = RuntimeField(
        description="List of paths to JSONL files containing particles/jets. All must have same number of events."
    )
    delta_r_threshold: float = RuntimeField(
        description="Delta R separation threshold. Objects with Delta R < threshold are affected by filter."
    )
    filter_mode: str = RuntimeField(
        description="Filter mode: 'remove_second', 'remove_first', 'remove_both', or 'keep_only_separated'"
    )
    output_paths: Optional[List[str]] = RuntimeField(
        default=None,
        description="List of output paths (default: auto-generated with '_filtered' suffix). Must match length of particle_arrays."
    )
    apply_to_arrays: Optional[List[int]] = RuntimeField(
        default=None,
        description="Indices of arrays to apply filtering to (default: all arrays)"
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

    def _calculate_delta_r(self, eta1: float, phi1: float, eta2: float, phi2: float) -> float:
        """Calculate Delta R between two particles."""
        delta_eta = eta1 - eta2
        delta_phi = phi1 - phi2

        # Wrap delta_phi to [-pi, pi]
        while delta_phi > np.pi:
            delta_phi -= 2 * np.pi
        while delta_phi < -np.pi:
            delta_phi += 2 * np.pi

        return float(np.sqrt(delta_eta**2 + delta_phi**2))

    def _calculate_eta_phi(self, px: float, py: float, pz: float) -> tuple:
        """Calculate pseudorapidity and azimuthal angle."""
        pt = np.sqrt(px**2 + py**2)
        eta = np.arcsinh(pz / pt) if pt > 0 else 0.0
        phi = np.arctan2(py, px)
        return eta, phi

    def _load_arrays(self, array_paths: List[str]) -> List[List[dict]]:
        """
        Load multiple particle arrays from JSONL files.

        Args:
            array_paths: List of paths to JSONL files

        Returns:
            List of event lists, one per array
        """
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

        return all_arrays

    def _extract_objects(self, event: dict) -> List[dict]:
        """
        Extract objects (particles or jets) from an event.

        Args:
            event: Event dictionary with 'data' key

        Returns:
            List of objects with px, py, pz, E, and computed eta, phi
        """
        if "data" not in event:
            raise ValueError("Event missing 'data' key")

        # Try both 'particles' and 'jets' keys
        objects = None
        if "particles" in event["data"]:
            objects = event["data"]["particles"]
        elif "jets" in event["data"]:
            objects = event["data"]["jets"]
        else:
            raise ValueError("Event has neither 'data.particles' nor 'data.jets' key")

        # Compute eta, phi for each object
        objects_with_coords = []
        for obj in objects:
            eta, phi = self._calculate_eta_phi(obj["px"], obj["py"], obj["pz"])
            obj_copy = obj.copy()
            obj_copy["_eta"] = eta
            obj_copy["_phi"] = phi
            objects_with_coords.append(obj_copy)

        return objects_with_coords

    def _run(self) -> str:
        """Run the Delta R filtering."""
        try:
            # Validate inputs
            if not self.particle_arrays or len(self.particle_arrays) == 0:
                return self.format_error(
                    error="Invalid Input",
                    reason="particle_arrays is empty. Provide at least one array path."
                )

            valid_modes = ["remove_second", "remove_first", "remove_both", "keep_only_separated"]
            if self.filter_mode not in valid_modes:
                return self.format_error(
                    error="Invalid Filter Mode",
                    reason=f"filter_mode must be one of {valid_modes}, got '{self.filter_mode}'"
                )

            # Auto-generate output paths if not provided
            if self.output_paths is None:
                self.output_paths = []
                for path in self.particle_arrays:
                    basename = os.path.splitext(path)[0]
                    ext = os.path.splitext(path)[1]
                    self.output_paths.append(f"{basename}_filtered{ext}")
            else:
                if len(self.output_paths) != len(self.particle_arrays):
                    return self.format_error(
                        error="Invalid Output Paths",
                        reason=f"output_paths length ({len(self.output_paths)}) must match particle_arrays length ({len(self.particle_arrays)})"
                    )

            # Load arrays
            print(f"Loading {len(self.particle_arrays)} particle array(s)...")
            all_arrays = self._load_arrays(self.particle_arrays)
            n_events = len(all_arrays[0])
            print(f"Loaded {n_events} events")

            # Initialize filtered arrays (start with all objects)
            filtered_arrays = []
            for array in all_arrays:
                filtered_arrays.append([[] for _ in range(n_events)])

            # Statistics tracking
            objects_before = [0] * len(all_arrays)
            objects_after = [0] * len(all_arrays)

            # Process events
            for event_idx in tqdm(range(n_events), desc="Filtering by Delta R", unit="evt", **TQDM_CONFIG):
                # Extract objects from all arrays for this event
                event_objects = []
                for array_idx, array in enumerate(all_arrays):
                    objects = self._extract_objects(array[event_idx])
                    event_objects.append(objects)
                    objects_before[array_idx] += len(objects)

                # Build removal sets for each array
                to_remove = [set() for _ in range(len(all_arrays))]

                # For single array: compute pairwise Delta R within the array
                if len(all_arrays) == 1:
                    objects = event_objects[0]
                    for i in range(len(objects)):
                        for j in range(i + 1, len(objects)):
                            dr = self._calculate_delta_r(
                                objects[i]["_eta"], objects[i]["_phi"],
                                objects[j]["_eta"], objects[j]["_phi"]
                            )
                            if dr < self.delta_r_threshold:
                                if self.filter_mode == "remove_both":
                                    to_remove[0].add(i)
                                    to_remove[0].add(j)
                                elif self.filter_mode == "keep_only_separated":
                                    to_remove[0].add(i)
                                    to_remove[0].add(j)

                # For multiple arrays: compute Delta R between arrays
                else:
                    for i, array1_idx in enumerate(range(len(all_arrays))):
                        for j, array2_idx in enumerate(range(i + 1, len(all_arrays)), start=i+1):
                            objects1 = event_objects[array1_idx]
                            objects2 = event_objects[array2_idx]

                            for idx1, obj1 in enumerate(objects1):
                                for idx2, obj2 in enumerate(objects2):
                                    dr = self._calculate_delta_r(
                                        obj1["_eta"], obj1["_phi"],
                                        obj2["_eta"], obj2["_phi"]
                                    )
                                    if dr < self.delta_r_threshold:
                                        if self.filter_mode == "remove_first":
                                            to_remove[array1_idx].add(idx1)
                                        elif self.filter_mode == "remove_second":
                                            to_remove[array2_idx].add(idx2)
                                        elif self.filter_mode == "remove_both":
                                            to_remove[array1_idx].add(idx1)
                                            to_remove[array2_idx].add(idx2)
                                        elif self.filter_mode == "keep_only_separated":
                                            to_remove[array1_idx].add(idx1)
                                            to_remove[array2_idx].add(idx2)

                # Apply filtering and build filtered events
                for array_idx in range(len(all_arrays)):
                    objects = event_objects[array_idx]
                    filtered_objects = []

                    for obj_idx, obj in enumerate(objects):
                        # Check if this array should be filtered
                        if self.apply_to_arrays is not None and array_idx not in self.apply_to_arrays:
                            # Don't filter this array, keep all objects
                            obj_clean = obj.copy()
                            del obj_clean["_eta"]
                            del obj_clean["_phi"]
                            filtered_objects.append(obj_clean)
                        else:
                            # Apply filtering
                            if obj_idx not in to_remove[array_idx]:
                                obj_clean = obj.copy()
                                del obj_clean["_eta"]
                                del obj_clean["_phi"]
                                filtered_objects.append(obj_clean)

                    # Reindex particles
                    for new_idx, obj in enumerate(filtered_objects):
                        obj["i"] = new_idx

                    filtered_arrays[array_idx][event_idx] = filtered_objects
                    objects_after[array_idx] += len(filtered_objects)

            # Save filtered arrays
            array_stats = []
            for array_idx in range(len(all_arrays)):
                output_path = self.output_paths[array_idx]
                safe_output = self._safe_path(output_path)
                if not safe_output:
                    return self.format_error(
                        error="Access Denied",
                        reason=f"output_path {array_idx} escapes base_directory: {output_path}"
                    )

                # Create output directory if needed
                output_dir = os.path.dirname(safe_output)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)

                # Write filtered events
                with open(safe_output, 'w') as f:
                    for event_idx in range(n_events):
                        # Reconstruct event structure
                        original_event = all_arrays[array_idx][event_idx]
                        filtered_event = original_event.copy()
                        filtered_event["data"] = original_event["data"].copy()

                        # Determine if this is particles or jets
                        if "particles" in original_event["data"]:
                            filtered_event["data"]["particles"] = filtered_arrays[array_idx][event_idx]
                            filtered_event["data"]["n_particles"] = len(filtered_arrays[array_idx][event_idx])
                        elif "jets" in original_event["data"]:
                            filtered_event["data"]["jets"] = filtered_arrays[array_idx][event_idx]
                            filtered_event["data"]["n_jets"] = len(filtered_arrays[array_idx][event_idx])

                        f.write(json.dumps(filtered_event, separators=(",", ":"), ensure_ascii=False) + "\n")

                # Compute statistics
                efficiency = objects_after[array_idx] / objects_before[array_idx] if objects_before[array_idx] > 0 else 1.0
                array_stats.append({
                    "index": array_idx,
                    "input_path": self.particle_arrays[array_idx],
                    "output_path": os.path.relpath(safe_output, self.base_directory),
                    "objects_before": objects_before[array_idx],
                    "objects_after": objects_after[array_idx],
                    "efficiency": float(efficiency)
                })

            # Build output
            result = {
                "status": "ok",
                "delta_r_threshold": self.delta_r_threshold,
                "filter_mode": self.filter_mode,
                "n_events": n_events,
                "arrays": array_stats
            }

            return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=str(e)
            )

