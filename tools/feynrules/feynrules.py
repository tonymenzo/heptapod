"""
# feynrules.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import os, json, subprocess
from pathlib import Path
from typing import Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

SCHEMA_VERSION = "tool-1.0"

def _utc_now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

# ====================================================================== #
# =================== FeynRules model \to UFO tool ===================== #
# ====================================================================== #

class FeynRulesToUFOTool(BaseTool):
    """
    Generate a UFO model directory from a FeynRules .fr model file by calling Mathematica's `wolframscript`.

    Inputs (runtime):
      - model_path: path to the FeynRules model file (.fr). Can be absolute or relative to base_directory.
      - output_dir: output directory where the UFO will be written. Relative to base_directory is recommended.
      - feynrules_path: optional path to the FeynRules installation root (directory that contains "FeynRules.m" and "Models/").
                        If omitted, the tool uses the environment variable FR_PATH, if set.
      - wolframscript_path: command or absolute path to the preferred wolframscript binary. Defaults to "wolframscript".
                           Use this to select a specific Mathematica version, e.g. "/Applications/Mathematica 13.3.app/Contents/MacOS/wolframscript".
      - log_dir: optional directory to store stdout/stderr logs. Defaults to {output_dir}/_logs.
      - timeout_sec: optional walltime limit for the Mathematica run. Defaults to 3600 seconds.

    Returns:
      JSON summary including paths, logs, and file listing of the generated UFO.
    """

    # --------------------------- Runtime fields --------------------------- #
    model_path: str = RuntimeField(default=None, description="Path to FeynRules .fr model file")
    output_dir: str = RuntimeField(default="UFO_Output", description="Directory for UFO output")
    log_dir: Optional[str] = RuntimeField(default=None, description="Directory to store logs")
    timeout_sec: Optional[int] = RuntimeField(default=3600, description="Timeout in seconds for Mathematica run")
    # ---------------------------------------------------------------------- #

    # ---------------------------- State fields ---------------------------- #
    feynrules_path: str = StateField(description="Path to FeynRules installation root")
    wolframscript_path: str = StateField(description="Command/path to wolframscript")
    base_directory: str = StateField(description="Base working directory for relative paths")
    # ---------------------------------------------------------------------- #

    def _abs_path(self, maybe_rel: Optional[str]) -> Optional[str]:
        """Convert a possibly relative path to an absolute path based on base_directory."""
        if maybe_rel is None:
            return None
        p = Path(maybe_rel)
        if not p.is_absolute():
            p = Path(self.base_directory) / p
        return str(p.resolve())

    def _ensure_dir(self, path: str) -> None:
        """Create directory if it doesn't exist."""
        Path(path).mkdir(parents=True, exist_ok=True)

    def _run(self) -> str:
        """Run the FeynRules to UFO conversion using wolframscript."""
        if not self.model_path:
            return self.format_error(error="Missing Parameter", reason="model_path is required")
        if not self.output_dir:
            return self.format_error(error="Missing Parameter", reason="output_dir is required")

        abs_model = self._abs_path(self.model_path)
        abs_out = self._abs_path(self.output_dir)
        abs_fr = self._abs_path(self.feynrules_path) if self.feynrules_path else None
        self._ensure_dir(abs_out)

        env = os.environ.copy()
        if abs_fr:
            env["FEYNRULES_PATH"] = abs_fr

        log_dir = self._abs_path(self.log_dir) if self.log_dir else str(Path(abs_out) / "_logs")
        self._ensure_dir(log_dir)
        stdout_path = str(Path(log_dir) / "wolframscript_stdout.log")
        stderr_path = str(Path(log_dir) / "wolframscript_stderr.log")

        # Assume .wl script is colocated with this file
        driver = Path(__file__).parent / "UFO_generator.wl"
        if not driver.exists():
            return self.format_error(error="Missing Script", reason=f"{driver} not found")

        # Build command
        cmd = [
            self.wolframscript_path, "-f", str(driver),
            f"ModelPath={abs_model}",
            f"FeynRulesPath={abs_fr or ''}",
            f"OutputDir={abs_out}",
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.base_directory,
                env=env,
                timeout=self.timeout_sec,
                check=False,
                text=True,
            )
        except subprocess.TimeoutExpired:
            return self.format_error(error="Timeout", reason=f"wolframscript exceeded {self.timeout_sec}s")
        except FileNotFoundError:
            return self.format_error(error="Executable Not Found", reason=f"wolframscript_path not found: {self.wolframscript_path}")

        Path(stdout_path).write_text(proc.stdout or "")
        Path(stderr_path).write_text(proc.stderr or "")

        ok = proc.returncode == 0
        files_created = []
        if ok and Path(abs_out).exists():
            files_created = sorted(p.name for p in Path(abs_out).iterdir())

        summary = {
            "schema_version": SCHEMA_VERSION,
            "ok": ok,
            "created_at_utc": _utc_now_iso(),
            "model_path": abs_model,
            "output_dir": abs_out,
            "wolframscript_path": self.wolframscript_path,
            "feynrules_path": abs_fr,
            "logs": {"stdout": stdout_path, "stderr": stderr_path},
            "files_created": files_created,
        }

        if not ok:
            hint = None
            for line in (proc.stderr or "").splitlines()[-20:]:
                if "Error" in line or "LoadModel" in line or "WriteUFO" in line:
                    hint = line.strip()
                    break
            return self.format_error(error="UFO Generation Failed", reason=hint or "wolframscript returned non-zero exit status")

        return json.dumps(summary, indent=2)