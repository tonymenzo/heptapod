"""
# validate_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Validate a generated FeynRules .fr model.

Compiles the .fr to a UFO via the existing FeynRulesToUFOTool (the deterministic
verifier), then runs structural checks: UFO generation succeeded, the expected
UFO Python files exist, and every declared new particle actually appears in the
UFO. A "failed" validation is a normal structured result (passed=False), not a
tool error — the driving agent uses it to run its repair loop. Deeper physics
checks (cross sections/widths) are left to MadGraph via the workflow.
"""

import json
import os
from typing import List, Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from tools.feynrules import FeynRulesToUFOTool
from tools.validate.ufo_parser import check_particle_properties
from tools.validate.width_gate import (
    analytic_scalar_lq_width,
    compare_width,
    find_partial_width,
    parse_decays,
    parse_external_params,
    safe_eval_width,
)

SCHEMA_VERSION = "model-validation-1.1"

EXPECTED_UFO_FILES = [
    "__init__.py",
    "particles.py",
    "parameters.py",
    "vertices.py",
    "couplings.py",
]

# UFO files scanned for particle-name presence.
_SCAN_FILES = ["particles.py", "parameters.py", "vertices.py", "couplings.py"]


def _safe_join(base_directory: str, rel_or_abs: str) -> Optional[str]:
    if not rel_or_abs:
        return None
    base = os.path.realpath(base_directory)
    full = os.path.realpath(os.path.join(base, rel_or_abs))
    if full != base and not full.startswith(base + os.sep):
        return None
    return full


def check_ufo_files(ufo_dir: str) -> List[dict]:
    """One check per expected UFO file: present or not."""
    return [
        {
            "name": f"ufo_file:{fname}",
            "passed": os.path.isfile(os.path.join(ufo_dir, fname)),
            "detail": fname,
        }
        for fname in EXPECTED_UFO_FILES
    ]


def _scan_ufo_text(ufo_dir: str) -> str:
    parts: List[str] = []
    for fname in _SCAN_FILES:
        path = os.path.join(ufo_dir, fname)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    parts.append(fh.read())
            except OSError:
                continue
    return "\n".join(parts)


def check_particles_present(ufo_dir: str, names: List[str]) -> List[dict]:
    """One check per name: does it appear anywhere in the UFO python files."""
    text = _scan_ufo_text(ufo_dir)
    checks: List[dict] = []
    for name in names:
        if not name:
            continue
        checks.append(
            {
                "name": f"particle_in_ufo:{name}",
                "passed": name in text,
                "detail": f"expected '{name}' in UFO output",
            }
        )
    return checks


def _expected_particle_names(feynrules_model_json: Optional[str]) -> List[str]:
    """Physical (non-unphysical) particle names to look for in the UFO."""
    if not feynrules_model_json:
        return []
    data = json.loads(feynrules_model_json)
    names: List[str] = []
    for p in data.get("particles", []):
        if p.get("unphysical"):
            continue
        # ClassName is the FeynRules symbol; particle_name is the MG5 label.
        for key in ("class_name", "particle_name"):
            v = p.get(key)
            if isinstance(v, str) and v:
                names.append(v)
                break
    return names


class ValidateModelTool(BaseTool):
    """
    Validate a FeynRules .fr model by compiling it to a UFO and checking it.

    Runs take minutes (Wolfram compile; MadGraph import when enabled). From
    an MCP agent, prefer submitjob(tool_name="validatemodel", ...) and poll
    jobstatus so the channel is never blocked.

    Runs FeynRulesToUFOTool on the .fr, then reports structured pass/fail checks:
      - ufo_generation: did the .fr compile to a UFO (Mathematica/FeynRules)?
      - ufo_file:<name>: are the expected UFO python files present?
      - particle_in_ufo:<name>: does each declared new particle appear in the UFO
        (when feynrules_model_json is provided)?

    Input:
        model_path: Path to the .fr file, relative to base_directory.
        output_dir: UFO output dir, relative to base_directory (default "UFO_validate").
        feynrules_model_json: Optional FeynRulesModel JSON (or the generator's
            output) used to cross-check that declared particles reached the UFO.
        timeout_sec: Walltime for the Mathematica run (default 1800).

    State:
        base_directory, feynrules_path, wolframscript_path.

    Returns JSON:
        {"status": "ok", "schema": "model-validation-1.0", "passed": bool,
         "checks": [{"name","passed","detail"}, ...], "ufo_dir": "...",
         "feynrules_log": "<stderr tail on failure>"}
    On a genuine tool error (e.g. model file missing) returns a formatted error.
    """

    # ======================== Runtime fields ======================== #
    model_path: str = RuntimeField(
        description="Path to the .fr model file, relative to base_directory"
    )
    output_dir: Optional[str] = RuntimeField(
        default="UFO_validate", description="UFO output dir relative to base_directory"
    )
    feynrules_model_json: Optional[str] = RuntimeField(
        default=None,
        description="Optional FeynRulesModel JSON to cross-check particle content",
    )
    timeout_sec: Optional[int] = RuntimeField(
        default=1800, description="Timeout for the Mathematica run (seconds)"
    )
    physics_checks: Optional[bool] = RuntimeField(
        default=True,
        description="Run FeynRules symmetry checks (Hermiticity, kinetic/mass "
        "terms) during UFO generation and surface them as wl:* checks",
    )
    width_gate: Optional[str] = RuntimeField(
        default=None,
        description="Optional JSON spec for an analytic decay-width cross-check, "
        'e.g. {"particle":"S1","finals":["e__minus__","u"],"formula":"scalar_lq",'
        '"mass_param":"MS1","coupling_param":"yRR11","rel_tol":0.02}. When set and '
        "the UFO built, compares the UFO partial width to the closed form and "
        "reports it as a width_gate:* check.",
    )
    madgraph_check: Optional[bool] = RuntimeField(
        default=False,
        description="Import the generated UFO into MadGraph5 (import model) and "
        "report whether it loads, as a madgraph:import check. Surfaces the real "
        "MadGraph error (undefined symbol, duplicate name, bad syntax) so an agent "
        "can repair the model. Requires mg5_path to be configured.",
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(description="Base sandbox directory")
    feynrules_path: str = StateField(description="Path to FeynRules installation root")
    wolframscript_path: str = StateField(description="Command/path to wolframscript")
    mg5_path: Optional[str] = StateField(
        default="", description="MG5_aMC install dir (for the optional MadGraph check)"
    )
    # ================================================================ #

    def _run(self) -> str:
        src = _safe_join(self.base_directory, self.model_path)
        if src is None:
            return self.format_error(
                error="Access Denied",
                reason="model_path escapes base_directory",
                context=self.model_path,
            )
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason=".fr model file does not exist",
                context=self.model_path,
            )

        # Optional content cross-check names (parse early so bad JSON surfaces).
        try:
            expected_names = _expected_particle_names(self.feynrules_model_json)
        except (json.JSONDecodeError, AttributeError) as e:
            return self.format_error(
                error="Invalid Input",
                reason=f"feynrules_model_json is not valid: {e}",
            )

        # Compile .fr -> UFO via the existing tool (the real verifier).
        ufo_tool = FeynRulesToUFOTool(
            base_directory=self.base_directory,
            feynrules_path=self.feynrules_path,
            wolframscript_path=self.wolframscript_path,
            model_path=self.model_path,
            output_dir=self.output_dir or "UFO_validate",
            timeout_sec=self.timeout_sec or 1800,
            run_checks=bool(self.physics_checks),
        )
        raw = ufo_tool._run()

        checks: List[dict] = []
        ufo_dir: Optional[str] = None
        wl_checks: List[dict] = []
        log_tail = ""
        try:
            ufo_res = json.loads(raw)
            ufo_ok = bool(ufo_res.get("ok"))
            ufo_dir = ufo_res.get("output_dir")
            wl_checks = ufo_res.get("checks") or []
        except json.JSONDecodeError:
            # FeynRulesToUFOTool returns a plain-string format_error on failure.
            ufo_ok = False
            log_tail = raw.strip().splitlines()[0] if raw.strip() else ""

        checks.append(
            {
                "name": "ufo_generation",
                "passed": ufo_ok,
                "detail": "compiled .fr -> UFO" if ufo_ok else (log_tail or "failed"),
            }
        )

        if ufo_ok and ufo_dir and os.path.isdir(ufo_dir):
            checks.extend(check_ufo_files(ufo_dir))
            if expected_names:
                checks.extend(check_particles_present(ufo_dir, expected_names))
            # Deep, property-level particle checks (spin/color/charge) from the
            # generated UFO vs the declared FeynRulesModel.
            if self.feynrules_model_json:
                try:
                    checks.extend(
                        check_particle_properties(
                            ufo_dir, json.loads(self.feynrules_model_json)
                        )
                    )
                except (json.JSONDecodeError, OSError):
                    pass
            # Analytic decay-width gate (opt-in via width_gate spec): compare a
            # UFO partial width to a textbook closed form to catch sign/norm errors.
            if self.width_gate:
                checks.append(self._width_gate_check(ufo_dir))
            # Full-chain: does the UFO actually load in MadGraph? (opt-in)
            if self.madgraph_check and self.mg5_path:
                checks.append(self._madgraph_import_check(ufo_dir))

        # Merge FeynRules symmetry checks (gauge invariance / Hermiticity) parsed
        # from the Mathematica run into named wl:* checks.
        for c in wl_checks:
            checks.append(
                {
                    "name": f"wl:{c.get('name', 'check')}",
                    "passed": bool(c.get("passed")),
                    "detail": c.get("detail", ""),
                }
            )

        passed = all(c["passed"] for c in checks) and ufo_ok

        result = {
            "status": "ok",
            "schema": SCHEMA_VERSION,
            "passed": passed,
            "checks": checks,
            "ufo_dir": (
                os.path.relpath(ufo_dir, os.path.realpath(self.base_directory))
                if ufo_dir and os.path.isdir(ufo_dir)
                else None
            ),
        }
        if not ufo_ok and log_tail:
            result["feynrules_log"] = log_tail
        return json.dumps(result, indent=2)

    def _width_gate_check(self, ufo_dir: str) -> dict:
        """Compare a UFO partial width to a textbook closed form (opt-in).

        Returns a single {name, passed, detail} check. Any spec/parse/eval
        problem is reported as a failed check rather than raised, so a bad gate
        request never turns the validation into a tool error.
        """
        analytic_formulas = {"scalar_lq": analytic_scalar_lq_width}
        try:
            spec = json.loads(self.width_gate)
        except json.JSONDecodeError as e:
            return {"name": "width_gate", "passed": False,
                    "detail": f"invalid width_gate JSON: {e}"}

        particle = spec.get("particle")
        finals = tuple(spec.get("finals") or ())
        formula = analytic_formulas.get(spec.get("formula"))
        label = (
            f"width_gate:{particle}->{'+'.join(finals)}" if particle and finals
            else "width_gate"
        )
        if not (particle and finals and formula):
            return {"name": label, "passed": False,
                    "detail": "spec needs particle, finals, and a known formula "
                    f"(one of {sorted(analytic_formulas)})"}
        try:
            params = parse_external_params(ufo_dir)
            decays = parse_decays(ufo_dir)
            expr = find_partial_width(decays, particle, finals)
            if expr is None:
                return {"name": label, "passed": False,
                        "detail": f"no UFO partial width for {particle} -> {finals}"}
            reference = safe_eval_width(expr, params)
            m_val = params[spec["mass_param"]]
            y_val = params[spec["coupling_param"]]
            analytic = formula(m_val, y_val)
            cmp = compare_width(analytic, reference, float(spec.get("rel_tol", 0.02)))
        except (KeyError, OSError, ValueError, ZeroDivisionError) as e:
            return {"name": label, "passed": False,
                    "detail": f"width gate error: {e}"}
        return {
            "name": label,
            "passed": bool(cmp["passed"]),
            "detail": (
                f"analytic {cmp['analytic']:.4g} vs UFO {cmp['reference']:.4g} GeV "
                f"(rel_err {cmp['rel_err']:.2%}, tol {cmp['rel_tol']:.0%})"
            ),
        }

    def _madgraph_import_check(self, ufo_dir: str) -> dict:
        """Import the UFO into MadGraph5 and report whether it loads. On failure,
        surface MadGraph's real diagnostic (from MG5_debug, not the misleading
        object_library.py wrapper) so an agent can repair the model."""
        import re
        import subprocess
        import tempfile

        mg5 = os.path.join(self.mg5_path, "bin", "mg5_aMC")
        if not os.path.isfile(mg5):
            return {"name": "madgraph:import", "passed": False,
                    "detail": f"mg5_aMC not found under mg5_path ({self.mg5_path})"}
        work = tempfile.mkdtemp(prefix="mg5imp_")
        cmd = os.path.join(work, "cmd.txt")
        with open(cmd, "w", encoding="utf-8") as fh:
            fh.write(f"import model {ufo_dir}\ndisplay particles\n")
        try:
            p = subprocess.run([mg5, cmd], capture_output=True, text=True,
                               timeout=300, stdin=subprocess.DEVNULL, cwd=work)
            out = (p.stdout or "") + "\n" + (p.stderr or "")
        except subprocess.TimeoutExpired:
            return {"name": "madgraph:import", "passed": False, "detail": "MadGraph import timed out"}
        loaded = re.search(r"Current model contains (\d+) particles", out)
        fatal = ("Traceback (most recent call last)" in out or "InvalidCmd" in out
                 or re.search(r'Command ".*" interrupted with error', out))
        if loaded and not fatal:
            return {"name": "madgraph:import", "passed": True,
                    "detail": f"UFO loaded in MadGraph ({loaded.group(1)} particles)"}
        # Prefer the real error from MG5_debug over the object_library.py wrapper.
        real = ""
        dbg = os.path.join(work, "MG5_debug")
        text = ""
        try:
            if os.path.isfile(dbg):
                text = open(dbg, encoding="utf-8", errors="replace").read()
        except OSError:
            text = ""
        for m in re.finditer(r"(?:models\.\S*Error|NameError|SyntaxError|InvalidModel|KeyError|ValueError)\s*:\s*(.+)",
                             text + "\n" + out):
            cand = m.group(1).strip()
            if "object_library.py, line 268" not in cand:
                real = cand
                break
        return {"name": "madgraph:import", "passed": False,
                "detail": ("MadGraph rejected the UFO: " + real) if real
                else "MadGraph did not load the model (see UFO parameters/couplings/lorentz)"}
