ClearAll["Global`*"];

(* =========== FeynRules UFO generator =========== *)
(*
   Usage examples:
   wolframscript -f GenerateUFO.wl \
        "ModelPath=<PATH_TO_MODEL.fr>" \
        "FeynRulesPath=<PATH_TO_FeynRules>" \
        "OutputDir=<PATH_TO_OUTPUT>"
*)
(* =============================================== *)

(* ---- Parse command-line args as key=value pairs. ---- *)
params = Association @ Map[
    Rule @@ StringSplit[#, "="] &,
    Rest[$ScriptCommandLine]
];

(* ---- Set defaults from env vars if not provided. ---- *)

If[!ValueQ[$FeynRulesPath],
  $FeynRulesPath = Lookup[params, "FeynRulesPath",
    With[{env = Environment["FEYNRULES_PATH"]}, If[env === $Failed, Missing["nf"], env]]
  ];
];
If[!ValueQ[$ModelPath],
  $ModelPath = Lookup[params, "ModelPath",
    With[{env = Environment["FR_MODEL_PATH"]}, If[env === $Failed, Missing["nf"], env]]
  ];
];
If[!ValueQ[$OutputDir], $OutputDir = Lookup[params, "OutputDir", "UFO_Output"]];
(* Optional FeynRules consistency checks (default off => behaviour unchanged). *)
$RunChecks = ToLowerCase[ToString[Lookup[params, "Checks", "false"]]] === "true";
(* WriteUFO auto-computes 1->2 partial widths by default. That routine is broken
   under Wolfram Engine >=15 (IndexRange[Index[Spin]] no longer iterates), so we
   default it OFF and let MadGraph compute widths from the UFO instead. Pass
   AddDecays=true only on a FeynRules-compatible kernel (<=14.x). *)
$AddDecays = ToLowerCase[ToString[Lookup[params, "AddDecays", "false"]]] === "true";
(* Total BSM Lagrangian symbol defined in the model file. Models name it
   idiosyncratically (LBSM, LLeptoQuark, ...); pass LagName to target it.
   The full Lagrangian handed to WriteUFO is LSM + <LagName>. *)
$LagBSM = Lookup[params, "LagName", "LBSM"];

(* ---- Validate inputs (existence only). ---- *)
If[$FeynRulesPath === Missing["nf"] || !DirectoryQ[$FeynRulesPath],
  Print["Error: FeynRulesPath must be a directory. Got: ", $FeynRulesPath]; Quit[1];
];
If[$ModelPath === Missing["nf"] || !FileExistsQ[$ModelPath],
  Print["Error: ModelPath not found: ", $ModelPath]; Quit[1];
];

(* ---- Load FeynRules. ---- *)
frm = FileNameJoin[{$FeynRulesPath, "FeynRules.m"}];

If[FileExistsQ[frm],
  Print["[INFO] Loading via explicit file: ", frm];
  Get[frm],
  Print["[INFO] Adding to $Path and using Needs[]. Dir: ", $FeynRulesPath];
  If[!MemberQ[$Path, $FeynRulesPath], AppendTo[$Path, $FeynRulesPath]];
  Needs["FeynRules`"]
];

(* ---- Load SM + your add-on model. ---- *)
smFR = FileNameJoin[{$FeynRulesPath, "Models", "SM", "SM.fr"}];
If[FileExistsQ[smFR],
  Print["[INFO] Loading SM from: ", smFR];
  LoadModel[smFR, $ModelPath],
  Print["[INFO] Loading SM from search path + add-on: ", $ModelPath];
  LoadModel["SM.fr", $ModelPath]
];

(* ---- Resolve the total Lagrangian (LSM + the model's BSM Lagrangian). ---- *)
$LTot = LSM + ToExpression[$LagBSM];
Print["[INFO] Total Lagrangian: LSM + ", $LagBSM];
If[Head[ToExpression[$LagBSM]] === Symbol,
  Print["Error: BSM Lagrangian symbol '", $LagBSM, "' is undefined in the model."];
  Quit[1];
];

(* ---- Optional consistency checks (gauge invariance / symmetry). ---- *)
(* Each check is wrapped in Check[...] so an aborted check still closes its
   sentinel block; output is parsed by tools/feynrules/wl_checks.py. *)
If[$RunChecks,
  Print["[INFO] Running FeynRules consistency checks."];
  Print["HEPTAPOD-CHECK-BEGIN: hermiticity"];
  Check[CheckHermiticity[$LTot], Print["HEPTAPOD-CHECK-ERROR"]];
  Print["HEPTAPOD-CHECK-END: hermiticity"];
  Print["HEPTAPOD-CHECK-BEGIN: kinetic_terms"];
  Check[CheckDiagonalKineticTerms[$LTot], Print["HEPTAPOD-CHECK-ERROR"]];
  Print["HEPTAPOD-CHECK-END: kinetic_terms"];
  Print["HEPTAPOD-CHECK-BEGIN: mass_spectrum"];
  Check[CheckMassSpectrum[$LTot], Print["HEPTAPOD-CHECK-ERROR"]];
  Print["HEPTAPOD-CHECK-END: mass_spectrum"];
];

(* ---- Write UFO. ---- *)
Print["[INFO] UFO output: ", $OutputDir, "  (AddDecays -> ", $AddDecays, ")"];
WriteUFO[$LTot, Output -> $OutputDir, AddDecays -> $AddDecays];

Print["[INFO] Done."];
Quit[0];