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

(* ---- Write UFO. ---- *)
Print["[INFO] UFO output: ", $OutputDir];
WriteUFO[LSM + LBSM, Output -> $OutputDir];

Print["[INFO] Done."];
Quit[0];