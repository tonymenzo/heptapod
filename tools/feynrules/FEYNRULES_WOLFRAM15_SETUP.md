# Running FeynRules `.fr → UFO` with the free Wolfram Engine (macOS/Linux)

The Lagrangian-extraction pipeline's second half (`feynrulestoufo`, `validatemodel`,
the width gate) needs a Wolfram kernel to run FeynRules. This works for **$0** with
the **free Wolfram Engine "Community Edition"** — no purchase, just a free Wolfram ID
and a one-time online activation. Below is the exact setup that this repo is verified
against (Wolfram Engine 15.0 + FeynRules 2.3.49 + MadGraph5 3.7.2).

## 1. Wolfram Engine (the kernel)

```bash
brew install --cask wolfram-engine            # or download from wolfram.com/engine
# claim the free licence once at https://www.wolfram.com/engine/free-license
```

Activation, when the paid desktop `Wolfram.app` is also installed (it hijacks the
default-kernel lookup and demands a paid activation key):

```bash
# point wolframscript at the Engine kernel, then activate with your Wolfram ID
wolframscript -configure WOLFRAMSCRIPT_KERNELPATH="/Applications/Wolfram Engine.app/Contents/MacOS/WolframKernel"
wolframscript -activate                        # enter Wolfram ID e-mail + password (NOT an xxxx-xxxx key)
wolframscript -code "1+1"                      # must print 2
```

Set `wolframscript_path` in `config.py` to `which wolframscript`
(`/opt/homebrew/bin/wolframscript` on Apple-Silicon Homebrew).

## 2. FeynRules — REQUIRED patch for Wolfram ≥ 14

FeynRules 2.3.49 (Sept 2021) predates Wolfram 14/15, which made `Commutator` and
`MatrixSymbol` **Protected** built-ins. FeynRules redefines both, so on an unpatched
install `WriteUFO` throws `SetDelayed::write ... is Protected` and then cascades into
`Table::iterb: IndexRange[Index[Spin]] ...` and malformed `ISUMObject...` symbol-name
errors — the UFO export hangs or produces broken output.

Fix = FeynRules upstream PR #6 (https://github.com/FeynRules/FeynRules/pull/6):
unprotect/reprotect the two symbols around their redefinitions. Patch the two files
in your FeynRules install (`$feynrules_path`):

**`Core/ExtractVertexTools.m`** — wrap the `Commutation relations` section:
```mathematica
FR$ReprotectCommutator = MemberQ[Attributes[Commutator], Protected];
If[FR$ReprotectCommutator, Unprotect[Commutator]];
(* ... existing Commutator[...] := ... definitions ... *)
If[FR$ReprotectCommutator, Protect[Commutator]];
```

**`Core/MassDiagonalization.m`** — wrap the `Getting back a matrix symbol` section:
```mathematica
FR$ReprotectMatrixSymbol = MemberQ[Attributes[MatrixSymbol], Protected];
If[FR$ReprotectMatrixSymbol, Unprotect[MatrixSymbol]];
(* ... existing MatrixSymbol[...] := ... definitions ... *)
If[FR$ReprotectMatrixSymbol, Protect[MatrixSymbol]];
```

Patching the **source** (not the caller) is deliberate: FeynRules launches parallel
subkernels that each reload the package, so the unprotect must live where every kernel
sees it. After patching, `SM.fr` and BSM add-ons export a complete UFO with no
protected-symbol errors.

## 3. Decays

`WriteUFO`'s auto 1→2 decay routine is still broken under Wolfram ≥ 15 (unrelated to
the patch above: `IndexRange[Index[Spin]]` no longer iterates in the decay code path).
`UFO_generator.wl` therefore defaults `AddDecays -> False`; decay widths are computed
downstream by MadGraph's `compute_widths`. Pass `AddDecays=true` only on a
FeynRules-compatible kernel (Wolfram ≤ 14.x) if you want FeynRules-side decays.

## 4. MadGraph5 (UFO consumer / width computation)

```bash
git clone --depth 1 https://github.com/mg5amcnlo/mg5amcnlo.git ~/MG5_aMC
```
Set `mg5_path = "/Users/<you>/MG5_aMC"` in `config.py`. MadGraph auto-converts the
(Python-2-style) FeynRules UFO to Python 3 on `import model`.
