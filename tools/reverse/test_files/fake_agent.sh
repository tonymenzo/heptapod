#!/bin/sh
# Fake blank-slate engine for offline tests.
# Usage (via blank_agent_cmd template): sh fake_agent.sh <mode> {output} {prompt}
#   mode=ok      -> write canned markdown to the output file, exit 0
#   mode=stdout  -> print canned markdown to stdout (no output file), exit 0
#   mode=fail    -> print an error to stderr, exit 3
#   mode=hang    -> sleep 30 (for timeout tests)
MODE="$1"; OUT="$2"; PROMPT="$3"
case "$MODE" in
  ok)
    printf '# Reconstruction\n\n$\\mathcal{L} = |D_\\mu S|^2$ (from LkinS1)\n\nprompt-len: %s\n' "${#PROMPT}" > "$OUT"
    exit 0 ;;
  stdout)
    printf '# Reconstruction (stdout)\n\nfields: 1 scalar triplet\n'
    exit 0 ;;
  fail)
    echo "engine exploded: not logged in" >&2
    exit 3 ;;
  hang)
    sleep 30
    exit 0 ;;
  *)
    echo "unknown mode $MODE" >&2; exit 2 ;;
esac
