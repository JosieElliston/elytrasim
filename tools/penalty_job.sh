#!/bin/bash
# One `floor penalty` run from a work-list line: `<name> <flags...>`, the flags passed through.
# Writes $RUN/out/<name>_s<k>.{pitches,trace} and $RUN/out/<name>.tsv (one line per stage).
# A file, not an exported function: exported functions do not survive xargs.
set -u
read -r NAME REST <<< "$1"
[ -s "$RUN/out/$NAME.tsv" ] && exit 0
# REST is split on purpose: it is a list of flags.
"$FLOOR" penalty $REST --out "$RUN/out/$NAME" > "$RUN/out/$NAME.tsv.tmp" 2> "$RUN/out/$NAME.err" \
  && mv "$RUN/out/$NAME.tsv.tmp" "$RUN/out/$NAME.tsv"
