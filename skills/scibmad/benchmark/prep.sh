#!/bin/bash
# prep.sh <round-dir>: one dir per chapter with chapter.md (match! cells removed) and earlier lattice files
RD="$(cd "$(dirname "$0")/../../../docs/src/ring-design" && pwd)"; R=$1
for spec in ch01:fodo:0 ch02:dispersion-suppressor:1 ch03:match-to-straight:2 ch06:interaction-region:5 ch07:tune-cell:6 ch09:rf:7 ch12:dynamic-aperture:9; do
  ch=${spec%%:*}; rest=${spec#*:}; page=${rest%%:*}; n=${rest#*:}; d=$R/$ch; mkdir -p $d/lattices
  python3 - "$RD/$page.md" "$d/chapter.md" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
out = re.sub(r"```[^\n]*\n.*?```", lambda m: "" if "match!" in m.group(0) else m.group(0), s, flags=re.S)
open(sys.argv[2], "w").write(out)
PY
  for f in $RD/lattices/*.jl; do b=$(basename $f); k=${b%%-*}; [ $((10#$k)) -le $n ] && cp $f $d/lattices/; done
  cp -r $RD/figures $d/ 2>/dev/null
done
