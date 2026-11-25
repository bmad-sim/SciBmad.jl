#!/bin/bash
# run.sh <round-dir> <effort> [chapters...]
S=${SCIBMAD_ENV_PARENT:?set to the dir holding a Julia env named env}; R=$1; E=$2; shift 2
for ch in "$@"; do
 (cd $R/$ch && claude -p "Use the scibmad skill. Julia: run scripts with \`julia +lts --project=$S/env <script>\` (SciBmad is installed there; loading takes ~30 s).

chapter.md is one chapter of the SciBmad ring design tutorial; the code cells that do the matching were removed (the lattice files of the earlier chapters are in lattices/). Do the matching the chapter describes (skip the exercises): write $ch.jl, run it, and write results.json mapping each quantity you varied to its final value, with keys \"<element>.<parameter>\" (e.g. \"qf.Kn1\") or \"context.<variable>\" (e.g. \"context.k2f\", \"context.xf[3]\"). Do not look for the tutorial's own solutions." --model sonnet --effort $E --output-format json --allowedTools "Bash Read Write Edit Glob Grep Skill" < /dev/null > transcript.json 2> stderr.txt) &
done; wait
