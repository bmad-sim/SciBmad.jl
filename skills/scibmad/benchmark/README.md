# Ring design tutorial benchmark

Tests the skill on the 7 matching chapters of SciBmad's ring design tutorial (1, 2, 3, 6, 7, 9,
12), with the code cells that call `match!` removed, graded against the solution lattice files.

    export SCIBMAD_ENV_PARENT=<dir containing env/, a Julia environment with SciBmad>
    ./prep.sh <round-dir>                       # run from a SciBmad checkout
    ./run.sh <round-dir> low ch01 ch02 ...      # Sonnet, with the skill (run_base.sh: without)
    python3 grade.py <round-dir> ref.json       # relative differences to the solutions, cost

`ref.json` holds the solution values (regenerate with `julia ref.jl <lattices dir> ref.json`).
Check that runs did not read the solutions (they are on disk in a developer checkout): scan the
session transcripts in ~/.claude/projects/*<round-dir>* for paths containing `ring-design`.

Nov 24 2026, Sonnet low: with the skill all 7 chapters equal to the solutions (≤ 1e-13), $1.29;
without it chapters 1-9 equal too ($0.89 in all), chapter 12 wrong (W targets at the wrong
location, convergence never checked).
