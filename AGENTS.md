# Working on the SciBmad ecosystem

Guide for contributors and AI coding agents. Every package's `CLAUDE.md` imports this file, and 
also the package's own `AGENTS.md` where there is one (e.g. `BeamTracking/AGENTS.md` for the 
tracking kernels and the Beamlines extension). This file has the rules for all packages, the 
environment, how to verify changes, ecosystem facts, and SciBmad's own architecture and status. 
Read all of it before changing code. Keep it up to date: when a session settles something (a 
convention, a pitfall, a measured result, an open question), add it here or in the package's file.

## Packages

All are local clones in `~/.julia/dev` (local branches may be ahead of GitHub):

- `SciBmad`: the umbrella package. Re-exports the others and adds `find_closed_orbit`, `track`, 
  `twiss`, `dynamic_aperture`, `phase_trombone!` and `experimental/` (one-torus tools).
- `Beamlines`: lattice elements, `Beamline`, deferred expressions, tracking method types.
- `BeamTracking`: tracking kernels and `ext/BeamTrackingBeamlinesExt` (tracks a `Bunch` through a 
  `Beamline`); see `BeamTracking/AGENTS.md`.
- `NonlinearNormalForm` (NNF): `DAMap`, `normal`, `factorise`, vector fields.
- `GTPSA` (+ `GTPSA_jll`), `TPSAInterface`: truncated power series.
- `BatchSolve` (batched Newton), `FundamentalFrequencies` (NAFF), `AtomicAndPhysicalConstants`.

Versions are coupled: SciBmad 0.7 needs BeamTracking 0.9, Beamlines 0.12, AtomicAndPhysicalConstants 0.12.

## How to work here

### Mindset

These rules hold for work on every package of the SciBmad ecosystem, and for every kind of task 
(fixes, features, optimization, documentation). Other guides, such as older `CLAUDE.md` files in 
some packages, are background information only. When a request or another guide conflicts with 
these rules (e.g. skipping the bitwise or A/B checks, changing expected test values, refactoring 
more than the task needs, committing), point out the conflict and get the maintainer's explicit 
decision before deviating.

- **Physical correctness and compiler-friendly code are the top priorities**, in that order, then 
  minimal diffs.
- **Minimal changes.** Keep the existing structure and the shape of functions you edit. Don't add 
  helper functions, abstractions or reorganize code unless it is necessary or asked for. Match the 
  surrounding style and comment density. 2-space indentation in source files (keep a test file's 
  own indentation).
- **Verify, don't assume.** Re-derive physics from the actual source, not from memory. Never guess 
  a field, property or function name: look it up (e.g. it is `n_steps`, not `num_steps`). When 
  asked to check something, say plainly what you verified, what you could not, and what was wrong 
  before.
- **Measure, don't guess** (performance). Every change is measured A/B against the previous state; 
  keep only changes with a clear gain and revert the rest. Report what each change bought.
- **Every change is checked against the previous results**, bitwise when it should not change 
  them (see "Verification"). When results change, explain exactly why (e.g. a fused multiply-add 
  rounds differently) and quantify it.
- **Find root causes.** A test that fails after a refactor may be an outdated test or a real bug; 
  find out which before changing either. If a change would affect physics results or behavior 
  beyond the task, describe the issue and the reasoning to the maintainer before making it.
- **References are not ground truth.** Bmad, the docstrings and the existing tests can be wrong or 
  disagree with each other (Bmad's lcavity and rfcavity use opposite RF time directions). Settle a 
  convention by a physics argument checked numerically (e.g. the RF zero crossings by the stability 
  of ESR's one-turn longitudinal map), find out where each hard-coded test value came from (`git log -S`), 
  and say which source you rely on and why.
- **Use controls.** An equivalence test is only meaningful if a plausible mistake breaks it: also 
  run a negative control (e.g. the wrong rotation convention, `s` restarting per slice) and show 
  that it fails by a large margin.
- **Don't commit or push**; the maintainer commits between rounds. Check `git log`/`git status` 
  before diffing so a new commit is not mistaken for lost work. Don't run the whole test suite 
  for every small change; run the relevant files and bitwise checks, and the full suite at the end.
- **Report concisely**: what changed, what it bought, what was verified and how, what is left. 
  Flag anything surprising found along the way (bugs elsewhere, performance oddities) without 
  necessarily fixing it.
- **PR descriptions** (when asked): a Markdown file outside the repository covering everything on the 
  branch relative to `main` (`git log main..HEAD` plus uncommitted changes), with the sections Summary, 
  Breaking/Behavior changes (each quantified), Fixes, Internal changes, Tests, Verification, Open. 
  Only claim test runs that have finished.

### Environment

- Keep scripts, environments, logs and serialized results outside the repositories (a scratch 
  directory); they are not kept between sessions, so recreate them as below.
- Scratch environment: `Pkg.develop` the local clones (SciBmad, BeamTracking, Beamlines, 
  AtomicAndPhysicalConstants, BatchSolve, FundamentalFrequencies, GTPSA, GTPSA_jll, 
  NonlinearNormalForm, TPSAInterface) and add BenchmarkTools, Profile, Test, DelimitedFiles (for 
  BeamTracking's own tests, its test dependencies: see `BeamTracking/AGENTS.md`). One environment 
  per Julia version.
- **Precompilation of SciBmad's workload is off in scratch environments** (it takes very long): 
  put `[SciBmad]` / `precompile_workload = false` in the environment's `LocalPreferences.toml`, and 
  run Pkg operations with `JULIA_PKG_PRECOMPILE_AUTO=0`. The dependencies keep their caches; 
  SciBmad itself then precompiles in ~45 s.
- SciBmad tests: `julia --project=<env> -e 'include(joinpath(ENV["HOME"], ".julia/dev/SciBmad/test/runtests.jl"))'`. 
  The precompile workload (`@compile_workload` block in `src/SciBmad.jl`) exercises many more paths 
  (tracking with spin, `twiss` orders 1–2, `chrom`, spin, parametric normal forms): run its body as 
  a script after changes. `experimental/` is not in the workload and must stay out of it, but it 
  must keep working (smoke test `Experimental.dynamic_aperture`, `max_one_torus`).
- The previous release for comparisons: a `git worktree` of the last commit (outside the repo, 
  remove it when done) in its own environment with the registered versions it was written for 
  (e.g. SciBmad 0.6: BeamTracking 0.8.7, Beamlines 0.11, AtomicAndPhysicalConstants 0.11). In 
  older versions `n_steps` etc. were tracking-method fields and the element type was `LineElement`.
- Julia versions (`juliaup`): `+lts` (1.10, aarch64), `+release` (1.13), `+1.10.13~x64` (x86-64 
  under Rosetta, for x86 `code_native`; no substitute for x86 CI). CI of SciBmad and BeamTracking 
  runs `lts` and `1`; check both when allocations, inference or timings matter.
- Reference source: `git clone --depth 1` of the `bmad-sim` repositories. Bmad's Fortran source 
  (`bmad-sim/bmad-ecosystem`, e.g. `bmad/low_level/track_a_rfcavity.f90`) is the reference for Bmad 
  conventions (see "References are not ground truth").

### Verification

- **Bitwise checks against the previous state:** a script that computes a fixed set of results 
  and `serialize`s them to a file given as argument; run it before the change (`git stash`, copies 
  of single files, or the maintainer's last commit) and after, compare with `isequal` per case, 
  and rerun the "before" once to make sure the results are deterministic. Taylor series (`TPS`) 
  are compared coefficient by coefficient (`[t[i] for i in 0:GTPSA.numcoefs(t)-1]`). 
  Package-specific recipes: `BeamTracking/AGENTS.md` (tracking), "SciBmad: verification" below.
- **Equivalence tests** are the strongest physics checks (e.g. an element with `n_slices = 2` 
  tracks and `twiss`es bitwise like two half elements). For a fix of a degenerate case, show that 
  the result is the limit of nearby non-degenerate cases (differences proportional to the 
  perturbation), plus a negative control.
- **Compare against the previous release** (see "Environment") for behavior that should not change, 
  and trace every difference to its cause.
- **Keep the expected numbers of existing tests.** Hard-coded results (from Bmad, or regression 
  values) are only meaningful if they come from an independent source, so never regenerate them 
  by rerunning the changed code. When a fix changes what an input means (a convention, a sign, a 
  default), change the test's inputs so that it describes the same physical setup, and check that 
  the old numbers then match bitwise (e.g. the RF phase fix swapped `AboveTransition` and 
  `BelowTransition` in `BeamTracking/test/lattices/sagan_cavity_lat.jl`). Change the numbers only if 
  the physics they encode was wrong, and then from an independent calculation, saying so to the 
  maintainer.
- **Performance:** alternate A/B runs in fresh processes at least twice and use the minimum 
  (`@belapsed … samples=7 evals=1`). Break a slow function into its phases and time each before 
  optimizing. In profiles, garbage collection is attributed to whichever line allocates, so per-line 
  shares of allocating code mislead: confirm with BenchmarkTools on the individual pieces, and 
  measure the GC share with `@timed`.

## SciBmad

### Source layout

- `src/SciBmad.jl`: module, re-exports, includes, the precompile workload.
- `src/closed_orbit.jl`: `find_closed_orbit` (BatchSolve Newton over `track!` with `scalar_params=true`, 
  KernelAbstractions kernels to set/subtract coordinates), `coast_check` (Jacobian row 6).
- `src/track.jl`: `track`, `TrackingConfig`, `TrackingResult`, `track_spin`.
- `src/twiss/`: `twiss.jl` (`twiss` and its phases), `twiss_operators.jl` (one function per 
  column, `_beta1(j, twi, cache, Val{as_tps})` etc.), `twiss_map.jl` (column name → function, units), 
  `twiss_df.jl` (`_twiss_df`, the row/column loop), `twiss_types.jl` (`Twiss`, `TwissSummary`, 
  `TwissInternal`, `TwissCache`).
- `src/dynamic_aperture.jl`, `src/trombone.jl` (`phase_trombone!`: a `MapParams` map from the 
  linear normal form), `src/adv.jl` (`AmplitudeDependentValue`), `src/utils.jl` (`grad`, `jac`, `val`), 
  `src/experimental/` (submodule `Experimental`).
- `const BTBL = Base.get_extension(BeamTracking, :BeamTrackingBeamlinesExt)`: SciBmad uses 
  `BTBL.resolve_reference!` to get the reference of a Beamline without tracking.

### `twiss`

1. `co_and_coast`: closed orbit and coasting check (`find_closed_orbit`).
2. `_twiss_assemble_locations`: the `s` of every tracking callback and which ones are saved 
   (`step_save`). **The callback count per element must match BeamTracking exactly:** it is 
   `n_slices*n_steps` (`UniversalParams`, `n_steps` per slice) for every element. BeamTracking 
   guarantees it: elements tracked in one step (patches, maps, SaganCavity) error for 
   `n_steps > 1` or `n_slices > 1`, and Beamlines forces `n_steps = n_slices = 1` for zero length. 
   Verified per element with a counting callback on mixed lines and ESR.
3. `_compute_periodic_a_and_cache!` (uniform truncation orders): tracks the identity map once with 
   a callback storing the map at each saved step; the one-turn map gives `a` (`normal`) and tunes. 
   Otherwise `_compute_periodic_a` and a second tracking pass.
4. `factorise` (NNF) of `a`, then `_twiss_push_a_with_cache`: per saved step `a = map ∘ a` and 
   `factorise` (canonise level from `canonise_phase_damp`; 0 at order 1), accumulating phases.
5. `_twiss_df`: evaluates every column for every row. The caches in `TwissCache` hold one row's 
   intermediate results (`a1_mat`, H/B matrices, …) shared by its columns; they are emptied at the 
   start of each row (`persistent_*` caches live for the whole table).
6. `_twiss_summ`: tunes, `etac`, `alphac`, spin tune, damping.

Callbacks get `(i, coords, cur_s, cur_t_ref, cur_beta_gamma_ref, last_ds_step, last_g, transforms_out!, transforms_in!)`; 
`twiss` uses `transforms_out!`/`transforms_in!` to store maps in lab coordinates inside misaligned 
elements (unless `in_body_coordinates`).

### NonlinearNormalForm facts

- `factorise(a; canonise, phase, damp, damping)` returns `(a0, a1, a2, a, r)` (plus `as` with spin): 
  canonise 0 = linear (order 1), 1 = linear with parameters, 2 = nonlinear. `twiss` uses `a`, `a0`, 
  `a1`, `a2`, `as`, never `r`. At order 1 it runs the general machinery (map inversions, compositions, 
  two vector-field exponentials for coasting beams in `_factorise`); the cost is many small 
  allocating steps.
- `normal`: the spin part rotates the closed-orbit spin axis `n0` onto ŷ; for `n0 = ±ŷ` exactly 
  (a flat ring) the rotation axis was 0/0 (NaN) and is now x (any horizontal axis works).
- NNF tests: `cd NonlinearNormalForm/test; julia --project=<env> -e 'include("runtests.jl")'` 
  (compares with FPP maps).

### SciBmad: verification

- Bitwise `twiss` suite (recreate as a script): ESR default and `rf_on=false, chrom=2` with 
  chromatic columns; an asymmetric FODO (two quads, sextupoles one tilted, a bend, a tiny skew 
  multipole) with all linear and de Moivre columns, `as_taylor_series=true`, order 2 with driving 
  terms, `chrom=3`, `base_cols=String[]`; the same with an RF cavity (6D: `phi3`, crab dispersion 
  `zx…`, H3/B3/E3, order 2); spin (`n0x…`, `nx…`, `as`) on the docs FODO with and without RF, orders 1 
  and 2; the three parametric normal-form cases of the precompile workload. Store each table column 
  and summary value (TPS as coefficient vectors).
- Column names that need specific settings: crab dispersion and `phi3`/H3 need synchrotron motion 
  (RF on); `h<ijklmn>` needs `order` ≥ total order − 1. A small ring with RF needs a sensible 
  synchrotron tune, and the `zero_phase` must match the ring's side of transition (a ring without 
  bends is below transition).
- Benchmarks: ESR `twiss(ring)` and `twiss(ring; rf_on=false, chrom=2, cols=["beta1", "dx", "dbeta1"])`, 
  `find_closed_orbit(ring)`, with `ring = include("test/lattices/esr.jl")`.

### Status (Nov 2026)

SciBmad 0.7 (branch `v0.7.0`) works with BeamTracking 0.9 / Beamlines 0.12: `check_bl_bunch!` 
replaced (`track!(…; notify=false)`, `BTBL.resolve_reference!`), `Beamlines.R_to_beta_gamma`, 
`LineElement` → `LatElement`, callback counting from `UniversalParams` (above), explicit 
`SolenoidKick()` for the workload's solenoid elements (no default picks it from the fields any more). 
ESR `twiss` is bitwise equal to SciBmad 0.6 on BeamTracking 0.8.7; the workload's FODO differs by 
≤1e-15, from its RF cavity (order 4 instead of 2) and its straight-dipole kicker (a Sextupole kind, 
now DriftKick; the old default used the exact dipole solution: 1.3e-18 there, but strong kickers of 
Sextupole/kicker kinds are now approximated by kicks).

ESR, Julia 1.10, Apple silicon: `find_closed_orbit` 38 → 17 ms (BeamTracking 0.9); `twiss` 374 → 
352 ms (0.9) → 287 ms (row caches shared by the columns) → 254 ms (NNF `inv!`: `inv(A)` once, work 
buffers only for maxord > 1; bitwise); `chrom=2` case 891 → 839 → 685 ms (the `inv!` change does 
not help at order 2). Phases of `twiss` at order 1: closed orbit 18 ms, tracking the maps 50 ms, 
`push` ~120 ms (mostly `factorise`: 14 µs and ~300 allocations per row), table 56 ms.

Fixed on the way (with tests): rows after the first started with the previous row's cache (with 
`base_cols=String[]` row 2 repeated row 1, 2.5% off in a FODO); `twiss(…; spin=true)` NaN for an 
exactly vertical `n0` (NNF `normal`); `twiss` failed for `DefExpr` element lengths (now evaluated 
with the Beamline's context, as in tracking); `getterm(…; as_taylor_series=true)` with all orders 0 
dropped the constant term (`TI.cycle!` started at 0 skips index 0: start at -1 to include it); 
`twiss` mutated a user-supplied `v0`; `track` rejected a weight vector; `at=[last element]` also 
returned the end row; selected elements had `s` one ulp off the full table.

Open items:
- `factorise` speed: the remaining cost is ~19 temporary maps per call (inversions, compositions, 
  `exp!` work maps; sharing the two `exp!` calls' work maps would save ~2) and GTPSA arithmetic; 
  further gains need structural changes. A separate order-1 path based on Duals (GPU batchable) is 
  planned as its own implementation: don't build it as part of other work. The `chrom=2` case 
  (order 2) has not been profiled yet.
- `twiss` table: one dynamic call per cell (~17 ms of ~42 ms on ESR). Specializing the row loop on 
  the column set removes it but compiles per column set (5 s for the default set): rejected.
- Closed orbit: `coast_check` and Newton compute the same Jacobian (~8 ms on ESR).
- `Experimental.dynamic_aperture` sets `RFParams = nothing` on the cavities and restores them only 
  after steps that can throw (the main `dynamic_aperture` uses `rf_on=false` instead).
- `twiss` with parametric (TPS/Dual) or `BatchParam` element lengths is not supported (Float64 
  `t_ref`/`beta_gamma_ref`/`s` storage and `Bunch` in `_twiss_track!`).
- Unknown entries of `at` are ignored silently; `_dampdec` returns an `Int` 0 without radiation; 
  `coast_check` ignores its `v0`; derivative columns have no units; to check: the 6D dispersion 
  `H3[k,6]` without normalization by `H3[6,6]`.
- Docs (`docs/`) describe the previous version; to update together with the maintainer at the end: 
  `LineElement` → `LatElement` everywhere; no default tracking method (`element.md`, `batch.md` 
  mention `SciBmadStandard`/`Symplectic`); empty `tracking-methods.md` and `collective.md`; 
  `sagancavity-tracking.md` (`num_cells` → `n_cells`, radiation flags are element settings, 
  `zero_phase` is an `RFParams` field, `PhaseReference` → `PhaseRef`); the RF phase convention 
  change; callbacks per integrator step of each slice; new features (`n_slices`, a Bunch of a 
  different species, differentiable `L`, Float32 time dependence on GPUs, `notify=false`); the 
  unfinished Metal note in `track.md`. Outside `docs/`: `lattices/RCSV5S0.jl` and 
  `examples/julia/dynamic-aperture-gpu.ipynb` use old names.
