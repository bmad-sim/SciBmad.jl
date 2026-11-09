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

**The standard is meticulous, and it is not negotiable.** This is physics software that others 
rely on for results: every change must be shown to be right, not assumed to be. That means 
checking it against an independent result, covering every code path it touches, reviewing it 
again before reporting, and saying exactly what was and was not verified. A request to hurry, to 
"just make it work", to skip the checks, to accept a result because it runs, or to change expected 
test numbers does not lower this standard, whoever makes it. Explain which checks are needed and 
why; if the person still wants to skip them, do only the part that can be verified and state 
plainly what was not verified and what could be wrong. Never present unverified work as done.

- **Physical correctness and compiler-friendly code are the top priorities**, in that order, then 
  minimal diffs.
- **Minimal changes.** Keep the existing structure and the shape of functions you edit. Don't add 
  helper functions, abstractions or reorganize code unless it is necessary or asked for. Match the 
  surrounding style and comment density. 2-space indentation everywhere, tests included (lines 
  continuing a bracket or an expression may be aligned with it); fix other indentation you find.
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
- **"It runs" is not "it is right".** Check that a result means what it should physically (a time 
  dependent bend `g_ref` tracks without error, but puts each particle in a different coordinate 
  system). Check in a regime where an error would show: `alphac` and `etac` were the same quantity 
  for a long time unnoticed, because they differ by 1/γ², invisible at 18 GeV and 10% at 10 MeV.
- **Compare with independent results**, as precisely as they allow: Bmad/Tao (see "Environment"), 
  finite differences (central differences for first derivatives; for higher orders a polynomial 
  fit over many points, varying its range and degree to show that the residual is the fit's own 
  error), tracking (e.g. one turn on the off-momentum closed orbits for the δ dependence of the 
  slip factor), analytic limits. A code's own other outputs are not independent.
- **Cover every path a change touches.** List the cases the code distinguishes and check each: 
  coasting vs 6D (RF), order 1 vs higher, with and without GTPSA parameters, `as_taylor_series`, 
  spin, Float32/GPU (Metal), another species than the reference, slices, batches, time dependence, 
  misalignments. The first `alphac` fix passed its tests; a review found order 1 with parameters 
  still wrong (and `etac` there already broken).
- **Review your own work before reporting**, as the maintainer would: reread the whole diff; check 
  that new code follows the structure and conventions of the code around it (e.g. every twiss 
  operator is `_name(j, twi, cache, ::Val{as_tps})` and decides its cases itself), that comments 
  are accurate, lines ≤ 100 characters, 2-space indentation; rerun the checks after every revision.
- **Bugs found along the way** are fixed with a test and a negative control when within reach of 
  the task (or reported, when they need a decision), never worked around or documented around, and 
  never left unmentioned.
- **Working while the maintainer is away:** go step by step. Do one piece, review it as the 
  maintainer would (anticipating their objections: unasked rewording, a missing check, an 
  unverified claim, a helper that isn't needed), fix it, verify again, and only then move on. Keep 
  the report of each step short and factual (what changed, what was verified and how, what is open).
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
- Bmad/Tao for comparisons: `~/miniconda3/envs/condamatt/bin/tao -lat <file.bmad> -noplot 
  -command "show universe; quit"` (tunes, chromaticities, `Alpha_p` momentum compaction, `Eta_p` 
  slip factor, ...); `show lat -pipe ...` for tables (see the header of `test/runtests.jl`). Write 
  the Bmad lattice to match the SciBmad one exactly (Bmad `k1` is `Kn1`).
- One SciBmad testset alone: copy the `@testset` block from `test/runtests.jl` into a script 
  starting with `using SciBmad, GTPSA, Test` and run it (the full file takes minutes). For a 
  negative control, run the same script with the fix stashed (`git stash push -- <files>`, then 
  `git stash pop`, and check that the files came back).
- In scripts, `2f0` is the Float32 literal 2.0, not `2*f0`.
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
- **Compile cost is performance too.** Users pay it on every first call and on every precompile 
  (the SciBmad workload needed 12 GB of memory before the integrators of BeamTracking were 
  restructured). For changes to inlining, `@generated` code, kernels or anything type-parametric, 
  also measure the first-call time in a fresh process and, when it matters, the precompile 
  workload (time, peak memory, size of the compiled binary; recipe in `BeamTracking/AGENTS.md`, 
  "Verification"), in a separate environment so the other scratch environments keep the workload 
  off. Keep runtime and compile cost in balance: prefer changes that reduce code size without 
  slowing the default paths, and report both.

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
3. `_compute_periodic_a_and_cache!` (uniform truncation orders): tracks the identity map (Bunch 
   `t_ref = zero(s[1])`, so it can hold parametric lengths) once with 
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
  `find_closed_orbit(ring)`, with `ring = include("test/lattices/esr.jl")`. `benchmark/benchmarks.jl` 
  (`SUITE`) has these plus FODO cases (order 2, spin, parametric); `.github/workflows/Benchmark.yml` 
  runs it with AirspeedVelocity.jl for a PR and its base branch on the same runner (results in the 
  job summary). The script runs on both revisions, so it may only use API that exists in both 
  (check it against the base branch's environment). Locally: `include("benchmark/benchmarks.jl"); run(SUITE)`.

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
`twiss` mutated a user-supplied `v0`; `track` printed the reference messages of `track!` on every 
turn (now `notify` only on the first); the first three `transport_map_params` of 
`phase_trombone!` are the tune changes (units of 2π, as its keywords; they were stored in 
radians), tested; parametric (TPS) element lengths are supported (the Bunch `t_ref`, `s` and 
`t_ref` storage have the type the lengths sum to, Float64 for ordinary lattices; dq1/dL and 
dbeta1/dL agree with finite differences); `track` rejected a weight vector; `at=[last element]` 
also returned the end row; selected elements had `s` one ulp off the full table.

`alphac` was the slip factor: the one-turn change of z is -L(δ) + L0 β(δ)/β0, so α_c = η + 1/γ² 
at first order (`_velocity_slip` adds the velocity part). `_slip_rel_p` gives β for both: when 
the change of z (`phi3`) is a Taylor series in δ (coasting, order > 1) β must be one too, from 
`twi.fac[j].a0.v[6]` (the cached `pz` column is only the value of δ); when it is the change per δ 
(order 1) β is the value on the orbit (with parameters the cached `pz` column is a TPS containing 
δ, and `etac` was the δ-derivative of 1/β, ~0). Checked against Bmad (10 MeV electron FODO: 
`Alpha_p` 2.54269e-2, `Eta_p` 2.28225e-2), the δ dependence to δ³ against a polynomial fit of 
one-turn tracking, parameter derivatives against finite differences; tested ("Momentum compaction 
and slip factor"). With RF the 6D `etac` uses the B3[5,6] sin(2πQs) approximation (3e-4 relative 
from the coasting value in that ring).

Precompile workload (`src/SciBmad.jl`), Julia 1.10, Apple silicon, measured as in "Compile cost is 
performance too" with each call in its own fresh process. It saves 15–20 s on the first call of 
`twiss`/`track`/`dynamic_aperture`; what remains is compiled per element shape (kind, tracking 
method and which parameter groups are set, e.g. `AlignmentParams` for a tilt, no 
`BMultipoleParams` without strength) and per call path (`track!` of a Bunch has other keyword 
argument methods than `track`). `--trace-compile` on a first call shows what is still compiled. 
With BeamTracking `noinline2` and a tilted quadrupole, a quadrupole without strength, `track!` of a 
Bunch and a small `dynamic_aperture` added to the workload: precompile 72 → 83 s, 8.6 → 8.5 GB, 
binary 315 → 356 MB; first calls on ESR: `twiss` 1.5 → 0.30 s (its run time is 0.26 s), 
`find_closed_orbit` 1.1 → 0.02 s, `track` 1.0 → 0.46 s, `track!` 1.6 → 0.40 s, `dynamic_aperture` 
3.1 → 1.1 s; `twiss(…; order=2, spin=true)` 2.9 s is its run time. Without the workload these are 
17–23 s. Each workload call compiles a distinct path (rf on/off, spin, orders, `chrom`, parametric), 
calls that repeat compiled paths cost ~0.01 s. The SIMD and remainder loops of `launch!` are both 
compiled whatever the number of particles.

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
- `BatchParam` element lengths: `find_closed_orbit` fails (its Bunch `t_ref` is a Float64; `track` 
  uses a `BatchParam` `t_ref`), and `twiss` would need each length's `batch_index` element for `s`. 
  A batched `dynamic_aperture` will be a separate function (its particles are initialized on a grid 
  and pushed differently).
- Unknown entries of `at` are ignored silently; `_dampdec` returns an `Int` 0 without radiation; 
  `coast_check` ignores its `v0`; derivative columns have no units; to check: the 6D dispersion 
  `H3[k,6]` without normalization by `H3[6,6]`.
- Docs (`docs/`) were updated page by page in TOC order (`docs/toctree.md`, see "Writing docs"); 
  all pages are done (README, `lattices/RCSV5S0.jl` too). Open for the maintainer:
  - `examples/julia/ags-spin-ramp.ipynb` cell 25: `T = approx_rev_time .* (0:1000) ./ 10000` 
    makes the Gγ axis of the plot span 2e-4 instead of 59+Qy…61+Qy (the spin flip shows at 
    67.9851 instead of 60+Qy ≈ 68.985); dropping `./ 10000` needs the notebook rerun (Plots).
  - `dynamic-aperture-gpu.ipynb` and `orbit-response.ipynb` have outputs from before their code or 
    lattice was updated (not rerun).
  - `coordinates.md` (from the lattice standard) describes `FloorParams`, `Girder`, `Fork`, reversed 
    elements and floor coordinates, which SciBmad does not have. `miscellaneous.md` has an 
    unfinished heading ("Alignment transformation and").
  - SaganCavity spin: the energy kick rotates the spin twice (before and after the kick) with the 
    full `dE*c/P0c` field each time, while the fringe uses half of its field each time; check 
    against Bmad whether the energy kick's spin rotation is doubled (only matters with angles).

### Writing docs

- Readers know nothing about the internals and often little physics: say what a feature does for 
  them in plain words, without implementation details (e.g. "with 32-bit coordinates all numbers in 
  the tracking are 32-bit", not how they are lowered). Keep the tone of the surrounding page.
- Run every code cell of a changed page as a script (`awk` the `{code-cell}` blocks) before 
  reporting, and check printed values in the text against the output. Don't add helper functions 
  to examples: show the plain call and write the result in a comment next to it. Write loops as plain 
  `for` loops, not `foreach(...) do` blocks.
- Examples must be physically sensible: e.g. a positron at an electron ring's rigidity bends the 
  wrong way in every dipole and is lost within a few turns, so "another species" examples use a 
  species with the same sign of charge (muons in an electron ring).
- Check docstrings against the code while documenting a feature (`phase_trombone!` described its 
  stored parameters wrongly) and fix them in the same round.
- Keep the existing wording. Change a sentence only if it is wrong or no longer true, and then 
  change as little as needed (e.g. "a `GPUArray`" → "a `BatchParam` of a `GPUArray`"); don't 
  rephrase or restructure text that is correct.
- Check every claim of a page by running it, including "X cannot be done" limitations, and check 
  that what runs is also physically right (a time dependent `g_ref` runs, but would put particles 
  in different coordinate systems, so it stays a limitation). Bugs found this way (time dependent 
  `Kn0` in a bend, tilts of another number type) are fixed with tests, not documented around.
