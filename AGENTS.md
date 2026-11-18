# Working on the SciBmad ecosystem

Guide for contributors and AI coding agents. Every package's `CLAUDE.md` imports this file, and 
also the package's own `AGENTS.md` where there is one (e.g. `BeamTracking/AGENTS.md` for the 
tracking kernels and the Beamlines extension). This file has the rules for all packages, the 
environment, how to verify changes, ecosystem facts, and SciBmad's own architecture and status. 
Read all of it before changing code. Keep it up to date: when a session settles something (a 
convention, a pitfall, a measured result, an open question), add it here or in the package's file.

## Packages

All are local clones in `~/.julia/dev` (local branches may be ahead of GitHub):

- `SciBmad`: the umbrella package. Re-exports the others (BeamTracking's names explicitly, without 
  `track!`: users track with `track`; `BeamTracking.track!` is for single elements and custom 
  loops) and adds `find_closed_orbit`, `track`, `twiss`, `dynamic_aperture`, `phase_trombone!` 
  and `experimental/` (one-torus tools).
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
  Only claim test runs that have finished. The current release texts (PR descriptions, release notes,
  Slack message for SciBmad, BeamTracking and Beamlines) are in `~/scibmad-release/`.

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
  a script after changes, inside the module (`using SciBmad; Core.eval(SciBmad, quote … end)`: 
  it calls `track!`, which SciBmad does not export). `experimental/` is not in the workload and 
  must stay out of it, but it must keep working (smoke test `Experimental.dynamic_aperture`, 
  `max_one_torus`).
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
  On Windows with Julia < 1.11 and KernelAbstractions ≥ 0.10, CPU arrays use broadcasts instead 
  of these kernels (`@static`, methods `set_v!(::KA.CPU)` etc.): KA 0.10 compiles CPU kernels with 
  POCL/GPUCompiler, and on Julia 1.10 a kernel compiled in the precompile workload leaves device 
  symbols (`julia.gpu.state_getter`, `__spirv_BuiltInGlobalInvocationId`) in SciBmad's package 
  image, which Windows can't link (macOS/Linux resolve them lazily; check with `nm -u` on the 
  image). Bitwise equal to the kernels (coasting and not, batches, BatchParams, Float32, twiss, 
  dynamic_aperture); broadcasts are faster than the POCL kernels on the CPU (0.7 vs 1.6 µs, 1000 
  particles). Upstream reproducer and reply: `~/scibmad-release/windows-lts-issue.md`.
- `src/track.jl`: `track`, `TrackConfig`, `TrackResult`, `track_spin`. `track` and 
  `dynamic_aperture` build the kernel chains once per call (`BTBL.prepare`) and run them every 
  turn with `track!(bunch, chains; <launch keywords>)` (bitwise equal to `track!` per turn); 
  `track`'s `elapsed_time` excludes building them (about one turn of one particle).
- `src/twiss/`: `twiss.jl` (`twiss` and its phases), `twiss_operators.jl` (one function per 
  column, `_beta1(j, twi, cache, Val{as_tps})` etc.), `twiss_map.jl` (column name → function, units), 
  `twiss_df.jl` (`_twiss_df`, the row/column loop), `twiss_types.jl` (`Twiss`, `TwissSummary`, 
  `TwissInternal`, `TwissCache`).
- `src/match/`: matching and optimization (see "Matching" below). `expressions.jl`
  (`LatticeExpr` nodes, `Knob`/`@knob`, `TwissExpr` and its locations, `ExprVector`, `lazy`,
  `deriv`), `engine.jl` (`value`, `value_and_jacobian`: plan, seed, one `twiss` per
  `TwissExpr`, extract, restore), `match.jl` (`Target`, `MatchProblem`, `match!`, `MatchResult`).
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
3. Tracking with `track` (`_twiss_track`: `_track` at the rows, `save_reference=true` so that all
   of `twiss`'s tracking compiles the same kernels; the Bunch `t_ref = zero(s[1])` holds parametric
   lengths). Uniform truncation orders: the identity is tracked once, the saved maps are the maps
   from the start to the rows (`_twiss_maps`), the last one the one-turn map, which gives `a`
   (`normal`) and the tunes. Otherwise the one-turn map gives `a`, and the last tracking of the
   fixed point `a0` (see 4.) is saved at the rows: `a` there is the tracked `a0` composed with
   inv(`a0`) ∘ `a`, which has no constant part in the oscillating variables, so the missing x²
   terms of the tracked map only give x² terms, which are truncated anyway (Nov 2026; before, `a`
   was tracked once more). With an `a_initial` it is tracked itself.
   `beta_gamma_ref` and `t_ref` of the rows come from the result.
4. `factorise` (NNF) of `a`, then `_twiss_push_a`: per row `a = map ∘ a` (or the tracked `a`) and
   `factorise` (canonise level from `canonise_phase_damp`; 0 at order 1) with fresh phases: the
   phases from the start mod 1, unwrapped against the previous row (so rows must be less than half
   a turn of phase apart, as before: with sparse `at` the integer tunes are lost, a known issue).
   The slip of a coasting beam and the damping are totals. Old and new agree to round-off (23
   cases), except the self-detuning with variable orders below the total order (an artifact of the
   truncation either way; with complete orders they agree to all digits).
   **Variables below the maximum order** (`chrom` above `order`, or a Descriptor with GTPSA
   parameters and the variables at order 1): the one-turn map around the closed orbit then lacks
   the terms (x², ...) that the dependence of the fixed point on δ or on a parameter turns into
   terms of lower order in x, so `a` from it is only right to first order in them (docs FODO,
   dispersion 1.8 m at its start: dq1/dδ -0.0759 instead of 0.0875617, which tracking differences,
   `order=2` and Tao (0.087581) give; ESR from IP6, dispersion 5e-8, was right to 4e-8, the same
   ring started at its largest dispersion gave 1.4166 instead of 1.1693). `twiss` therefore refines
   the fixed point (Nov 2026): it tracks `a0` (the translation to the fixed point) and takes `a`
   from the one-turn map around it, ⌈log2(maximum order)⌉ times (the error of the fixed point is
   squared each time). It must be `a0` and not `a`: with these Descriptors `a` carries incomplete
   second order terms (x·y but not x²), and composing with the first order part of the orbit
   in δ turns them into wrong first order terms (uncoupled lattices were right either way, coupled
   ones with dispersion wrong). Now `chrom=n` equals `order=n` to 1e-13 (n = 2, 3; coupled, with
   spin), the Descriptors of `parametric-nf.md` are right with a kicked orbit and with dispersion
   (dq1/dkqf 0.572409 against 0.572409 from tracking, 0.5566 before), also in 6D. Cost: ESR
   `chrom=2` 683 -> 774 ms (`order=2`: 1300 ms). Test "Chromatic and parametric functions with a
   parameter dependent closed orbit" (12 of 16 fail without the refinement). Uniform orders (as
   `value_and_jacobian` uses) never needed it. Taking the rows from the last tracking of `a0`
   (3.) instead of tracking `a` once more: ESR `chrom=2` 775 -> 690 ms, with 5 rows 337 -> 258 ms;
   round-off against tracking `a` (10 cases: `chrom` 2 and 3, coupled, spin, `delta0`, sparse
   `at`, parameters with the variables at order 1, 6D, ESR: ≤ 1.4e-12 of each column's scale),
   and as close to `order=2` as before (1e-15 to 2.5e-13); composing with `a` instead of
   inv(`a0`) ∘ `a` fails 17 of 27 checks of that test.
   **δ of a coasting beam (`delta0`):** in the maps δ is a parameter without a value (their sixth
   component has no constant part). `_twiss_track` adds the δ of the orbit to the tracked sixth
   coordinate, `_twiss_map` takes it off the tracked map again, and `TwissInternal.delta0` gives
   it to the `pz` column and to `_slip_rel_p`. (Before Nov 2026 the map was tracked at δ = 0
   around the orbit of δ: q1 0.05346401 instead of 0.05345806 at δ = 1e-4 on the docs FODO, `pz`
   0. Putting δ into the constant part of the identity's sixth component instead made order 1
   right but counted the orbit twice in `x` and broke order 2: the normal form takes that constant
   as a displacement.) Tested against tracking and the expansion in δ from δ = 0, orders 1 and 2
   and `chrom=2`.
5. `_twiss_df`: evaluates every column for every row. The caches in `TwissCache` hold one row's 
   intermediate results (`a1_mat`, H/B matrices, …) shared by its columns (`persistent_*` caches
   live for the whole table). The rows go in blocks of 64, column by column (`_twiss_df_column!`,
   a function barrier on the column function), each row of a block with its own caches: one
   dynamic call per column and block. Per cell, the 26 column functions at one call site missed
   Julia's call-site cache and took the slow method lookup (Nov 2026: ESR table 39.6 -> 11.2 ms,
   88 -> 30 MB, `twiss` 251 -> 214 ms; bitwise on 18 cases; first calls unchanged within their
   spread, 17 s ESR without the workload).
6. `_twiss_summ`: tunes, `etac`, `alphac`, spin tune, damping, at the end row. It empties the row
   caches first: they hold the last row of the table, and with RF the 6D `etac`/`alphac` used `B3`
   of that row when the table did not end at the end (4.4e-5 relative on a test ring; fixed Nov
   2026, test "Phases and tunes with few rows" compares all summary values).

Callbacks get `(i, coords, p)` with a `TrackPoint` `p` (`p.s` and `p.step` from the beginning of 
the Beamline, `p.t_ref`, `p.beta_gamma_ref`, `p.last_ds_step`, `p.last_g`); `twiss` uses 
`transforms_out!(i, coords, p)`/`transforms_in!(i, coords, p)` to store maps in lab coordinates 
inside misaligned elements (unless `in_body_coordinates`). `track` saves its locations (`at`) with 
one callback (`_track_make_callback`): `location_of_step[p.step]` is the location or 0, `slot[1]` 
the column of the turn (0 if not saved, set before each turn); the location where a turn begins is 
copied before the turn (no callback runs at step 0). It always tracks with this callback (also 
with `at=[]`), so `track`, `twiss` and `dynamic_aperture` compile the same kernels. Saved: the 
initial particles (column 1, at every location), then every `save_every_n_turns`-th turn.
Reading a `TrackResult` (Nov 2026, maintainer's choices): `res.x` … `res.pz`, `res.v`,
`res.q`, `res.state`, `res.t_ref`, `res.beta_gamma_ref` are `TrackColumn`s read at the
locations of `twiss` (same `_column_locations`; no location = the only saved location, an error
if several were saved) with the
keywords `turn` (the turn number, 0 = initial) and `particle`: `res.x[qd, turn=2, particle=1]`.
Julia indexing takes keywords after a comma (`a[i, k=1]`), not a semicolon, and lowers `end`
with several indexes or keywords to `lastindex(a, 1)` (all column types define it). `TrackResult.rows`
maps (beamline index, step in the element) to the location number (from
`_twiss_assemble_locations`, whose last output is the step within the element); `res.turns` lists
the saved turns. The saved arrays moved to `res.data` (`res.data.v[particle, coordinate,
location, saved turn]`, …; breaking) and are what internal code uses (`twiss` reads `tr.data`).
`track_spin(res.q, s0)` accepts the column. Tested against `res.data` ("Tracking results read at
locations"; shifting the turn mapping fails 5 of 16).
`save_reference=true` also saves `p.t_ref` and `p.beta_gamma_ref` at the locations (one value per
location and turn, written by particle 1; per particle with `BatchParam`s, returned as
`BatchParam`s). It is off by default: the reference time is then computed at every integration
step (6% for one particle on ESR; `_DtRef`, a lazy time in BeamTracking, still left 3-6% from the
values kept alive for the save). With it off the kernels' native code is identical to without it.

### Matching (`src/match/`)

- User level: `tw = TwissExpr(bl; twiss kwargs...)`; `tw.beta1[ele]` etc. are `LatticeExpr`s
  (nothing is computed), arithmetic builds trees, `==`/`<=`/`>=` make `Target`s; `match!(vary=
  [@knob(qf.Kn1), ...], targets=[...])`; `value(exprs)`, `value_and_jacobian(exprs, knobs)`;
  `MatchProblem` gives `value(prob, x)`, `value_and_jacobian(prob, x)`, `penalty(prob, x)` for
  other optimizers (checked with NonlinearSolve, Optim LBFGS, Metaheuristics DE). Everything has a
  function form (`Knob(obj, :prop)`, `Target(expr, value)`) for a later Python wrapper. Docs:
  `docs/src/optimize.md`.
- Engine (`_evaluate_once`): collect the leaves per `TwissExpr` (dependencies such as
  `a_initial=tw.a[end]` first); the knobs that carry derivatives are those varied (order 1) plus
  those of `deriv` (their own orders); one Descriptor for all `twiss` calls with **all six
  variables at order v+n** (v: the largest order a quantity needs, `_col_order`/`_leaf_order`; n:
  the total order in the knobs), which uses the single-tracking path of `twiss`; set `GTPSA.desc_current`, set each knob
  to value + parameter, one `twiss(...; at, cols, GTPSA_descriptor)` per `TwissExpr`, then in
  `finally` set the numbers back (with the Descriptor still current), `scalarize!` the elements
  touched and restore `desc_current`. Each leaf is reduced to a Taylor series in the knobs alone
  (`_to_p`: the part without phase space dependence, in a parameter-only Descriptor), and the
  trees are evaluated with these (`_peval`); `deriv` is `GTPSA.deriv` there.
- Rows: a location is (beamline index, step); the end of an element is the beginning of the next
  (-1: the end of the beamline). Beginnings are requested by index, interior steps by an s-range
  over the element, the end by `(s_end - δ, Inf)`; the rows of the result are found with
  `Twiss.rows` (below).
- Locations are the same for `TwissExpr` and `twiss` tables (maintainer, Nov 2026: "everything in
  TwissExpr accessible identically in regular twiss"): `_column_locations(bl, stop, args...)` in
  `expressions.jl` turns any index (integer = beamline index, element, name, vector, range,
  `[ele, :exit]`, `[ele, :]`, `[ele, k]`, s-range, `:`, `begin`/`end` of the beamline or of
  `start`/`stop`) into locations (beamline index, step in the element) and says whether it is one
  value. `twiss` stores the row of every location of its table (`Twiss.rows`, from the step of
  each row within its element, which `_twiss_assemble_locations` returns last), with `start` and
  `stop`; `_twiss_row(tw, loc)` finds a row (also for the engine, replacing its search by s;
  `value`/`value_and_jacobian` bitwise unchanged). A table column (`TwissColumn`) is
  therefore not an `AbstractVector` (an integer is not a row): it iterates, broadcasts, compares
  (`==`, `≈`) and displays as the vector of all rows; `collect(tw.beta1)` or `tw.df.beta1` is that
  vector. Plotting it works through package extensions (`SciBmadMakieExt`:
  `Makie.convert_single_argument`; `SciBmadRecipesBaseExt`: a type recipe for Plots), checked with
  CairoMakie 0.24 and Plots. Tests compare expressions with raw rows (`t.df.beta1[r]`) to stay
  independent of the shared code. The docs present this once, in `twiss.md` ("Reading the
  results"); the Quickstart introduces it and `optimize.md` refers to it.
- Tunes and phases: `twiss` unwraps the phases row by row, so with few rows they are only known
  modulo 1 (`twiss(ring; at=[])` gives the tunes in (-0.5, 0.5]). When `q1`, `q2`, `phi1` or
  `phi2` are used, the engine adds rows (`phase_rows` of the `TwissExpr`: elements at most 0.2
  turns apart, chosen from one full `twiss`, chosen again when two rows are more than ~0.35
  apart, `_StalePhaseRows`). With derivatives the rows would each cost a parametric normal form
  (ESR, 40 knobs: 1.8 s instead of 0.48 s), so the integer parts are taken from an evaluation
  without derivatives instead. An unwrapping inside `twiss` from the linear maps at every step
  would make this unnecessary (not done).
- Open lines are checked with numbers first (`_check_open_line`: the initial orbit and the
  transfer matrix by differences, 13 particles): with wild knob values the tracked map overflows
  and GTPSA ends the process in the normal form (`invsqrt(NaN)`), which a solver step can cause
  (seen with BatchSolve's Levenberg-Marquardt on the four-quadrupole line of the docs). Periodic
  lattices throw before that (closed orbit, unstable one-turn map).
- BatchSolve's `levenberg_marquardt!(val!, val_and_jac!, y, jac, x)` runs a `MatchProblem` as it
  is (failed evaluations as NaN residuals). Compared with `match!` on the test problems
  (evaluations, residual-only + with Jacobian): tunes 5 vs 4+5; a first step into instability 13
  vs 17+10; two targets with one knob 108 vs 23+19; four-quadrupole line to β = 10 m 7 vs 7+8;
  ESR tunes 5 vs 4+5; a tune with max β <= 6: met vs not met; three problems neither solves. It
  evaluates the residual at the trial point and then residual and Jacobian again at the same
  point; it has no bounds and no step limits.
- `match!`: residuals = weight × distance from the goal (zero inside an interval, its Jacobian
  row zero) and `minimize_weight` × the expressions to minimize; steps from the SVD of the
  Jacobian of the knobs not held at a bound. Trials per iteration: the Gauss-Newton step, 1/2,
  1/4, 1/8 of it, then Levenberg-Marquardt steps (Nielsen's update); the first trial is evaluated
  with its Jacobian. Measured on the test problems: pure LM stalls at stability limits (a failed
  evaluation counts as infinite), pure step halving stalls on ill-conditioned problems and takes
  108 evaluations for an inconsistent least-squares problem (9 with LM); neither is best
  everywhere. A weight of 1e-5 for `minimize` (to hold the targets exactly) made LM stall: the
  default is 1, and targets that must hold get a larger weight (constrained minimization would
  need a null-space step: not done).
- ESR (5 rows, quadrupole strengths as knobs, Julia 1.10, Apple M4): `value` 60 ms (85 ms with
  tunes); `value_and_jacobian` with 2/10/40 knobs 127/197/478 ms (231/315/587 ms with tunes);
  the same Jacobian by forward differences would take 3/11/41 × 60 ms. A tune match with 4
  quadrupoles: 4 iterations, 1.6 s.
- `_check_cachable` (twiss): the single-tracking path (track the identity once, compose the maps
  with `a`) is now also used when the six variables have the maximum order and parameters lower
  ones (it required all orders equal). The composition loses nothing then, because `a` has no
  constant part. Agrees with the previous path (tracking `a`) to round-off (1e-13 relative on
  small rings with and without RF and spin, chromatic terms, a chain, `deriv`; ESR 4e-14, 8e-15
  absolute on an `alpha1` of 4e-7), and halves the cost of a Jacobian (ESR, 10 knobs: 395 ->
  208 ms).
- Not supported in expressions yet: matrix columns (`N`, `H1`, ...); maps only as inputs of
  another `TwissExpr`. Not done: tracking quantities (`track` at locations as expressions), a
  `start`/`stop` range for `twiss`, chunks of knobs, knob generation.
- Precompile workload: two small `match!` calls (tunes with context knobs and an inequality,
  chromaticities) and a `value` on the parametric FODO. First `match!` in a fresh process (docs
  FODO, element knobs): 16 s without any workload, 3.4 s with the workload before, 1.3 s now;
  `value` 0.17 -> 0.00 s; a tune match on ESR 2.0 s either way (its run time). Cost (two runs
  each, Julia 1.10): precompile 174-179 s both, binary 602 -> 608 MB, peak memory 13.8 and
  15.8 GB before, 16.2 and 15.4 GB now (the spread between runs is larger than the difference).
- Knobs on vectors (maintainer, Nov 2026): `@knob ring.context.coilstrengths` is one knob per
  entry (`Knob.index`; `_knobs` expands it, names `coilstrengths[i]`, the same bounds);
  `@knob obj.prop[i]` is one entry. `_setknob!` writes an entry in place when the vector can
  hold the value, else into a promoted copy. `_evaluate` replaces each vector with varied entries
  by a copy holding the Taylor series and puts the original object back afterwards (writing
  numbers back into the Taylor-series vector would leave the `DefExpr`s returning Taylor series,
  and the next plain tracking aborts in GTPSA). Bitwise equal to separate scalar variables
  (values, Jacobian, `match!`).
- Function knobs (`Knob(get, set)`) may set element parameters the engine does not know about:
  after evaluating with one, every element of the evaluated beamlines is `scalarize!`d (ESR:
  1.8 ms). Before (Nov 2026), the elements kept Taylor-series parameters and the next plain
  `twiss` aborted Julia ("descriptors differ").
- `MatchResult` display: the REPL (`text/plain` with `:limit`) fits each table into the terminal
  height; `print` (two-argument `show`) shows every row (was 30 rows always).
  Test "Knobs on vectors, function knobs, display of many knobs" (negative controls: without
  putting the vector back 2 fail and 3 error, without the copy 1 fails, without the element
  reset Julia aborts).

### Columns: facts found by differentiating them (Nov 2026)

- `h` columns (`_make_h!`): exp(:h:) is the one-turn map around the fixed point, in the
  coordinates of the Courant-Snyder map (the part of `a1` linear in the oscillating variables)
  and with the linear rotation taken off. The rotation keeps its dependence on δ and parameters.
  The Courant-Snyder map is the one at δ = 0, and **with GTPSA parameters it keeps its dependence
  on them**: only then is `h` as a Taylor series in a parameter the series of the `h` computed at
  changed values of the parameter. With the map fixed at the expansion point (as it was) the
  derivative in a quadrupole strength was another quantity (docs FODO, d h3000/d kqf 1.84-2.00im
  instead of -2.52-1.59im), and so was any derivative with a kicked orbit; sextupole strengths
  around a zero orbit were right. Now they agree with central differences (to the step error,
  which goes to zero with the step) for h3000, h2100, h1020, h1011, h2200, h4000, dh2000, dh3000
  with a kicked orbit and coupling, and for 6D terms. Without parameters the code path is the
  previous one (bitwise).
  Convention (checked numerically, documented in `twiss.md`): the map is exp(:h:) first, then the
  rotation; h is in z₁ = (x̃+ip̃)/√2 = √J e^{-iφ}, z₂ its conjugate; to first order a thin normal
  multipole adds KnL Re(x+iy)^{n+1}/(n+1)! with x = √(β₁/2)(z₁e^{-iφ₁} + z₂e^{iφ₁}), φ the phase
  advance from the row to it: h3000 = K2L β^{3/2} e^{-3iφ}/(12√2), h2100 = 3× that with
  e^{-iφ}, h1020 = -K2L β₁^{1/2} β₂ e^{-i(φ₁+2φ₂)}/(4√2), h2200 = K3L β²/16, h4000 = K3L β²
  e^{-4iφ}/96 (all to 6 digits at several positions).
- `etac` and `alphac` with RF: `B3[5,6] sin(2π q3)`. The branch with GTPSA parameters (maximum
  order above 1) lacked the sine: -0.666 instead of 0.0205 on a small ring. Fixed; values and
  derivatives in quadrupole strengths, a length and the RF voltage agree with differences.

### NonlinearNormalForm facts

- `factorise(a; canonise, phase, damp, damping)` returns `(a0, a1, a2, a, r)` (plus `as` with spin): 
  canonise 0 = linear (order 1), 1 = linear with parameters, 2 = nonlinear. `twiss` uses `a`, `a0`, 
  `a1`, `a2`, `as`, never `r` (it calls it with `compute_r=Val(false)`). At order 1 it runs the general machinery (map inversions, compositions, 
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

### Comparison with other codes

ESR (SciBmad's, 6,271 elements) without radiation against Xsuite (xtrack 0.118.7) and MAD-NG (1.1.15, a native arm64 build in
`~/madng-arm64`), each converged to the same 1000-turn error with its cheapest settings (integrator
order and steps per element kind), one thread, Apple M4. Scripts, the settings that make the codes
compute the same physics (exact Hamiltonian everywhere, hard-edge fringes, MAD-NG `frngmax = 3` on
each sextupole, sub-1e-10 m drifts removed for MAD-NG's `minlen`) and the method are in
`~/scibmad-benchmarks/esr-codes/` (README). The converged codes agree to ~3e-10 (round-off floor
~3e-9 of the metric). Nov 2026, ms per turn at a 1000-turn error ≤ 1e-7 (SciBmad: quads MatrixKick
order 6 Yoshida x4, sextupoles DriftKick order 6):

| Particles | SciBmad | Xsuite | MAD-NG (best of 5 processes) |
|---|---|---|---|
| 1 | 1.31 | 1.02 | 5.58 |
| 10 | 4.92 | 7.67 | 6.89 |
| 100 | 40.3 | 77.5 | 39.1 |
| 1000 | 387 | 776 | 676 |

At 1e-5 and 1e-8 the pattern is the same (SciBmad fastest at 10 and 1000 particles, level with
MAD-NG at 100, 1.3-1.5x behind Xsuite for one particle). Rerun SciBmad's side after tracking
changes (`time_sci.jl`, `tN_sci.jl`; settings may need re-selecting with `pareto.py`).

### Status (Nov 2026)

SciBmad 0.7 (branch `v0.7.0`) works with BeamTracking 0.9 / Beamlines 0.12: `check_bl_bunch!` 
replaced (`track!(…; notify=false)`, `BTBL.resolve_reference!`), `Beamlines.R_to_beta_gamma`, 
`LineElement` → `LatElement`, callback counting from `UniversalParams` (above), explicit 
`SolenoidKick()` for the workload's solenoid elements (no default picks it from the fields any more). 
ESR `twiss` is bitwise equal to SciBmad 0.6 on BeamTracking 0.8.7; the workload's FODO differs by 
≤1e-15, from its RF cavity (order 4 instead of 2) and its straight-dipole kicker (a Sextupole kind, 
now DriftKick; the old default used the exact dipole solution: 1.3e-18 there, but strong kickers of 
Sextupole/kicker kinds are now approximated by kicks).

Tracking (BeamTracking's `test/lattices/esr.jl`, 4,891 elements; SciBmad's has 6,271 and is used
for `twiss`, `find_closed_orbit` and first calls; Julia 1.10, Apple M4, per turn): `track` (prepared) 0.50/1.3/8.0/75 ms for
1/10/100/1000 particles, 36 ms for 1000 in Float32 (`track!` per turn: 3.2 ms for one particle).

ESR, Julia 1.10, Apple silicon: `find_closed_orbit` 38 → 17 ms (BeamTracking 0.9); `twiss` 374 → 
352 ms (0.9) → 287 ms (row caches shared by the columns) → 254 ms (NNF `inv!`: `inv(A)` once, work 
buffers only for maxord > 1; bitwise); `chrom=2` case 891 → 839 → 685 ms (the `inv!` change does 
not help at order 2). Phases of `twiss` at order 1: closed orbit 18 ms, tracking the maps 50 ms, 
`push` ~120 ms (mostly `factorise`: 14 µs and ~300 allocations per row), table 56 ms. Through
`track` (Nov 2026; 267-274 -> 261-265 ms, `chrom=2` 692-720 -> 658-676 ms): closed orbit 15 ms,
tracking the identity 46 ms (3 ms of it saving the rows), maps 3 ms, composing 4 ms, `factorise`
per row 155 ms, table ~60 ms. Nov 17 2026 (profile of 233 ms): `factorise` 36% (its two
vector-field exponentials of a coasting beam 17%), tracking 23%, table 20% -> after the blocked
table 214 ms (table 11 ms); `chrom=2`: `factorise` 30%, tracking 26%, table 17%, 690 ms.

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
Nov 2026 (prepared tracking, 64-byte SIMD lanes): the workload also tracks a FODO with apertures 
on every element with length (and `twiss`), with radiation damping (and `twiss`), and with 
Float32 coordinates. First calls on ESR with radiation 3.45 -> 0.21 s, apertures 1.94 -> 0.21 s, 
Float32 6.29 -> 0.21 s; RCSV5S0 (apertures on 5032 elements) `track`/`twiss`/`find_closed_orbit` 
1.52/2.14/1.40 -> 0.36/1.01/0.74 s. Cost: precompile 174 -> 198 s, binary 484 -> 659 MB, peak 
memory unchanged (12.5 GB). All three SIMD loops of `launch!` (64-byte, native width, scalar) 
are compiled with `launch!`: first calls with 1/3/9/17/100 Float64 and 1/5/17/100 Float32 
particles take 0.06-0.08 s. Already covered (first calls <= 1 s): AGS, ESR v6.3.1 and tapered 
`track`/`twiss`/`find_closed_orbit`, AGS spin, `dynamic_aperture`; `twiss(…; order=2)` on ESR 
v6.3.1 takes 2.2 s (1.6 s run time). Time dependent parameters can't be covered in general (the 
function is part of the type).

Open items:
- GTPSA aborts the Julia process (no exception) when two Descriptors are mixed or an invalid
  monomial is indexed. A Float64 stored next to a TPS in a parameter group is converted with
  `GTPSA.desc_current`, so the Descriptor used for parameters must be the current one when they
  are set (creating another Descriptor in between changes it). Setting a number back into a
  parameter group that held a TPS keeps the TPS type: `scalarize!(ele)` (20 µs) is needed.
- The first design draft of the matching interface (before the implementation, with the
  evidence for the two items above): https://claude.ai/artifact/FDK4ajRV3EWLhVaf4j5PLW.
- `twiss` should not assemble its rows itself (`_twiss_assemble_locations`): `track` resolves `at`
  already (it calls the same function), so `twiss` can take the locations from `track`. Remove
  it from `twiss` (not done yet).
- After the `track`/`twiss` rework (Nov 2026): notebooks rerun (Nov 17; `track_spin(res.q, …)`
  is (particle, component, location, turn): the spin notebooks index `[1, 2, 1, :]`; the
  Python `nonlinear-twiss` notebook, legacy for SciBmad < 0.5.0, was removed);
  Julia 1.13 suite as 1.10 (725 passed, 4 broken). BeamTracking's `test/performance_test.jl`
  still allows 2 allocations per prepared chain, now 5 per ESR turn in total: a bound independent
  of the number of elements would catch a regression of the boxed `Coords`.
- First calls with the precompile workload (Nov 17 2026, Julia 1.10, M4, fresh processes; build
  149 s, 13.1 GB peak, 625 MB image), ESR: `twiss` 0.99 s (run 0.42), `chrom=2` 1.72 (0.70),
  `find_closed_orbit` 0.73 (0.02), `track` 0.09, reading a `TrackResult` 0.22, `match!` (docs
  FODO) 1.19, `value_and_jacobian` 1.93 (0.21), `dynamic_aperture` 4.9 (0.13). `main` (0.7.1,
  BeamTracking 0.9.2) in the same way: `find_closed_orbit` 0.83, `twiss` 1.14, `track` 0.08,
  `dynamic_aperture` 1.92. Without the workload `twiss` was 16.7-17.1 s (14.7-14.9 s before
  `twiss` tracked with `track`; the extra host-side compile was not found).
  - KernelAbstractions 0.10 (also on `main`): the closed-orbit kernels are compiled through
    POCL/SPIR-V on the first call instead of coming from the image: `find_closed_orbit` 0.73 s,
    with KA 0.9.48 0.02 s (`twiss` 0.99 -> 0.25 s).
  - `dynamic_aperture` first call (coasting rings, e.g. the ESR test lattice, whose 18
    cavities leave it coasting) was 4.9 s: its batched closed orbits evaluate the residual in
    Float64 with plain `track!` of a Bunch (DI's sparse `value_and_jacobian!` calls f! for the
    value, then the Jacobian with Duals: two trackings per Newton iteration), which `main`'s
    workload covered with `track!(b, fodo)`. The workload now runs `dynamic_aperture` with
    `delta_dependent_orbits=true` (the user-facing path; `track!` itself stays out): first call
    0.90 s (`main` 1.92 s), batched closed orbits 4.17 -> 0.05 s; image 624.9 -> 678.2 MB
    (`main` 665 MB), build 129-133 s and 13.1-13.2 GB either way. Not done: BatchSolve's
    AutoBatch taking the value from the Dual pass would make the Float64 tracking (and its
    image cost) unnecessary and save one tracking per batched iteration.
  - GPU tracking (Nov 17 2026, Metal, tapered ESR v6.3.1 with radiation, Float32): it was
    host-bound, ~1 s per turn for 2,000 and for 20,000 particles alike, because `launch!` waited
    for each of the 6,271 kernels of a turn. BeamTracking now waits once at the end of `track!`
    (`synchronize_launches`): 0.147 s per turn for 2,000, 0.155 s for 20,000 particles; bitwise
    equal (prepared and plain tracking, `track` at locations with transforms and the reference,
    `dynamic_aperture`, IBS, whose statistics are read on the host between kernels: Metal waits
    for the queue before host access). The `track` callback then cost +88% at 20,000 particles:
    `_track_copy!`/`_track_reference!` were `@noinline` calls, which cost registers in GPU kernels
    even where not taken. `_call_or_inline` keeps the call on the CPU (`_PtrArray`, `Array`: CPU
    code and the image unchanged, 624.9 MB) and inlines on GPU arrays: +14% (0.177 s). The GPU
    `dynamic_aperture` example: 1.25-1.34 -> 0.205 s per turn (0.7: 1.03-1.09 s). The remaining
    14% was taken apart (20,000 particles, interleaved, medians +-1%): an empty callback +0.4%;
    the per-step check with any store in its branch +4.7-5.8% (also with `j` captured by value
    instead of `slot[1]`, and with an empty `location_of_step`); the full save +13.6-14.1%, with
    or without the transform code. Likely the compiler must assume the stores alias the
    coordinates (not proven). Not removable inside the kernel (a runtime flag leaves the code in);
    only by tracking without the callback when no location is inside an element (segments of
    chains, copies between them), which adds a kernel set: not done.
    Measure GPU changes interleaved in one process: separate processes scattered by 10-30% while
    the tracking was host-bound.
- `factorise` speed: the remaining cost is ~19 temporary maps per call (inversions, compositions, 
  `exp!` work maps; sharing the two `exp!` calls' work maps would save ~2) and GTPSA arithmetic; 
  further gains need structural changes. A separate order-1 path based on Duals (GPU batchable) is 
  planned as its own implementation: don't build it as part of other work. The `chrom=2` case 
  (order 2) has not been profiled yet.
- `twiss` table: was one dynamic call per cell; now per column and block of rows (`twiss` 5.).
  Specializing the row loop on the column set compiles per column set (5 s for the default set):
  rejected.
- Checked as necessary in the TrackPoint PR (Nov 2026, ESR, Julia 1.10): BeamTracking's
  `LaunchOptions` (without it the keywords that `track` passes allocate per chain: 4,901
  allocations and 0.52-0.54 instead of 0.42-0.46 ms per turn for one particle; equal without
  keywords and at 100 particles); the lazy `p.t_ref` (eager: ~5% slower for one particle, bitwise
  equal); SciBmad's `_PtrArray` (3-6% for 1 and 10 particles). `PreKernelChain.step_exit` is used
  by `track`'s `start`; `s_exit` and setting the Bunch's position after prepared tracking are only
  for consistency with plain tracking (tested in "Callback steps", read by nothing else): kept
  (maintainer, Nov 2026). The precompile workload no longer has `track!` of a Bunch: `track!` is
  not exposed to users, and `track` always tracks with its callback (other kernels), so it is
  not to be restored (maintainer, Nov 2026).
- `factorise(…; compute_r=Val(false))` (NNF, Nov 2026) leaves out `r` (an inverse of the
  canoniser, which `twiss` never uses; still computed at canonise 2 for the phase of a coasting
  beam). A `Val`, so that the result type is inferred. `twiss` uses it: bitwise (29 cases), ESR
  217-227 -> 210-211 ms, `chrom=2` 690-700 -> 662-665 ms.
- `coast_check` and the Newton of `find_closed_orbit` compute the same Jacobian at the zero orbit
  (~3% of ESR `twiss`); sharing it needs BatchSolve's `newton!` to take a starting Jacobian.
- Closed orbit: `coast_check` and Newton compute the same Jacobian (~8 ms on ESR). Prepared tracking 
  does not help `find_closed_orbit` on ESR: it tracks each coordinate type (numbers, and Duals for 
  each Jacobian) once (Newton converges in one step from the zero guess), and preparing costs 
  5.5-6 ms per type, more than the plain turn (13.3 -> 14.4 ms; reverted). It would pay off only 
  with two or more evaluations per type (several Newton iterations).
- `Experimental.dynamic_aperture` sets `RFParams = nothing` on the cavities and restores them only 
  after steps that can throw (the main `dynamic_aperture` uses `rf_on=false` instead).
- `BatchParam` element lengths: `find_closed_orbit` fails (its Bunch `t_ref` is a Float64; `track` 
  uses a `BatchParam` `t_ref`), and `twiss` would need each length's `batch_index` element for `s`. 
  A batched `dynamic_aperture` will be a separate function (its particles are initialized on a grid 
  and pushed differently).
- `twiss` without `order` and `chrom` takes the orders the columns need (`_twiss_auto_orders`,
  from `_col_order` as `TwissExpr`): driving terms raise `order`, the other columns `chrom`; with
  synchrotron motion both. Default columns: order 1 (bitwise unchanged). Summary values (e.g. the
  chromaticities) still need `chrom` given. A term of an amplitude-dependent value that the
  truncation does not resolve is an error (`_adv_get`; before silently 0 or inexact): δ-terms need
  the maximum order 1+delta+2ΣJ and δ to order delta, amplitude terms all phase space variables to
  that order (with mixed orders J1·J2 was 5e-7 off). Checked on 90 term/setting combinations
  against order 6: every accepted value agrees, 29 formerly silent wrong values are errors, 2 right
  ones (δ·J1 with mixed orders) are rejected. `getterm` of a plain number (order 1) explains.
  The spin tune comes from the constant part of the spin map, so its terms need one order less
  (`degree0` = 0 in `AmplitudeDependentValue`, 1 otherwise): measured on a coasting and a 6D ring
  (11 terms, 14 settings), the rule accepts only exact values and rejects every wrong one; with
  the orbital rule `getterm(qspin, delta=1, J1=1)` at order 3 was rejected (nonlinear-twiss
  notebook). Damping has no amplitude terms (no nonlinear normal form with damping).
  `TwissExpr` takes the same order (`_leaf_order`; tested, one order lower fails 2 checks).
  Terms (maintainer, Nov 2026): indexing an amplitude-dependent value gives a derivative,
  `q1[delta=2]` = ∂²q1/∂δ² (the coefficient times the factorials; first-order terms are
  unchanged), in tables and `TwissExpr` (factor in `_to_p`), consistent with `dx_n`; `getterm`
  stays the Taylor coefficient (as printed). `dq1_n`, `dq2_n` = `q[delta=n]` in both;
  `te.dq1[J1=1]` is an error (it silently gave q1[J1=1], dropping the δ). `value` and
  `value_and_jacobian` return complex values and Jacobians for complex expressions (h terms);
  targets and `minimize` still need `abs`/`real`/`imag`. Idea, not done: split a complex target or
  `minimize` expression into its real and imaginary parts automatically (smooth residuals; `abs`
  has a kink at 0) — measure against `abs` on a sextupole match first. The table column type
  is `TwissColumn` (was `TwissColumnVector`; `TwissExpr`'s internal one is `_TwissExprColumn`),
  and the tracking types are `TrackResult`, `TrackConfig`, `TrackColumn` (Nov 2026). `Experimental` (`one-torus.jl`, `coord == 0`
  with `ais`) uses `tw.table.a[1]`, which does not exist (`tw.df`). `max_one_torus(…;
  verbose=false)` still prints every turn.
- Future task (known, not urgent): `MatrixKick` is not differentiable at quadrupole strength 0 (the
  kernel takes √K): a knob on a quadrupole with `Kn1 == 0` aborts Julia in GTPSA
  (`sqrt(+0.0000E+00)`, no exception), e.g. `value_and_jacobian([tw.q1], [@knob(q0.Kn1)])`. The
  map is analytic in K; GTPSA's `sincu`-type functions would avoid the square root.
- Unknown entries of `at` are ignored silently; `_dampdec` returns an `Int` 0 without radiation; 
  `coast_check` ignores its `v0`; derivative columns have no units; to check: the 6D dispersion 
  `H3[k,6]` without normalization by `H3[6,6]`.
- Docs (`docs/`) were updated page by page in TOC order (`docs/toctree.md`, see "Writing docs"); 
  all pages are done (README, `lattices/RCSV5S0.jl` too). `coordinates.md` no longer describes 
  what SciBmad does not have (floor coordinates, `FloorParams`, `Girder`, `Fork`, reversed 
  elements, reflection patches); it keeps the frame math the misalignment sections use. 
  `twiss.md` has an open-lattice example (`a_initial`, tested in "Open-lattice Twiss"). 
  Open for the maintainer:
  - A showcase draft (README gallery, first-look example, Speed section, a Performance page with
    figures) is on the local branch `docs-showcase`, not for release; figure data and scripts in
    `~/scibmad-benchmarks/docs-figures/`.
  - SaganCavity spin: the energy kick rotates the spin twice (before and after the kick) with the 
    full `dE*c/P0c` field each time, i.e. twice ∫E_z (the fringe uses half of its field each 
    time). Bmad's lcavity (`this_energy_kick` in `track_a_lcavity.f90`) does the same; tracked 
    alone, the kick rotates exactly 2x the Thomas-BMT angle as dE → 0, and with half the field 
    it matches to the trapezoidal error. Reported to Bmad; fix both after confirmation.
- Python (juliacall) cannot index with keywords: `res.x[turn=3]` has no Python syntax, so
  Python users read `np.asarray(res.data.v)[particle, coordinate, location, saved turn]`
  (documented in `installation.md`, "Using SciBmad from Python"). Twiss columns work
  (`tw.beta1[jl.qd]`). Idea (maintainer, Nov 2026: in the Python wrapper, no Julia changes; not
  done, untested): `res.x[qd, turn=3]` is `getindex(res.x, qd; turn=3)`, and juliacall passes
  Python keywords as Julia keywords, so `jl.getindex(res.x, jl.qd, turn=3, particle=1)` should
  work as is. The wrapper would wrap result objects so that `res.x(qd, turn=3)` and
  `tw.beta1(qd, "exit")` call it, translating a name for `end`, `slice(None)`/"all" to a Julia
  colon, Python ranges/lists to Julia vectors, strings like "exit" to symbols.
- Notebooks (`examples/julia`) are rerun with `jupyter nbconvert --to notebook --execute --inplace` 
  (condamatt env) and `JUPYTER_PATH=<scratch>/jupyter`, holding a kernelspec with the notebooks' 
  kernel name that runs a scratch environment (plus Plots, CairoMakie, LaTeXStrings, Metal, IJulia), 
  so the notebook metadata and `~/Library/Jupyter` stay unchanged. Check every number quoted in 
  the markdown against the new output (orbit-response quoted a stale dy/dtilt, 3% off).
  IJulia ≥ 1.34 has no `src/kernel.jl`: the kernelspec runs `-e "import IJulia;
  IJulia.run_kernel()" {connection_file}`. The Python notebooks run through juliacall with
  `PYTHON_JULIAPKG_PROJECT=<env>`, `PYTHON_JULIAPKG_OFFLINE=yes`, `PYTHON_JULIAPKG_EXE=<julia>`;
  PythonCall in that env must be juliacall's version (0.9.31 for condamatt's juliacall; with a
  newer one Python fails to start: "initialization of math did not return an extension module").

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
