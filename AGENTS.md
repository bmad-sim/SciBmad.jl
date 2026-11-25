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

Versions are coupled: SciBmad 0.8 needs BeamTracking 0.10.3, Beamlines 0.14.1, AtomicAndPhysicalConstants
0.12, NonlinearNormalForm 0.5.5, GTPSA 1.5.6, TPSAInterface 0.3.1, BatchSolve 0.1.7, FundamentalFrequencies
0.1.2 (SciBmad 0.7: BeamTracking 0.9, Beamlines 0.12). BeamTracking 0.10.3 works with Beamlines 0.13 and 0.14
(its batch, time and Exact tests pass with 0.13.0, Nov 22 2026). Release order: Beamlines 0.14.1, BeamTracking
0.10.3, then SciBmad 0.8.0 (checked Nov 22 2026: the other packages are at their registered versions).

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
- `src/locations.jl`: the locations of a Beamline for `twiss`, `track`, their results and
  `TwissExpr` (see "Locations" under "Matching" below).
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
- Phase trombones (maintainer, Nov 23 2026): the map is a matrix A R A⁻¹ applied around the
  closed orbit, both computed once by `phase_trombone!` (with its `rf_on`), so it is invisible to
  the closed orbit and keeps the lattice functions only while the orbit and the optics stay as
  they were. They are applied at the very end, after orbit correction (ring design chapter 14
  takes them out and puts them back). No trombone-specific code in `twiss`/`track`/etc.: a
  rebuild per call was implemented and reverted (hooks in `twiss`, `track`, `find_closed_orbit`;
  `twiss` 117 -> 177 ms on the tutorial ring); building the map lazily in a `DefExpr` of
  `MapParams` needs no hooks but runs at every preparation (3 per `twiss`, 4 per `chrom=2` or
  `value_and_jacobian`, 2 per `find_closed_orbit`), one `twiss` per trombone each, and cannot see
  the call's `rf_on`, `start`/`stop`, `initial`.
- `const BTBL = Base.get_extension(BeamTracking, :BeamTrackingBeamlinesExt)`: SciBmad uses 
  `BTBL.resolve_reference!` to get the reference of a Beamline without tracking.

### `twiss`

1. `co_and_coast`: closed orbit and coasting check (`find_closed_orbit`).
2. Locations (`src/locations.jl`): `_at_locations` gives the rows (the locations of `at`), and
   `_location_info` the `s`, name, kind and tracking step (`step_save`) of each, plus the
   beginning and the end of the beamline, which are always tracked (`include_start`/`include_end`
   say whether they are rows). **The callback count per element must match BeamTracking
   exactly:** it is `n_slices*n_steps` (`UniversalParams`, `n_steps` per slice) for every
   element. BeamTracking guarantees it: elements tracked in one step (patches, maps,
   SaganCavity) error for `n_steps > 1` or `n_slices > 1`, and Beamlines forces
   `n_steps = n_slices = 1` for zero length.
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
   phases from the start mod 1, unwrapped against the previous row; when the rows are not all
   integration steps, their whole turns come from the order-1 maps at every step (`_twiss_turns`,
   commit 182d287), so phases and tunes have their integer parts with any `at` (ESR `at=[]`:
   q1 48.08, as the full table).
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
   squared each time; the linear part is that at the previous fixed point, so its error lags one
   step: after k steps it is O(δ^(2^k)) in x·δ^(2^k) terms, which the truncation drops when
   2^k ≥ maximum order. The minimum, measured Nov 22 2026 on the docs FODO with a skew quadrupole,
   `chrom=3` against `order=3`: 0 steps dq1 0.12 off, 1 step dq1 exact but dq1_2 1.15 and
   dbeta1_2 1693 off, 2 steps all ≤ 2e-9, 3 steps no change). It must be `a0` and not `a`: with
   these Descriptors `a` carries incomplete
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
   **Orbit-only rows (Nov 2026, maintainer: the same machinery for `twiss` and `TwissExpr`):** the
   orbit is `TwissInternal.a0[j]` (its own vector; `fac` holds the rest of `factorise`). When all
   columns only need the orbit (`_orbit_col`: base columns, `x`, `px`, `y`, `py`, `pz`; 6D also
   `z` and `a0`; a coasting beam also `dx` … `dpy`, `dx_n`; not spin), the rows between the first
   and the last only get `factorise_a0` (NNF: the first factor of `factorise`, the same code), and
   the last row's tunes get their whole turns from `_twiss_turns` (with rows at every step and
   uniform orders from the identity tracking already done, else the order-1 tracking of sparse
   rows); the longitudinal phase is unwrapped against the first row (`jprev`). Bitwise equal to the
   full path in every column and summary value (18 cases: coasting with dispersion and `dx_2`, 6D
   with `z` and `a0`, `delta0`, sparse rows, uniform and mixed parameter orders, an open line, the
   ESR; a wrong `a0` at the rows fails 15). For a coasting beam `z` and `a0` must take the full
   path: `factorise` canonises `a` first, and the canoniser's time-like row has a δ term, which
   changes their δ and parameter dependence (their values agree). ESR: orbit 210 -> 147 ms,
   orbit with `dx_2` (`chrom=2`) 690 -> 385 ms; `value_and_jacobian` of the orbit at 40 BPMs in
   40 correctors (vector knob) 168 -> 65 ms. Next (not done): unstable linear optics (the start
   from `normal(m, 0)`, Newton rounds for higher orders, summary `NaN`).
   **`initial` (Nov 2026, maintainer's choices):** the start of an open line by name, the names of
   the columns (`beta1`, `beta2` required; `alpha1`, `alpha2`, `c11` … `c22`, `dx` … `dpy`, `x`
   … `z`, `pz` = δ, else `delta0`; giving both, or `v0`, or `a_initial`, is an error). Anything
   with names: NamedTuple, Dict (Symbol or String keys: a Python dict through juliacall works,
   checked), a `Context` (its variables are then knobs of `TwissExpr`; the engine's
   `_check_open_line` takes the orbit from `_initial_v0`), a table row (other column names are
   ignored, unknown names are errors, via `_col_order`). `_twiss_initial_map` builds a coasting
   `a` = `factorise_a0`(orbit + η δ) ∘ (A = V N) and the rest is the `a_initial` path. **The `a` of
   `twiss` starts at the normal form origin (`v0` = 0) and the orbit is its constant part**
   (building it around the orbit as `v0` lost the orbit: β off by 0.19 with a kicked orbit). Values
   may be Taylor series in parameters (copied into the twiss TPS type: a static-Descriptor TPS
   plus a dynamic one is a MethodError), not in the variables or δ (error). The linear optics can
   only depend on parameters if the total order is ≥ 2 (x·p): Descriptor(6,1,np,1) gives numbers,
   as before. RF cavities with voltage and `rf_on` are an error, for `initial` and a coasting
   `a_initial` alike (maintainer): the coasting map drops the energy change (an off-crest cavity:
   pz 0 instead of 2e-4, β1 off by 2.5e-4 and 3.7e-4). Zero-voltage cavities are allowed (the
   ESR test lattice has 20). Verified: periodic row 1 -> open twiss of a coupled ring
   (c11 0.78, dy 0.5) with a kicked orbit at δ = 1e-3 to 8e-14 (C negated: 1.6 off, transposed:
   0.12; the testset fails 20 of 100 with C transposed), thin-quadrupole Courant-Snyder, start/stop,
   `chrom=2` against differences in δ, parametric c12 and x against differences, `TwissExpr`
   `value` bitwise equal to `twiss`, its Jacobian in six initial values and a strength against
   differences (≤ 7e-9), `match!` of the start to a waist. Testset "Open-line Twiss from initial
   Twiss parameters"; the bitwise suite is unchanged (the `v0`/`delta0` defaults moved into the
   body).
   **Transfer matrix columns `r11` … `r66` (Nov 2026):** r_ik = ∂z_i/∂z_k from the start of the
   table (Xsuite's `r_ij` convention, checked in its source). Only when asked for, the push loop
   stores per row `a_j ∘ inv(ã_1)` (`TwissInternal.transfer`), with `a` before `factorise`
   canonises it (the canonised `fac[j].a` include the rotation by the phase advance) and `ã_1` the
   first row's `a` without its parameter-dependent translation (a pure translation: a coasting
   `a0` also changes z with x and px). That is the map around the fixed point for the parameters,
   and the inverse has no constant part, so mixed orders lose nothing (composing the uncentred map
   with the translation afterwards lost the x² feed-down: dr11/dδ -0.116 instead of -0.065). Numbers
   without parameters are `NNF.jacobian`; with parameters or a coasting δ Taylor series
   (`_linear_coefficient`), and the δ column of a coasting beam r_i6 = η_i - Σ r_ik η0_k (η0 from
   `transfer[1]`, not `a0[1]`: with `a0` r56 was 3.9e-5 off). Checked against central differences of
   tracking (values to ~1e-9, the differences' own error in z; cases coasting, 6D, sparse rows,
   mixed orders, `start` with `a_initial`, open line), symplecticity 9e-16, d r12/dKn1 with an orbit
   that moves (twiss and `TwissExpr`) and the δ dependence of r11, r12, r16, r56 (mixed and uniform
   orders) against differences; testset "Transfer matrix columns" (uncentred: 10 of 15 fail;
   η0 from `a0`: 1 fails). **One-turn matrix `m11` … `m66`** (periodic tables only): M(s) =
   R(s) R(end) R(s)⁻¹; with parameters from the centred maps without their constant parts
   (`cj ∘ center(T_end) ∘ inv(cj)`), and for a coasting δ m_i6 = η_i − Σ m_ik η_k plus, in z, the
   δ-derivative of the slip per turn (z of the fixed point is not periodic: without it m56 was
   4.5e-2 off). Checked against one turn of tracking from the row (`start` at the row, turn 2),
   m at the first row = r at the end (9e-16), a periodic part, d m12/dKn1, δ dependence of m11,
   m16, m52, m56 (no slip term: 4 of 27 fail; R⁻¹ R(end) R: 3 fail).
   **NNF trap:** `copy` of a coasting `DAMap` shares the δ row (`v[6]`) with the original, so
   writing into the copy's `v[6]` changes the original.
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
initial particles (column 1, at every location the first turn reaches), then every
`save_every_n_turns`-th turn. With `start`, the locations before it are first reached in turn 2:
they are NaN/`STATE_PREBORN` in turns 0 and 1 (maintainer, Nov 22 2026; before, turn 0 held the
initial particles there, giving alive, NaN, alive). `res.start` is the start's beamline index
(1 for `twiss`'s internal tracking). `track(…; species)` sets the species of the default Bunch
(bitwise equal to `bunch=Bunch(…; species)`).
Reading a `TrackResult` (Nov 2026, maintainer's choices): `res.x` … `res.pz`, `res.v`,
`res.q`, `res.state`, `res.t_ref`, `res.beta_gamma_ref` are `TrackColumn`s read at the
locations of `twiss` (same `_column_locations`; no location = the only saved location, an error
if several were saved) with the
keywords `turn` (the turn number, 0 = initial) and `particle`: `res.x[qd, turn=2, particle=1]`
(`res.x[[d], turn=2]`, every placement of `d`, is [particle, location]).
Julia indexing takes keywords after a comma (`a[i, k=1]`), not a semicolon, and lowers `end`
with several indexes or keywords to `lastindex(a, 1)` (all column types define it). `TrackResult.rows`
maps (beamline index, step in the element) to the location number; `res.turns` lists
the saved turns. The saved arrays moved to `res.data` (`res.data.v[particle, coordinate,
location, saved turn]`, …; breaking) and are what internal code uses (`twiss` reads `tr.data`).
`track_spin(res.q, s0)` returns a `TrackColumn` named `:spin` (its array in the field `data`,
which is `nothing` for the columns read from `res.data`), read like `res.q`; `track_spin` of an
array (`res.data.q`, also what Python uses) returns the array. Tested against `res.data`
("Tracking results read at locations"; shifting the turn mapping fails 5 of 16).
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
  (-1: the end of the beamline). The engine passes exactly the locations of its expressions to
  `twiss` (`at=_Locations(...)`, internal); the rows of the result are found with `Twiss.rows`.
  (Before Nov 18 2026 it asked for every step of an element for one step, and for everything
  within 1e-9 m of the end, e.g. zero-length end markers: 2 extra rows on ESR.)
- Locations are the same for `TwissExpr` and `twiss` tables (maintainer, Nov 2026: "everything in
  TwissExpr accessible identically in regular twiss"): `_column_locations(bl, part, args...)` in
  `locations.jl` turns any index (integer = beamline index, element, name, vector, range,
  `[ele, :exit]`, `[ele, :]`, `[ele, k]`, s-range, `:`, `begin`/`end` of the beamline or of
  `start`/`stop`) into locations (beamline index, step in the element) and says whether it is one
  value. `at` of `twiss` and `track` is resolved by the same code (`_at_locations`: each entry as
  one index, but an element is every placement; integers outside a part are left out). Rules
  (maintainer, Nov 19 2026; the shape depends only on the type of the index, as for Julia arrays):
  an integer (also `begin`, `end`, `[5, :exit]`, `[5, 2]`) and an element (`[qd]`, `[qd, :exit]`,
  `[qd, 2]`) give one value; an element must have exactly one placement in the table (or the part;
  copies, also of copies, count as placements: `_is_or_inherits`), else an error naming the
  beamline indexes (`_one_placement`), also with a position (`[b, :]` too); in a vector,
  `[[qd]]`, `[[qd], :exit]`, `[[qd], :]`, it gives every placement. (Nov 18 an element always
  gave a vector: the docs needed `only(...)` in ten places, and an outside reviewer objected to
  dereferencing the common single value.) Beamlines (0.14) indexes a `Beamline` the same way:
  `bl[ele]` is one element (an error if placed several times or not at all), `bl[[ele, 3]]` every
  placement in order (an element of `bl.line` is its own placement; not placed: an error).
  Names, s-ranges and `:` give vectors; `end` is the
  beamline index n+1 (or the index of `stop`), so `end-1` is the last
  element, and the end is like a zero-length element (`[end]`, `[end, :entrance]`,
  `[end, :exit]`, `[end, 1]` are the same location; `[end, :]` is a 1-vector; other positions
  error; in a part table `[end, :exit]` is the exit of `stop`, outside the part: an error); `[]`
  is an error (for `TrackColumn` it is the only saved location); in a part table (`start`,
  `stop`) vectors keep only the locations in the part and an integer outside it is an error; `at`
  entries not in the beamline (unknown names, indexes, elements) are errors. The s of every
  location is s at the beginning of its element as Beamlines sums it (`ele.s`) plus k*L/n, for
  the table and for indexing alike (`_line_steps`; the table used to sum the steps, which differs
  in the last bits where L/n is not exact: 257 of 281 rows on a FODO with L/3 and L/7 steps, and
  an s-range starting at a row's own `s` did not start at that row for 95 of them; `s` of a part
  table was `ele.s` of `start` plus the part's own sums). `twiss` stores the row of every location
  of its table (`Twiss.rows`), with `start` and
  `stop`; `_twiss_row(tw, loc)` finds a row (also for the engine, replacing its search by s;
  `value`/`value_and_jacobian` bitwise unchanged). A table column (`TwissColumn`) is
  therefore not an `AbstractVector` (an integer is not a row): it iterates, broadcasts, compares
  (`==`, `≈`) and displays as the vector of all rows; `collect(tw.beta1)` or `tw.df.beta1` is that
  vector. Plotting it works through package extensions (`SciBmadMakieExt`:
  `Makie.convert_single_argument`; `SciBmadRecipesBaseExt`: a type recipe for Plots), checked with
  CairoMakie 0.24 and Plots. Tests compare expressions with raw rows (`t.df.beta1[r]`) to stay
  independent of the shared code. The docs present this once, in `twiss.md` ("Reading the
  results"); the Quickstart introduces it and `optimize.md` refers to it.
- Tunes and phases: `twiss` gives their whole turns itself with any rows (`_twiss_turns`, see
  `twiss` 4.), so the engine adds no rows for `q1`, `q2`, `phi1`, `phi2` (the earlier
  `phase_rows`/`_StalePhaseRows` of the `TwissExpr` are gone). ESR, 5 rows, 10 quadrupole knobs
  (Nov 19 2026, without GC): `value` 62.3 ms, with the tunes 61.2 ms; `value_and_jacobian`
  260.9 ms, with the tunes 262.4 ms.
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
- `match!` defaults to the own method below (`solver=:gauss_newton`, maintainer's decision Nov
  23 2026); BatchSolve's `levenberg_marquardt!` is `solver=:levenberg_marquardt`. Evidence: on the
  11 problems of the "match!" test set plus ESR tunes and the four-quadrupole waist, Gauss-Newton
  met every target that can be met with as many or fewer evaluations, except the inconsistent
  least-squares problem (1 knob, 2 targets: 108 vs 34); LM failed "q1 + max β ≤ 6" without a
  weight. On the 9 matches of the ring design tutorial GN met 9, LM 7 (it crawls along a curved
  valley in the IR matches, λ 40-450, until maxiter: not a bug); GN found Bmad's root of the
  matching section, LM another. With more knobs than targets, only Gauss-Newton takes minimum-norm
  steps (Nov 22 2026, one tune with two quadrupoles: the change is along the gradient, cos 1.0000;
  Levenberg-Marquardt: cos 0.39). BatchSolve's LM clamped diag(JᵀJ) to an absolute [1e-6, 1e16]
  for its damping scale, so a knob in volts (dq3/dV ~ 1e-8) took 45 evaluations instead of 4;
  the bounds are now relative to the largest entry (Nov 23 2026, test "the units of the unknowns
  do not change the iteration"). `value(prob, x)`, `value_and_jacobian` and
  `penalty` leave the knobs at `x` (documented in `optimize.md`).
- A knob on a parameter set to a deferred expression replaces the expression by a constant, also
  in `value`/`value_and_jacobian` (maintainer, Nov 22 2026: making it a knob sets it, as
  `qf.Kn1 = 0.38` does, which breaks the dependence by design; not a bug). On an element that is
  in no Beamline it errors only because DEFAULT_CONTEXT lacks the variable. Setting a number into
  a parameter group that held a `DefExpr` stores a constant `DefExpr`, so `rawget(...) isa
  DefExpr` does not tell a real expression; `_knob_value`'s `x isa DefExpr` check is unreachable
  for element parameters (`getproperty` evaluates). Setting a dependent property (e.g. `Kn1`
  when `Bn1` is stored) makes it the stored one, by the same rule.
  If a `DefExpr` ever needs to know that it is a constant, do it as `TimeDependentParam` does
  (maintainer, Nov 22 2026; `BeamTracking/src/time.jl`): a `_isconst::Bool` field, not another
  type or type parameter. It is `true` when built from a number (`DefExpr{T}(a::Number)`, `convert`,
  `zero`, `one`), inherited by conversions, kept by an operation with a number, and `true` for an
  operation of two only if both are constant; comparisons (`==`, `isapprox`) are then
  meaningful only for constants.
- `match!` (`solver=:gauss_newton`): residuals = weight × distance from the goal (zero inside an interval, its Jacobian
  row zero) and `minimize_weight` × the expressions to minimize; steps from the SVD of the
  Jacobian of the knobs not held at a bound. Trials per iteration: the Gauss-Newton step, 1/2,
  1/4, 1/8 of it, then Levenberg-Marquardt steps (Nielsen's update); the first trial is evaluated
  with its Jacobian. Measured on the test problems: pure LM stalls at stability limits (a failed
  evaluation counts as infinite), pure step halving stalls on ill-conditioned problems and takes
  108 evaluations for an inconsistent least-squares problem (9 with LM); neither is best
  everywhere. A weight of 1e-5 for `minimize` (to hold the targets exactly) made LM stall: the
  default is 1, and targets that must hold get a larger weight (constrained minimization would
  need a null-space step: not done).
- ESR (5 rows, quadrupole strengths as knobs, Julia 1.10, Apple M4): `value` 60 ms;
  `value_and_jacobian` with 2/10/40 knobs 127/197/478 ms (the tunes, then 85 ms and
  231/315/587 ms, now cost nothing extra: see "Tunes and phases");
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
- `slip` column (coasting): d(z/β₀)/dδ from the start of the table = −c dΔt/dδ, so at the end
  `etac` = −slip/(c T₀). Checked against tracking the δ = ±1e-6 closed orbits (docs FODO, 18 GeV
  and 10 MeV, where z/β₀ and z differ: 1.0290 vs 1.0276 m). With RF it is `B3[5,6] sin(2π q3)/β`.
  Truncation needed (Nov 22 2026, docs FODO): orbit and dispersion columns to the map's order in
  δ (`dx_n`: `chrom` ≥ n), other columns and the tunes one less (`dbeta1_n`, `q1[delta=n]`: n+1);
  amplitude terms of total order m in J: `order` ≥ n+2m+1 (`q1[delta=1, J1=1]` 4, `q1[J1=2]` 5).
  The exact drifts alone give detuning: h2200 = 0.265 in the thin-quadrupole cell of `twiss.md`
  (a thin octupole adds K3L β²/16 = 0.146 for K3L = 1, as the formula below).
- Damping decrements (`damp1`…, `dampdec1`…) are −ln|λ| of the one-turn matrix eigenvalues (to
  1e-10, docs FODO with RF and radiation): the amplitude falls by e^(−damp) per pass. `q3` is
  negative above transition and positive below (checked both sides). Spin tune of the docs FODO:
  −aγθ/2π up to a whole number, to 1e-9, δ term likewise. These are on `twiss.md` (Nov 22 2026).
- Amplitude detuning checked against NAFF of tracking (Nov 22 2026, FODO of `fma.md`, sextupoles
  Kn2 20/−30, 1024 turns; NAFF converged to 1e-11 with 4096): `q1[J1=0] + q1[J1=1] J +
  q1[J1=2] J²/2` agrees to 4e-12 at x = 1 mm, 4e-9 at 3 mm, 3e-7 at 6 mm, **with J the nonlinear
  normal-form action** (from `inv(a)` of the `a` column at `order=5`). With the linear action
  γx²/2 it was off by 2-15% of the shift: the two differ at first order in the amplitude (2.4% at
  1 mm, 16% at 6 mm). `naff` fails on Metal (Float64 inside, and Metal.jl has no FFT).
- `etac` and `alphac` with RF: `B3[5,6] sin(2π q3)`. The branch with GTPSA parameters (maximum
  order above 1) lacked the sine: -0.666 instead of 0.0205 on a small ring. Fixed; values and
  derivatives in quadrupole strengths, a length and the RF voltage agree with differences.

### Contexts (Beamlines 0.13, Nov 2026)

- `GLOBAL_CONTEXTS` is gone (it could not override a Beamline's own value: a context looked at its
  own variables first). Each `Context` has a `saved` stack: `push!(c; k=v, ...)` sets values and
  remembers the old ones (or that a variable was new), `pop!(c)` restores, `withvalues(f, c; ...)`
  restores in `finally`, also what `f` pushed and did not pop. A typed context coerces all values
  before setting any. `merge`/`copy` start with nothing saved.
- Lookup is the context's own variables, then `DEFAULT_CONTEXT`'s (converted to the context's
  type; Beamlines 0.14, maintainer). Elements not in a Beamline (and a `DefExpr` called
  without a context, `deval` without one) use `DEFAULT_CONTEXT` (maintainer's choice, instead of
  an error); BeamTracking's `track!` of a single element too. `NULL_CONTEXT` stays as the empty
  context the members of a Branch/Lattice store.
- DataStructures is no longer a Beamlines dependency. Checked: Beamlines (1,486 + 3,718),
  BeamTracking (all) and SciBmad suites; the twiss bitwise suite; the `defexpr.md` page and Beamlines'
  quickstart blocks as scripts. SciBmad, BeamTracking's src and the matching engine never used
  the stack. Julia 1.10 and 1.13.
- **Julia 1.13.3 bug:** a `while` loop inside a `finally` block segfaults when the `try` block
  throws (plain Julia: `try return f() finally while length(v) > n; pop!(v) end end` with
  `f() = error()`; also without `return`; 1.10 is fine). A function called from `finally` that
  runs the loop works: `withvalues` uses `_pop_to!`. Don't write loops in `finally` blocks.

### NonlinearNormalForm facts

- `factorise_a0(a)` (Nov 2026, NNF after 0.5.5, exported): `factorise`'s first factor `a0` alone
  (`a ∘ zero(a)` plus the identity; for a coasting beam with the time-like correction), the code
  `_factorise` itself calls (bitwise; NNF tests check it against `factorise(a).a0`). Note that
  `factorise` applies it to the canonised `a`. `normal(m, 0)` returns only the first-order
  parameter-dependent fixed point (`-(M_x - I)⁻¹ M_p`, before any eigen-decomposition): it needs
  no stable optics.
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
- Locations (Nov 18 2026): `_twiss_assemble_locations` is gone; `twiss`, `track`, the column
  types and the engine all use `src/locations.jl`. Old bugs fixed with it (tests, negative
  controls on the old code: 19 failures and 6 errors of 56 checks in "Twiss table indexed with
  locations", 9 of 33 in "Twiss of a part of the beamline"): an element in `at` lost its entrance
  row when an s-range had other steps of it; a table whose first row was inside the first
  element gave that row phase 0 (now the full table's phases, bitwise); part tables failed for
  `:`, s-ranges, names and elements placed outside the part; `[end, k]` ignored `k`; names in
  `at` were ignored silently without `start`/`stop`; only one level of copies was followed.
  Bitwise against the previous state on 31 cases (ESR default, `chrom=2`, sparse; FODOs with all
  default columns, Taylor series, order 2, `chrom=3`, spin, RF, `r`/`m` columns, parts, open
  lines, `track` with `at`, `value`/`value_and_jacobian` incl. a part): equal except `s` in the
  last bits (≤ 4.4e-16 on the FODO, 2.3e-13 on an ESR part; ESR itself bitwise) and the
  explained row changes above; `alphac` divides by `s[end]` (1.7e-23 -> 1.4e-23 on a ring
  without bends). Performance (ESR, Julia 1.10, alternating fresh processes): `_line_steps` 0.88
  ms against 1.5-1.8 ms of `_twiss_assemble_locations`; `twiss` 241.6-244.0 ms vs 240.2-246.8,
  `chrom=2` 696-701 vs 711-712, part 111-114 vs 113-114, sparse 79.6-81.2 vs 82.6-84.5, `track`
  with `at=:` 7.4-8.3 vs 8.5; `value` without GC 60.9-61.3 vs 63.9-64.4 (the same call allocates
  87-186 MB from call to call in both versions, so the minimum of 15 samples with GC is noisy;
  the mean of 60 calls with GC: 108-110 vs 111-114 ms); an integer index 0.45 vs 0.36 µs.
  Finding an element is a scan of the line through the copy chains (`_is_or_inherits`, 0.67 ms
  per element on ESR's 6,271 elements). The elements of a vector index or of `at` are found in
  one pass (`_placements`, Nov 19 2026): ESR, 50 elements, `tw.beta1[eles]` 34 -> 1.1 ms,
  `twiss(at=eles)` 111 -> 82 ms, `track(at=eles)` 42 -> 8.2 ms; results equal, bitwise suite
  unchanged; test with nested, repeated and copied elements (walking only the element itself
  fails 7 of 74). `get` instead of `haskey` + `getindex` on the `pdict` was slower for one
  element (0.88 ms) and equal in the pass. A single element stays one scan: a cache per Beamline
  would need invalidation when the line changes (not done).
- After the `track`/`twiss` rework (Nov 2026): notebooks rerun (Nov 17; the spin
  notebooks read `track_spin(res.q, …)[particle=1][2, :]` (Julia; changed Nov 22 without a
  rerun, the expressions checked equal) and `track_spin(res.data.q, …)` (Python); the
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
- `dynamic_aperture` (Nov 22 2026; there were no tests, now testset "dynamic_aperture"): the
  boundary along each angle is the first lost particle of the grid (checked against tracking the
  grid points with `track`), `Inf` if none, a zero column if the particle on the closed orbit is
  lost; `n_r` counts the radius 0. Fixed: `coordinates_number_type` was ignored on the CPU (Float32
  now half the time); `theta_lims` beyond π failed the sanity check (angles now compared modulo
  2π); `verbose=false` still printed; `BatchParam`s in the lattice gave a meaningless result
  silently (each grid particle saw another entry), now an error (`_has_batches`: the prepared
  chains' arguments and reference, ~60 ms on ESR, first call 4.62 -> 4.78 s). Default results
  bitwise unchanged (coasting, 6D, limited angles). With `delta_dependent_orbits` forced on a ring
  with RF, the orbits are computed without RF but tracking runs with RF on unless `rf_on=false` is
  passed: intentional (maintainer, Nov 2026; documented). A ramped reference energy fails in the
  closed orbit (not defined): left as is (maintainer).
- `Experimental.dynamic_aperture` sets `RFParams = nothing` on the cavities and restores them only 
  after steps that can throw (the main `dynamic_aperture` uses `rf_on=false` instead).
- Batched element lengths and reference energies: `find_closed_orbit` gives its Bunch a
  `BatchParam` `t_ref` (and `p_over_q_ref` for a batched energy) as `track` does (Nov 22 2026;
  `_co_reference`, computed once per call and passed to the residuals: before, both failed). Each
  entry equals the unbatched lattice's orbit (coasting and 6D; test "Closed orbit with batched
  lengths and energies", which errors without the fix); unbatched results bitwise unchanged (11
  cases incl. ESR and `twiss`); detecting batched lengths costs ~0.4 ms on ESR (`find_closed_orbit`
  13.3 -> 13.7-14.1 ms; summing the lengths, as `track` does, cost 0.8 ms). Still open: `twiss`
  with `batch_index` fails for batched lengths (it would need each length's `batch_index` entry
  for `s`) and for batched reference energies (`Cannot convert BatchParam to Float64`); batched
  element strengths work.
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
- `_dampdec` returns an `Int` 0 without radiation; 
  `coast_check` ignores its `v0`; derivative columns have no units; to check: the 6D dispersion 
  `H3[k,6]` without normalization by `H3[6,6]`.
- Docs (`docs/`) were updated page by page in TOC order (`docs/toctree.md`, see "Writing docs"); 
  all pages are done (README, `lattices/RCSV5S0.jl` too). A second review in the settled style
  (Nov 22 2026) went through every page (twiss, parametric-nf, optimize, fma rewritten; physics
  pages checked against the kernels); Sphinx builds without warnings. Release texts for 0.8:
  `~/scibmad-release/scibmad-0.8-release-notes.md`, `scibmad-0.8-pr.md` (the whole branch),
  `slack-0.8.md`. `coordinates.md` no longer describes 
  what SciBmad does not have (floor coordinates, `FloorParams`, `Girder`, `Fork`, reversed 
  elements, reflection patches); it keeps the frame math the misalignment sections use. 
  `twiss.md` has an open-lattice example (`initial` and `a_initial`, tested in "Open-lattice Twiss"
  and "Open-line Twiss from initial Twiss parameters").
  Open for the maintainer:
  - A showcase draft (README gallery, first-look example, Speed section, a Performance page with
    figures) is on the local branch `docs-showcase`, not for release; figure data and scripts in
    `~/scibmad-benchmarks/docs-figures/`.
  - SaganCavity: an element with `dE_ref` carries `InitialBeamlineParams`, so it must be the first
    element of a `Beamline` (a `Branch` splits there; BeamTracking's tests use `Branch`). On crest
    (default `zero_phase`), 1 MV gives an electron 1 MeV: `dE_ref = 1e6` leaves pz at 5e-17
    (checked Nov 22 2026; the old docs example had 1 MV with `dE_ref = 0.1e9`, pz −0.09). The
    tracking and physics pages were checked against the kernels (cells, `L_active`, edge,
    energy and ponderomotive kicks, misalignment frames of `coordinates.md`: all agree).
  - SaganCavity spin: the energy kick rotates the spin twice (before and after the kick) with the 
    full `dE*c/P0c` field each time, i.e. twice ∫E_z (the fringe uses half of its field each 
    time). Bmad's lcavity (`this_energy_kick` in `track_a_lcavity.f90`) does the same; tracked 
    alone, the kick rotates exactly 2x the Thomas-BMT angle as dE → 0, and with half the field 
    it matches to the trapezoidal error. Reported to Bmad; fix both after confirmation.
- Python (juliacall) has no keywords in indexing: `res.x[turn=3]` is written
  `jl.getindex(res.x, turn=3)` (juliacall passes Python keywords as Julia keywords; checked Nov 22
  2026 for `res.x` with and without a location and `tw.q1[delta=1]`). `installation.md` ("Using
  SciBmad from Python") states this generally, without results the reader does not know yet.
  Twiss columns index as usual (`tw.beta1[jl.qd]`; `np.asarray(tw.beta1)` is a float64 array).
  Idea (maintainer, Nov 2026: in the Python wrapper, no Julia changes; not done): the wrapper
  would wrap result objects so that `res.x(qd, turn=3)` and
  `tw.beta1(qd, "exit")` call it, translating a name for `end`, `slice(None)`/"all" to a Julia
  colon, Python ranges/lists to Julia vectors, strings like "exit" to symbols.
- Notebooks (`examples/julia`) are rerun with `jupyter nbconvert --to notebook --execute --inplace` 
  (condamatt env) and `JUPYTER_PATH=<scratch>/jupyter`, holding a kernelspec with the notebooks' 
  kernel name that runs a scratch environment (plus Plots, CairoMakie, LaTeXStrings, Metal, IJulia),
  so the notebook metadata and `~/Library/Jupyter` stay unchanged. Check every number quoted in 
  the markdown against the new output (orbit-response quoted a stale dy/dtilt, 3% off).
  The kernelspec needs `--color=yes` in its `argv` (as IJulia's `installkernel` writes it):
  without it `Base.have_color` is `nothing`, IJulia passes that as `:color` of its stderr, and
  GPUCompiler's `__init__` fails ("Cannot convert Nothing to Bool") on `using SciBmad`. Execute into
  the scratch directory first (`--output-dir`), compare the outputs with the committed notebook,
  then copy it back. nbconvert stores every `\r` progress update (`dynamic_aperture`'s "Finished
  turn ...") as its own output; merge consecutive stream outputs and keep the text after the last
  `\r` per line, as the committed notebooks have it. The login shell sets
  `PYTHON_JULIACALL_PROJECT` and `PYTHON_JULIACALL_EXE` to empty strings, which juliacall rejects:
  set them (to the scratch environment and its Julia) along with the `PYTHON_JULIAPKG_*` ones.
  Last run Nov 22 2026, all six notebooks: outputs identical to the committed ones except
  `ags-spin-ramp` (Julia: the `TrackResult` display and the elapsed time; committed).
  IJulia ≥ 1.34 has no `src/kernel.jl`: the kernelspec runs `-e "import IJulia;
  IJulia.run_kernel()" {connection_file}`. The Python notebooks run through juliacall with
  `PYTHON_JULIAPKG_PROJECT=<env>`, `PYTHON_JULIAPKG_OFFLINE=yes`, `PYTHON_JULIAPKG_EXE=<julia>`;
  PythonCall in that env must be juliacall's version (0.9.31 for condamatt's juliacall; with a
  newer one Python fails to start: "initialization of math did not return an extension module").

### Writing docs

This is how the maintainer wants documentation written. It was worked out page by page with them
(Nov 2026), above all on `docs/src/twiss.md`, which is the model page: read it before writing a
page. The maintainer rejected several drafts before this; the rules below are what made the
difference. Docstrings follow the same rules.

**The goal.** Two readers must both be served. A newcomer, who knows nothing about SciBmad's
internals and often little physics, must be able to follow from top to bottom. An expert, who
reads every word, must not find any statement imprecise, awkward or slightly wrong. Keep the
physics light, but every nuance correct. Text that reads like generic AI output (long sentences
of clauses chained with commas, semicolons and colons; vague words; explanations of everything
up front) is rejected: the maintainer said such a paragraph made them feel their "brain
dissociating".

**Structure of a page** (as `track.md` and `twiss.md`):
- A short introduction: what the function or feature is for, in one or two paragraphs. Not a
  summary of every option.
- Then a `Basics` section that starts with the simplest call and its output, and only then
  explains that output (what is shown, what it means, how it was computed).
- Every further section starts with an example of its case, then explains it, then shows the next
  complication. Never explain all concepts up front and give the examples afterwards: the
  maintainer asked twice to "break up the paragraphs around the whole page for a simpler build
  up".
- Ideas go where they are first needed. E.g. on `twiss.md` the modes are introduced in Basics
  where `q1`/`beta2` first appear, `chrom` in the chromatic section, `initial` in Open lines,
  `start`/`stop` in their own section after Open lines (because its open-line example uses
  `initial`).
- Lists of options, columns, values or rules go in a table or a bulleted list, not in one long
  sentence. The `{docstring}` reference goes at the end of the page (or of its section).

**Sentences and paragraphs.**
- One idea per paragraph. Short sentences. Avoid chains like "A, so B; C (D), and E: F".
  Rejected example (the location rules on `twiss.md`): "An element gives one value, so it must be
  in the beamline once (else it is an error); in a vector, `tw.beta1[[d]]`, it gives one value
  per placement, in beamline order. Names, vectors and ranges always give a vector. The end of
  the beamline is like a zero-length element: ..." Accepted: "If an element is placed more than
  once in the beamline, `tw.beta1[d]` is an error, because it is not clear which placement is
  meant. `tw.beta1[[d]]` gives the values at all of them. Names, vectors and ranges always give a
  vector, even if they match only one location." (and the corner case left to the docstring).
- No vague words or placeholders: "the central one", "stuff", "etc." for something that matters.
- Say *why* where it helps the reader decide (e.g. "the chromaticities need a beam without
  synchrotron motion, in which δ stays constant").
- Keep the tone of the surrounding pages. Don't add exclamations or marketing.
- Describe the physics and the method where the reader needs them (what `twiss` computes and
  how), but not software internals: say what a feature does for the reader in plain words (e.g.
  "with 32-bit coordinates all numbers in the tracking are 32-bit", not how they are lowered).

**Precision.** Every statement must be exactly true, including its scope:
- Name the conditions: tunes exist only for periodic beamlines (an open line has phase advances);
  the normal-form motion is "a rotation in each mode (a damped one with radiation damping)";
  "rotation by the tunes" is wrong at higher orders (the angle depends on the amplitude); the
  actions J "measure" the amplitudes (they are not the amplitudes); resonance driving terms are
  part of the nonlinear motion, not "a dependence on the amplitudes".
- Describe what the code really does. `twiss` does not "bring the map to normal form": it finds
  the closed orbit, computes the map once through the beamline as a Taylor series, computes its
  normal form (tunes and normalizing map), then tracks the normalizing map through the beamline
  and computes the lattice functions from it. Read the source before describing a method.
- Never overclaim. Measure before stating ("more knobs than targets: the knobs change as little
  as possible" was false for the default solver), and quantify approximations ("hardly depends on
  the misalignments (here 2e-4 of its size)" instead of "does not depend").
- Distinguish derivatives from Taylor coefficients, a column from a summary value, the stored
  value of a parameter from a derived one.
- Every number has its units; every printed value in a comment matches the output.

**Terminology** (use consistently on every page and in docstrings):
- "Twiss parameters" are only β, α and γ = (1+α²)/β. Phase advances, dispersion, coupling and
  tunes are not: say "lattice functions", "optics", "linear optics", or "what `twiss` computes".
- "Periodic" (or "closed") beamline, never "ring" for it: one cell of a ring can be analyzed as a
  periodic beamline by itself. "Ring" only for an actual ring. Open line: a transfer line, or a
  section of a ring given starting values.
- "Summary values" for the values for the whole beamline (`tw.q1`, `tw.etac`, ...); "columns" for
  the table; "normalizing map"; "mode 1 (horizontal-like)", "mode 2 (vertical-like)", "mode 3
  (longitudinal-like, synchrotron motion, only with RF cavities)".
- δ is the relative momentum deviation, the coordinate `pz`. "Branch" coordinates: a `Beamline`
  is a branch of a lattice (Beamlines' `Branch`), so the term is kept on the physics pages.
- One name per concept on a page; define it once.

**Nothing before its introduction.** Read the page in order and check that every term, keyword
argument, function, variable and symbol is introduced before it is used, in text and in code
cells (found on `twiss.md`: `chrom` used in a cell before it was explained, `rf_on`, "the map",
γ in `etac` (say "the Lorentz factor"), "invariant spin field"). Don't reuse a variable name for
something else later on the same page (`twc` was the chromatic table and later a cell's table;
redefining `qf` in a later section changed an earlier example's digits): give new objects new
names (`qa`, `twcell`, `thincell`). A later section may use objects from Basics.

**Examples (code cells).**
- Physically sensible values. Checked cases: an RF cavity with a realistic voltage for its length
  and a frequency that is a harmonic of the revolution frequency (`harmon=15`); `dE_ref` equal to
  the energy gain (the old example had 100 times the voltage); "another species" with the same
  sign of charge as the reference (muons in an electron ring, not positrons).
- The plain call, with the result in a comment next to it (`tw.beta1[qd] # 6.695 at the beginning
  of qd`). Round comment values to what matters (4 digits) unless the digits are the point. No
  helper functions; plain `for` loops, not `foreach(...) do`.
- Use the `remove-cell` tag only for setup the reader does not need (and save and restore real
  values there, never hard-code numbers that can drift).
- Prefer an example that also checks something independent where it is cheap (e.g. `fma.md`
  compares NAFF of tracked particles with the tune of `twiss`; `twiss.md` compares the `h3000`
  column with the thin-sextupole formula).

**Verify every claim.**
- Run every code cell of a changed page as a script (`awk` the `{code-cell}` blocks; to print the
  commented values, wrap each commented expression line in `println`), and compare every number
  in the text and comments with the output.
- Check every statement by running it, including "X cannot be done" and "X is an error", and
  check against the source that it is what happens and why (e.g. that the parent element of a
  placement errors for another reason than the one the test comment claimed).
- Check physics against an independent result: finite differences (with a convergence check in
  the step), tracking (orbits at ±δ, NAFF), Tao/Bmad, analytic formulas. Record the facts and
  numbers in this file (a physics guide is planned: keep derivations, conventions and measured
  facts here, e.g. the fixed-point refinement of `twiss`, the RF phase convention, the slip
  factor, the `h` columns, the detuning check against NAFF).
- Check docstrings against the code while documenting (`phase_trombone!` described its stored
  parameters wrongly) and fix them in the same round. Bugs found this way are fixed with a test
  and a negative control (or reported when they need a decision), not documented around. When
  the maintainer decides a behavior is intended (e.g. a knob replaces a deferred expression, as
  setting the parameter does), document it as the rule.
- Say in the report what was verified and how, and what was not (e.g. CUDA could not be tested).

**The maintainer's own edits.** They often edit a page while it is being reviewed. Their wording
is the source of truth: keep it, build around it, and only fix what is wrong (a typo, an inverted
condition such as "mode 3 if no RF cavities", a missing word) and say so in the report. Rewrap
their lines to 100 characters without changing words. When told "do it like I would", apply all
of the above to the whole page, not only to the paragraph mentioned.

**Process for a page.**
1. Back up the page to the scratch directory.
2. Read it in full as a newcomer and as a careful reader; list every problem (wrong, unclear,
   used before introduced, terminology, overclaims, examples that are not physical).
3. Run its code and check its claims (above), including against the source.
4. Rewrite: keep correct text unless the structure or style rules require a change; then reread
   the whole page imagining the maintainer's objections (run-on paragraphs, explanations before
   examples, undefined terms, awkward wording for experts, vague words, "ring" for periodic, a
   misused "Twiss parameters", an unverified number) and fix them.
5. Rerun the code; check line lengths (≤ 100 characters except tables); check links and labels
   with Sphinx (below).
6. Report concisely: what changed and why, what was verified and how, what is open.

**Mechanics.**
- MyST heading anchors are off: link to `(label)=` targets (`[Open lines](#twiss.open)`), and
  give sections that are linked to a label (`(twiss.orders)=`). Link to a page by file
  (`[Twiss](twiss.md)`) when its label has the page's name (`(twiss)=`): `#twiss` is ambiguous.
- Pages with code cells have the jupytext front matter and the `remove-cell` setup cell (see
  `twiss.md`).
- Check links, labels and MyST syntax with Sphinx alone, without executing code and without
  touching `docs/Project.toml` or `~/Library/Jupyter` (which `docs/build.py` changes): a venv in
  the scratch directory with `docs/requirements.txt`, an `rsync` copy of the repository, and
  `sphinx-build -b html -D nb_execution_mode=off -q . <out>` in its `docs/src`. Only the
  `{docstring}` lookups warn then (they need the docs' Julia environment). For the full build
  as CI runs it (every page executed, docstrings, Documenter), run `docs/build.py` on an `rsync`
  copy that is made a throwaway Git repository (`git init`, `remote add origin` the GitHub URL, one
  commit: Documenter needs it), with `PATH` starting with a directory holding `pip` and
  `sphinx-build` of the venv and a `julia` script that execs the Julia 1.13 binary by its full
  path (CI uses Julia `1`; a script calling `julia +release` finds itself and loops), with
  `JUPYTER_DATA_DIR`/`JUPYTER_PATH` in the scratch directory (the IJulia kernel goes there), and
  with the local clones developed into the copy's `docs` environment first (unregistered
  versions). Nov 22 2026: builds; the only warning is the highlighting of the intended error's
  traceback on `tracking-methods.md` (relaxed mode, cosmetic). The `{docstring}`
  directive evaluates the name in `Main` after `using SciBmad`: unexported names need the module
  (`{docstring} SciBmad.value_and_jacobian`).
- Every name used in examples must work as written after `using SciBmad` (unexported functions
  are written with `SciBmad.`, e.g. `SciBmad.value_and_jacobian`, which is not exported because
  DifferentiationInterface exports a function of the same name).

### The Claude Code skill (`skills/scibmad/`)

The repository is a Claude Code plugin and its own marketplace (`.claude-plugin/plugin.json`,
`.claude-plugin/marketplace.json`); users install it as described in `installation.md`. The
skill is `skills/scibmad/`: `SKILL.md` (loaded whenever the skill triggers), `references/*.md`
(one per topic, read only when needed), `scripts/overview.jl` (a tested lattice summary),
`evals/` (cases for `claude plugin eval`), `benchmark/` (the ring design tutorial chapters with
their `match!` cells removed, graded against the solution lattices; README there).

How it was built (Nov 2026), and how to change it:
- **Update it in the same PR as the API change** it describes, and keep the version stamp in
  `SKILL.md` and `plugin.json` at the SciBmad release.
- **General, never fitted to a test.** A failing benchmark or eval run is fixed by the general
  rule it exposes (how to choose knobs, rows, orders; what to report), never by task-specific
  values or steps. Keep non-tutorial cases in the test set.
- **Content is what the docs do not say:** pitfalls, conventions and the maintainer's judgment
  (e.g. trust the code's output and cross-check only when asked; odd results → more `n_steps`
  first; optimize with `match!`; complex terms as separate real and imaginary entries; s-
  dependent driving terms for the DA; dispersion with RF is an approximation; never describe
  `twiss` as evaluating integrals; read the user's lattice file first). API details belong to
  docstrings and the online docs: link them, don't copy them. Users have no local clone.
- **Every snippet is run as written** before it goes in, and every claim checked against the
  code (a wrong `angle`, an invalid splat and a wrong `minimize` form were caught this way).
- **Lean:** `SKILL.md` is the per-use cost; keep it to pitfalls and judgment, push detail into
  the references, and put repeated exploration into tested scripts.
- **Testing costs the maintainer's usage limits:** state the cost before running anything.
  Prefer one targeted run on Sonnet at low effort over broad reruns; scan the run transcripts
  for reads of the solution files; validate with `claude plugin validate .`.
