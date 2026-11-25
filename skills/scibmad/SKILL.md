---
name: scibmad
description: Writes correct Julia code with SciBmad, the bmad-sim accelerator physics package (lattices with LatElement/Beamline/@elements/DefExpr, optics with twiss, tracking with track, closed orbit, dynamic aperture, matching with TwissExpr/@knob/match!). Use it whenever someone builds or analyzes an accelerator lattice in Julia, asks for tunes, beta functions, dispersion, chromaticity, resonance driving terms, detuning, dynamic aperture optimization, spin tune, particle or spin tracking (CPU/GPU), orbit correction or lattice matching with SciBmad or the bmad-sim Julia packages, converts a Bmad/Tao/MAD-X/Xsuite lattice or workflow to SciBmad, or hits a SciBmad error, even if the package is not named.
---

# SciBmad

Written for SciBmad 0.8 (Nov 2026). `using SciBmad` re-exports Beamlines (elements, beamlines,
deferred expressions), BeamTracking's API, NonlinearNormalForm, GTPSA and `Species`.
Docs: https://bmad-sim.github.io/SciBmad.jl/ (sources:
https://github.com/bmad-sim/SciBmad.jl/tree/main/docs). Most users have no local clone. For
anything not covered here read the docstring (`@doc twiss`; `??twiss` lists all columns) or the
docs page; don't guess names.

Start a task on an existing lattice with `include(joinpath(<this skill's dir>, "scripts",
"overview.jl")); overview(ring)`: one call prints the reference particle, length, elements by
kind, families (shared parents and the context variables driving `DefExpr`s), misalignments,
RF and radiation, tunes and chromaticities. Read only the reference file the task needs:

| File | For |
|---|---|
| `references/lattice.md` | elements, parameter names, beamline indexing/search, `DefExpr`/contexts, trombones, Bmad/MAD-X name table |
| `references/tracking.md` | `track` and its results, spin, GPU/Float32, tracking methods and steps, radiation, IBS, ramping, `BatchParam`, closed orbit, `dynamic_aperture`, `naff` |
| `references/twiss.md` | summary values, columns, locations, orders (`chrom`, `order`), RF, spin, open lines, parts, `h` terms, GTPSA parameters |
| `references/optimize.md` | `TwissExpr`, knobs, targets, `match!`, orbit correction, derivatives, other optimizers |
| `references/nonlinear.md` | driving terms and detuning: computing and ranking them; dynamic aperture optimization |

```julia
using SciBmad
@elements begin                       # names each element after its variable
  qf = Quadrupole(Kn1=0.36, L=0.5)    # Kn1 [1/m^2], normalized by the reference rigidity
  sf = Sextupole(Kn2=1.2, L=0.2)
  d  = Drift(L=0.1)
  b  = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-1.2, L=0.2)
end
fodo = Beamline([qf, sf, b, d, qd, sd, b, d], species_ref=Species("electron"), pc_ref=18e9)
tw = twiss(fodo); tw.q1, tw.beta1[qd]               # summary value; column at qd's entrance
twiss(fodo, chrom=2).dq1                            # chromaticity needs chrom=2
te = TwissExpr(fodo)
match!(vary=[@knob(sf.Kn2), @knob(sd.Kn2)], targets=[te.dq1 == 1, te.dq2 == 1])
res = track(fodo, v0=[1e-3 0 0 0 0 1e-3], n_turns=10); res.x[turn=10, particle=1]
```

## Pitfalls

- Coordinates are Bmad's `x, px, y, py, z, pz` (`pz` = δ). `v0` is a **float** matrix, one row
  per particle (`zeros(1, 6)`; an all-integer matrix fails with `InexactError`).
- All kinds are one type `LatElement`; the kind sets only `kind` and a default
  `tracking_method`. Any element takes any parameter, but a solenoid field needs
  `SolenoidKick()`. `angle` is write-only (sets `g_ref` and `Kn0`).
- Placements are children of the element: `qf.Kn1 = …` changes all of them. `bl[qf]` is the
  single placement (error if 0 or ≥2), `bl[[d]]` all. Read `DefExpr` parameters through the
  placement (`bl[qf].Kn1`): `qf` itself evaluates with `DEFAULT_CONTEXT`.
- Strength forms `Kn1`, `Kn1L`, `Bn1`: the one set last is stored. With `Kn1` stored, `Bn1`
  scales with the reference energy, and vice versa.
- An integer index of a result is a beamline index, not a row (`tw.beta1[5]`); `end` is the end
  of the beamline. Raw rows: `tw.df`.
- Orders: `chrom=2` for chromaticity, `order=3` for detuning. Without them `twiss` picks orders
  from `cols`, but summary values (`dq1`) are not columns. `TwissExpr` picks orders itself.
- RF makes the beam 6D: chromatic quantities need `rf_on=false`.
- `q1[delta=2]` is ∂²q₁/∂δ² (`dq1_2`); `getterm(q1, delta=2)` the coefficient (÷2!).
- A `DefExpr` parameter is varied through its context variable: `@knob(bl.context.kq)`, not
  `@knob(qf.Kn1)` (a knob sets its parameter and would replace the expression).
- A knob on a `Kn1 == 0` quadrupole with `MatrixKick` aborts Julia (GTPSA `sqrt(0)`): use
  `DriftKick()` for it or a nonzero start. Mixing GTPSA Descriptors also aborts: after setting
  Taylor-series parameters by hand, `scalarize!(bl)`.
- `track(...; at=[...])` saves only those locations (add `length(bl.line)+1` for the end); an
  exit is not an `at` entry (save the next element). The state after the last turn is always in
  `res.bunch.coords` (`v`, `q`, `state`): don't add save points for final values.
- `find_closed_orbit(bl; v0)` overwrites `v0`. Deferred expressions are evaluated once per
  call; use `Time()` for changes during tracking.
- `SciBmad.value_and_jacobian` is not exported. First calls compile: time the second call.

## Explaining results

- `twiss` uses map-based perturbation theory with Lie-series normal forms (E. Forest). Never
  describe it as evaluating integrals over the lattice; no need to mention this unprompted.
- Tunes are normal-form tunes; lattice functions use the Edwards-Teng/Sagan-Rubin definitions.
  Say horizontal-like/vertical-like (modes 1, 2) unless the lattice is uncoupled.
- Dispersion is d(closed orbit)/dδ, well defined with `rf_on=false`. With RF on, `dx` is an
  approximation valid when the synchrotron tune is much smaller than the whole transverse tunes.
  Sextupoles change the higher-order dispersion, and the linear one when the orbit is off-center.
- Know conventions instead of hedging: a positive bend `angle` bends toward −x; an electron spin
  along z through a bend ends with s_x = −sin(aγθ).
- Bmad and SciBmad are different codes; simple lattices agree. Resolve differences by testing
  (converge steps: Tao tracks thick sextupoles in one step by default), not by guessing.
- Flag implausible results (e.g. sextupoles 20× stronger than before a match).
- Report what was done and found. Check properties directly (`rawget(qf, :Kn1) isa DefExpr`),
  not by perturbing and restoring the lattice; omit your own intermediate mistakes.

## Working style

- Read the user's lattice file before computing: it shows the families, the context variables
  that are the natural knobs, markers, and intent, at a fraction of the cost of probing objects.
- Trust SciBmad's output: no NAFF, finite-difference or tracking cross-checks unless the user
  asks or a result looks wrong. Then first increase `n_steps` of strong/long elements.
- Optimizations go through `match!`/`MatchProblem` (exact derivatives), never
  finite-difference loops. Confirm a match from its result (`res.converged`, and `value` of the
  target expressions at their locations) before reporting it.
- Keep calculations minimal: `at=[]` or `at=[1]` when only summary values or one row are
  needed, the lowest orders that resolve the quantity, cheap `dynamic_aperture` settings first.
- Units: m, eV (`E_ref`; `pc_ref` = p·c in eV), T·m (`p_over_q_ref`), rad, tunes and phases in
  units of 2π, Hz, V.
- Python (juliacall): `jl.seval("using SciBmad")`, `jl.twiss(...)`; `x[turn=3]` is
  `jl.getindex(x, turn=3)`; macros only inside `jl.seval`.
- Changing SciBmad's own source: follow the repository's `AGENTS.md`.
