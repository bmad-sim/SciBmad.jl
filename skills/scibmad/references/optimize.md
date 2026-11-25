# Matching, optimization and derivatives

Docs pages: `optimize.md`, `parametric-nf.md`, the ring design tutorial (`ring-design/*.md`,
lattice files in `ring-design/lattices/`), which builds a whole ring by successive matches.

## Contents
1. `TwissExpr` and expressions
2. `match!`: targets, knobs, minimize
3. Knobs
4. Orbit correction and response matrices (LOCO)
5. Open lines and chaining beamlines
6. Values and exact derivatives
7. Other optimizers (`MatchProblem`)
8. How it works and its caveats

## 1. `TwissExpr`

`TwissExpr(bl; twiss keywords...)` is a lazy `twiss`: it is read exactly like a `twiss`
result, but nothing is computed until `value`/`match!` needs it, and only the rows, columns and
orders the expressions need are computed. Don't pass `at`, `cols`, `base_cols`,
`as_taylor_series`, `GTPSA_descriptor` (chosen automatically); do pass e.g. `rf_on=false`,
`initial`, `a_initial`, `start`, `stop`.

```julia
tw = TwissExpr(fodo)
tw.q1, tw.dq1, tw.q1[J1=1], tw.beta1[qd], tw.beta1[qd, :exit], tw.beta1[[b], :],
tw.beta1["q*"], tw.dbeta1[qd], tw.h3000[1], tw.phi1[qd] - tw.phi1[qf]   # all expressions
e = tw.beta1[qd]^2 - 5*mean(tw.alpha1[[b], :])   # arithmetic; sum/mean/maximum/minimum reduce
value(e)                                          # compute
value(maximum(tw.beta1[:]))
value(lazy(() -> fodo[1].Bn1))                    # any other quantity of the lattice
```

A location with several places is a vector expression; arithmetic and comparisons apply
element-wise. `h` terms are complex: `value` returns complex, but targets and `minimize` need
their real and imaginary parts as separate entries (`real(h)`, `imag(h)`, or
`reim(h)...`), never `abs(h)`. Chromatic quantities need `TwissExpr(ring; rf_on=false)`
on a ring with RF.

## 2. `match!`

```julia
tw = TwissExpr(fodo)
res = match!(vary    = [@knob(qf.Kn1), @knob(qd.Kn1)],
             targets = [tw.q1 == 0.25, tw.q2 == 0.25])
res.converged, res.iterations, res.x
undo!(res)            # back to the starting knob values
match!(res)           # continue a match that stopped early
```

With one beamline the targets may be a function of its `TwissExpr`:
`match!(fodo; vary=[...], targets = tw -> [tw.q1 == 0.25], twiss=(rf_on=false,))`.

Targets:
- `expr == v`, `expr <= v`, `expr >= v` (with a `TwissExpr`, `==` builds a target, it does not
  compare). Vector expressions: `tw.beta1["q*"] <= 30` is one per location. Inequalities only act
  while violated.
- `Target(expr, v; tol=1e-6, weight=10)`, `Target(expr; lower=5, upper=30, weight=0.1)`.
  Default `tol=1e-8`. Give targets of very different size (tune vs β) weights that make their
  residuals comparable.
- `minimize = [exprs...]`: sum of squares made small together with the target residuals. Give
  targets that must hold a larger weight:

```julia
twc = TwissExpr(ring; rf_on=false)
match!(vary = [@knob(s1.Kn2), @knob(s2.Kn2), @knob(s3.Kn2), @knob(s4.Kn2)],
       targets = [Target(twc.dq1, 1; weight=100, tol=1e-4), Target(twc.dq2, 1; weight=100, tol=1e-4)],
       minimize = [reim(twc.h3000[1])..., reim(twc.h2100[1])..., reim(twc.h1020[1])...])
```

Solver: Gauss-Newton steps from the SVD of the exact Jacobian (minimum-norm steps when there are
more knobs than targets; least squares when there are more targets), with step halving and
damped fallbacks; knobs stay within bounds. `solver=:levenberg_marquardt` uses BatchSolve's LM.
Keywords: `maxiter` (50), `verbose`, `solver`, `rcond`. It finds a *nearby* solution: a bad
start, an unstable region, or too-different target scales can stall it. `print(res)` shows all
knobs.

## 3. Knobs

```julia
@knob qf.Kn1                          # element parameter (all placements of qf together)
@knob qf.Kn1 lower=0 upper=0.6        # bounds
@knob qf.Kn1 step=0.01                # max change per iteration
@knob d.L                             # lengths, offsets, RF voltage, ... anything settable
@knob ring.context.kq                 # a context variable used by DefExprs
@knob ring.context.kicks              # vector variable: one knob per entry
@knob ring.context.kicks[2]           # one entry
Knob(qf, :Kn1)                        # function form (also Knob(ctx, :kicks, j))
[Knob(k, :Kn0L) for k in hkickers]    # many elements
```

To vary several different elements together (or any combination), make their parameters
`DefExpr`s of one context variable and vary the variable. A knob is also an expression:
`@knob(qf.Kn1) + @knob(qd.Kn1) <= 0.05`.

**A parameter controlled by a `DefExpr` is varied through its context variable**: with
`qf.Kn1 = DefExpr(c -> c.kq)` in the beamline `bl`, the knob is `@knob(bl.context.kq)`, not
`@knob(qf.Kn1)`. A knob sets its parameter, so a knob on `qf.Kn1` itself would replace the
expression by a number (also in `value`/`value_and_jacobian`) and break the dependence.

## 4. Orbit correction and response matrices

The orbit is a column like any other:

```julia
twr = TwissExpr(ring)
orbit = [twr.x[bpm] for bpm in bpms]           # or twr.x[[bpm]] / twr.x["bpm*"]
res = match!(vary=[@knob(ring.context.kicks)], minimize=orbit)
R = [deriv(twr.x[bpm], Knob(ring.context, :kicks, j)) for bpm in bpms, j in 1:ncorr]
value(R)                                       # orbit response matrix
match!(vary=[@knob(ring.context.dk)], targets=vec(R .== R_measured))   # LOCO-style fit
```

Only the orbit is computed for orbit-only expressions (fast). If errors make the ring unstable,
correct with sextupoles off first, then turn them on in steps. Phase trombones must be removed
before orbit correction and rebuilt afterwards (`lattice.md`).

**Errors, then correction** (e.g. "apply these errors and correct the orbit", "turn on
radiation and correct the orbit"):

```julia
err = Beamline(deepcopy.(ring.line), species_ref=ring.species_ref, pc_ref=ring.pc_ref)
for q in err["Quadrupole::*"]                    # each placement is now independent
  q.x_offset = σx*randn(); q.y_offset = σy*randn()   # truncate the Gaussian if asked
end
for e in err.line; e.rad_damping = true; end     # radiation: the orbit gets the energy sawtooth
te = TwissExpr(err)                              # RF on: 6D orbit with radiation
match!(vary=[Knob(k, :Kn0L) for k in err["HKicker::*"]], minimize=te.x["bpm*"])
match!(vary=[Knob(k, :Ks0L) for k in err["VKicker::*"]], minimize=te.y["bpm*"])
```

- Errors go on independent elements: placements pass sets to their shared parent, so offsetting
  `ring[[q]][i]` moves every copy. Use `deepcopy.(ring.line)` (keeps the original as the
  design), or per-element `DefExpr`s of context vectors (`x_offset = DefExpr(c -> c.dx[i])`)
  when the errors themselves should be knobs (fits, LOCO).
- Horizontal correctors are `Kn0L` (or `Kn0`), vertical ones `Ks0L`; any element can be a
  corrector (`Multipole(Kn0L=…)`, a corrector coil on a quadrupole).
- With radiation the residual orbit includes the sawtooth (δ times dispersion), which
  correctors can reduce only in the least-squares sense; say so rather than chasing it.
- Errors large enough to lose the closed orbit or stability: correct with sextupoles off
  first, then ramp them on in steps; remove phase trombones before correcting.
- Set a seed (`Random.seed!`) and report it, and the error distribution, with the results.

**Commissioning-style sequence** (large errors, no closed orbit yet): sextupoles off (a
context scale, `Kn2 = DefExpr(c -> c.sx*k2)`, `ctx.sx = 0`); thread the first pass as an open
line from the design start, `to = TwissExpr(err; initial=twiss(ring, at=[1]).df[1, :])`, and
`match!` the trajectory `to.x["bpm*"]`/`to.y["bpm*"]` with the correctors; then correct the
closed orbit with the periodic `TwissExpr(err)`; then ramp `ctx.sx` (and higher multipoles) up
in steps, correcting the closed orbit at each step.

**Realistic correction** (what a control room does: the response comes from the *model*, not
the machine; readings are noisy; correctors are limited): run `match!` on a design copy to find
the kicks that cancel the measured orbit, and add them to the machine; repeat.

```julia
model = Beamline(deepcopy.(ring.line), species_ref=ring.species_ref, pc_ref=ring.pc_ref,
                 context=copy(ring.context))                   # design model, no errors
tm = TwissExpr(model); mk, mb = model["HKicker::*"], model["bpm*"]
meas = value(te.x["bpm*"]) .+ σ_bpm .* randn(length(mb))    # BPM readings of the machine
for k in mk; k.Kn0L = 0.0; end
match!(vary=[Knob(k, :Kn0L; lower=-kmax, upper=kmax) for k in mk],
       minimize=[tm.x[p] + m for (p, m) in zip(mb, meas)], rcond=1e-2, maxiter=1)
for (k, m) in zip(err["HKicker::*"], mk); k.Kn0L += m.Kn0L; end   # apply to the machine
```

`rcond` is the SVD cutoff (singular values below `rcond` × the largest are dropped) and the knob
bounds are the corrector limits; one `maxiter=1` step is one linear SVD correction.

## 5. Open lines and chaining

```julia
tl = TwissExpr(line; a_initial=tw.a[end])   # start with the periodic functions of another TwissExpr
match!(vary=[@knob(m1.Kn1), @knob(m2.Kn1), @knob(m3.Kn1), @knob(m4.Kn1)],
       targets=[tl.beta1[end] == 10, tl.beta2[end] == 10, tl.alpha1[end] == 0, tl.alpha2[end] == 0])

ini = Context(beta1=10.0, alpha1=0.0, beta2=10.0, alpha2=0.0)   # initial values as knobs
ti = TwissExpr(line; initial=ini)
match!(vary=[@knob(ini.beta1), @knob(ini.alpha1), @knob(ini.beta2), @knob(ini.alpha2)],
       targets=[ti.beta1[end] == 5, ti.beta2[end] == 5, ti.alpha1[end] == 0, ti.alpha2[end] == 0])
```

Because the chained start is recomputed each time, knobs of the upstream cell can be varied in
the same match. A ring can be built piece by piece: arc cell, dispersion suppressor
(`tl.dx[end] == 0, tl.dpx[end] == 0`), straight, then tunes and chromaticities of the ring.

## 6. Values and derivatives

```julia
v, J = SciBmad.value_and_jacobian([tw.q1, tw.q2], [@knob(qf.Kn1), @knob(qd.Kn1)])
# J[i, j] = ∂(expr i)/∂(knob j), exact
value(deriv(deriv(tw.q1, kf), kf))            # second derivative, as an expression
value(deriv(tw.dh2000[1], @knob(sf.Kn2)))     # complex
```

`value_and_jacobian` is not exported (DifferentiationInterface has the same name). No truncation
order has to be chosen. Derivatives (`deriv`) can be targets too.

## 7. Other optimizers

```julia
prob = MatchProblem(vary=[@knob(qf.Kn1, lower=0.2, upper=0.6), @knob(qd.Kn1, lower=-0.6, upper=-0.2)],
                    targets=[tw.q1 == 0.25, tw.q2 == 0.25])
prob.x, prob.lower, prob.upper
value(prob, x)                        # residual vector
SciBmad.value_and_jacobian(prob, x)   # residuals + exact Jacobian
penalty(prob, x)                      # sum of squares, Inf where the lattice is unstable
setknobs!(prob, x_best)               # these functions leave the knobs at the last x: reset!
match!(prob)                          # polish a global search result
```

Examples in `optimize.md`: NonlinearSolve `NewtonRaphson` (`NonlinearFunction` with `jac`),
Optim `LBFGS` (gradient `2 J'r`), Metaheuristics `DE` on `penalty` within `[prob.lower prob.upper]'`.

## 8. Caveats

- Knobs become GTPSA parameters while an evaluation runs; the lattice is restored afterwards.
  Don't evaluate the same lattice from several threads at once.
- Contexts and `DefExpr`s that knobs act through must not be typed `Context{Float64}` /
  `DefExpr{Float64}`.
- A knob on a quadrupole with `Kn1 == 0` tracked with `MatrixKick` aborts Julia (GTPSA
  `invalid domain sqrt(+0)`, no exception). Use `tracking_method = DriftKick()` for that
  element (its derivative agrees with central differences) or a nonzero starting strength.
- `match!` on a periodic lattice needs a closed orbit and stable linear optics at every
  evaluated point; failed points count as infinite residuals and are stepped back from.
