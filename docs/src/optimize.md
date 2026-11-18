---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Julia
  language: julia
  name: julia
---

(optimize)=
# Matching and Optimization

```{code-cell} julia
:tags: [remove-cell]
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

Matching means varying parameters of a lattice (the "knobs") until quantities such as the tunes,
the beta functions at some point, or the chromaticities have the values you want. In SciBmad you
write down the quantities and their goals, say what may be varied, and call `match!`.

Most optimizers need the derivatives of the quantities with respect to the knobs. In
non-differentiable codes these are computed with finite differences, which are approximate and
need one more calculation per knob. Because SciBmad is fully differentiable, `match!` gets the
*exact* derivatives with respect to all knobs from a single calculation.

## A First Match

Let's make the phase advance of a FODO cell $90^\circ$ in both planes by varying its two
quadrupoles.

```{code-cell} julia
using SciBmad

@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.2)
  b = SBend(L=6.5, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-1.2, L=0.2)
end

fodo = Beamline([qf, sf, d, b, d, qd, sd, d, b, d],
        species_ref=Species("electron"), pc_ref=18e9)
```

First we need a way to refer to the Twiss quantities of `fodo`. `TwissExpr` gives exactly that: it
takes the same arguments as [`twiss`](twiss.md), and its quantities are read exactly like those of
the result of `twiss`, but nothing is computed yet.

```{code-cell} julia
tw = TwissExpr(fodo)
```

The tunes of the cell are then `tw.q1` and `tw.q2`. A target is written with `==`, and a knob with
`@knob`:

```{code-cell} julia
res = match!(vary    = [@knob(qf.Kn1), @knob(qd.Kn1)],
             targets = [tw.q1 == 0.25, tw.q2 == 0.25])
```

The quadrupole strengths have been changed, and `twiss` confirms the tunes:

```{code-cell} julia
twiss(fodo, at=[])
```

`undo!(res)` would set the knobs back to where they were, and `match!(res)` continues a match
that stopped early.

If only one beamline is involved, it can also be given to `match!` directly. The targets are then
written as a function of its Twiss quantities (the keyword arguments for `twiss`, if any, are
given with `twiss=(rf_on=false,)`):

```julia
match!(fodo; vary = [@knob(qf.Kn1), @knob(qd.Kn1)],
             targets = tw -> [tw.q1 == 0.25, tw.q2 == 0.25])
```

Now let's make the chromaticities $+1$ in both planes with the sextupoles. The chromaticities are
`tw.dq1` and `tw.dq2`. They are only defined without synchrotron oscillations, so in a ring with
RF cavities use `TwissExpr(ring; rf_on=false)`.

```{code-cell} julia
res = match!(vary    = [@knob(sf.Kn2), @knob(sd.Kn2)],
             targets = [tw.dq1 == 1, tw.dq2 == 1])
```

You do not need to set any truncation order: each quantity knows the order it needs.

## Quantities and Locations

Every summary value and every scalar column of [`twiss`](twiss.md) is available from a `TwissExpr`,
at the same locations and written in the same way as for the result of `twiss` (see
[Reading the results](twiss.md)): `tw.q1`, `tw.dq1`, `tw.q1[J1=1]`, `tw.beta1[qd]`,
`tw.beta1[6]`, `tw.beta1[qd, :exit]`, `tw.beta1[b, :]`, `tw.beta1[end]`, `tw.beta1["q*"]`,
`tw.beta1[(2.0, 10.0)]`, and so on. Chromatic derivatives (`tw.dbeta1[qd]`, `tw.dx_2[1]`) and the
`h` terms (`tw.h3000[1]`) too. The `h` terms are complex numbers: `value` gives them as such, but a
target or `minimize` needs `abs`, `real` or `imag` of them. Unlike for `twiss`, you do not need to choose truncation orders or rows: each quantity
knows what it needs.

These expressions can be combined with ordinary arithmetic and functions. For example, the square
of the beta function at `qd` minus five times the average of the alpha function through the bends:

```{code-cell} julia
e = tw.beta1[qd]^2 - 5*mean(tw.alpha1[b, :])
```

`value` computes an expression. Only the rows and columns of `twiss` that it uses are computed:

```{code-cell} julia
value(e)
```

When a location is several places, the result is a vector, and arithmetic and comparisons apply
to each element. `sum`, `mean`, `maximum`, and `minimum` reduce it to one expression:

```{code-cell} julia
value(tw.beta1["q*"]) # beta1 at qf and at qd
```

```{code-cell} julia
value(maximum(tw.beta1[:]))
```

A phase advance between two points is simply a difference:

```{code-cell} julia
value(tw.phi1[qd] - tw.phi1[qf]) # in units of 2π
```

Anything else can be included with `lazy`: `lazy(f)` is the result of the function `f()`, computed
from the lattice itself when it is needed. For example, the field gradient of `qf` in this
beamline:

```{code-cell} julia
value(lazy(() -> fodo[1].Bn1))
```

## Targets

Besides `==`, a target can be an inequality:

```julia
tw.beta1[qd] <= 30
tw.beta1["q*"] <= 30            # at each of these elements
maximum(tw.beta1[:]) <= 100     # everywhere
```

An inequality only acts while it is violated. A target counts as met when it is within its
tolerance, which is `1e-8` unless you set it with `Target`. A `weight` makes a target count more
or less than the others while they cannot all be met:

```julia
Target(tw.q1, 0.25; tol=1e-6)
Target(tw.beta1[qd]; lower=5, upper=30, weight=0.1)
```

There may be fewer, as many, or more targets than knobs. With more knobs than needed, the knobs
are changed as little as possible. With more targets than can be met, `match!` returns the
closest it can get (in the least-squares sense) and reports the match as not converged.

`minimize` is for quantities that should just be as small as possible, for example driving terms
of a ring with four sextupole families:

```julia
twc = TwissExpr(ring; rf_on=false)
match!(vary = [@knob(s1.Kn2), @knob(s2.Kn2), @knob(s3.Kn2), @knob(s4.Kn2)],
       targets = [Target(twc.dq1, 1; weight=100, tol=1e-4), Target(twc.dq2, 1; weight=100, tol=1e-4)],
       minimize = [abs(twc.h3000[1]), abs(twc.h2100[1]), abs(twc.h1020[1])])
```

The sum of the squares of the expressions in `minimize` and of the distances from the targets
(each times its weight) is made as small as possible. If they cannot all vanish, the result
balances them, so give targets that must hold a larger weight, as here.

## Knobs

A knob is any parameter that can be set with `object.parameter = value`:

```julia
@knob qf.Kn1                           # an element parameter
@knob qf.Kn1 lower=0 upper=0.6         # with bounds
@knob d.L                              # also lengths
@knob ring.context.kqf                 # a variable used by deferred expressions
@knob ring.context.coilstrengths       # a vector variable: one knob per entry
@knob ring.context.coilstrengths[2]    # one entry of it
```

`@knob qf.Kn1` is the same as `Knob(qf, :Kn1)`. Setting a parameter of an element changes it
everywhere the element is placed (see [Defining a Beamline](beamline.md)), so `@knob(qf.Kn1)`
varies all `qf`s together. To vary several different elements together, or to build any other
knob, use a [deferred expression](defexpr.md) and vary the variable it depends on:

```{code-cell} julia
qf.Kn1 = DefExpr(c -> c.kq)
qd.Kn1 = DefExpr(c -> -c.kq)
fodo.context.kq = 0.38

res = match!(vary = [@knob(fodo.context.kq)], targets = [tw.q1 == 0.25])
```

```{code-cell} julia
:tags: [remove-cell]
qf.Kn1 = 0.3804445782410782
qd.Kn1 = -0.3805887076051259
```

A variable can also be a vector, e.g. the strengths of many correctors, each set by a deferred
expression such as `DefExpr(c -> c.coilstrengths[3])`. `@knob(ring.context.coilstrengths)` then
varies every entry; the result lists them as `coilstrengths[1]`, `coilstrengths[2]`, ... (as
many as fit in the terminal; `print(res)` shows all of them).

`step` limits how much a knob may change in one iteration (`@knob qf.Kn1 step=0.01`). A knob is
also an expression, so it can appear in targets: `@knob(qf.Kn1) + @knob(qd.Kn1) <= 0.05`.

## Orbit Correction

The closed orbit is a Twiss quantity like any other (`tw.x[bpm]`, `tw.y[bpm]`, ...), so orbit
correction is a match too. Here is a ring of ten cells, each with a misaligned quadrupole, a
horizontal corrector and a BPM. The strengths of all correctors come from one vector variable of
the context, `kicks` (and gradient errors of the quadrupoles from another, `dk`, used below):

```{code-cell} julia
ncell = 10
offsets = 1e-4 .* [0.8, -1.2, 0.3, 1.5, -0.6, 0.9, -1.1, 0.2, 1.3, -0.4] # in m
elements = LatElement[]
bpms = LatElement[]
for i in 1:ncell
  bpm = Marker()
  push!(bpms, bpm)
  append!(elements, [Quadrupole(Kn1=DefExpr(c -> 0.36 + c.dk[i]), L=0.5, x_offset=offsets[i]),
                     Multipole(Kn0L=DefExpr(c -> c.kicks[i])), Drift(L=0.2), bpm,
                     SBend(L=6.5, angle=pi/ncell), Drift(L=0.2),
                     Quadrupole(Kn1=-0.36, L=0.5), Drift(L=0.2),
                     SBend(L=6.5, angle=pi/ncell), Drift(L=0.2)])
end
ring = Beamline(elements; species_ref=Species("electron"), pc_ref=18e9);
ring.context.kicks = zeros(ncell);
ring.context.dk = zeros(ncell);

twr = TwissExpr(ring)
orbit = [twr.x[bpm] for bpm in bpms]
maximum(abs, value(orbit)) # 5.8e-4 m at the BPMs
```

`minimize` makes the orbit at the BPMs as small as possible (in the least-squares sense), varying
every corrector:

```{code-cell} julia
res = match!(vary = [@knob(ring.context.kicks)], minimize = orbit)
maximum(abs, value(orbit)) # 3e-15 m
```

Only the orbit is computed for this (none of the optics at the BPMs), which is faster. The closed
orbit needs stable linear optics: if the errors make the ring unstable, correct the orbit first
with the sextupoles turned off, then turn them on in steps.

The orbit response matrix, the change of the orbit at each BPM per kick of each corrector, is a
matrix of derivatives (`deriv`, see "Values and Derivatives" below), here with respect to the
entries of `kicks`:

```{code-cell} julia
R = [deriv(twr.x[bpm], Knob(ring.context, :kicks, j)) for bpm in bpms, j in 1:ncell]
```

A measured response matrix can then be fitted, as in LOCO. It depends on the focusing of the ring,
not on the misalignments (they shift the orbit, but not its response to a kick, as long as there
are no sextupoles), so it is fitted with gradient errors. Here the "measurement" is the response
matrix of the ring with the gradient errors `dk_true`, and the model, which starts without them,
finds them again:

```{code-cell} julia
dk_true = 1e-3 .* [1.0, -2.0, 0.5, 1.5, -1.0, 0.8, -0.3, 2.0, -1.2, 0.4]
ring.context.dk = copy(dk_true)
R_measured = value(R)
ring.context.dk = zeros(ncell)

res = match!(vary = [@knob(ring.context.dk)], targets = vec(R .== R_measured))
maximum(abs, ring.context.dk - dk_true) # 8e-13: the errors are found again
```

Further knobs fit the same way, for example quadrupole tilts for the coupling (with the vertical
orbit in the matrix) and the calibrations of the BPMs and correctors.

## Lines That Are Not Periodic

For a transfer line, a linac, or a section of a ring by itself, give the Twiss functions at its
start with `initial` or `a_initial` (see [Open lattices](twiss.md)). `a_initial` can be the `a` of
another `TwissExpr`, so that the line continues from another beamline. Here a line of four quadrupoles
follows the FODO cell, and is matched to a waist with beta functions of 10 m at its end:

```{code-cell} julia
@elements begin
  m1 = Quadrupole(Kn1=0.3, L=0.5)
  m2 = Quadrupole(Kn1=-0.3, L=0.5)
  m3 = Quadrupole(Kn1=0.3, L=0.5)
  m4 = Quadrupole(Kn1=-0.3, L=0.5)
  dm = Drift(L=3.0)
end

line = Beamline([m1, dm, m2, dm, m3, dm, m4, dm],
        species_ref=Species("electron"), pc_ref=18e9)

tl = TwissExpr(line; a_initial=tw.a[end]) # starts with the periodic functions of the cell

res = match!(vary    = [@knob(m1.Kn1), @knob(m2.Kn1), @knob(m3.Kn1), @knob(m4.Kn1)],
             targets = [tl.beta1[end] == 10, tl.beta2[end] == 10,
                        tl.alpha1[end] == 0, tl.alpha2[end] == 0])
```

Because the start of the line is taken from the cell each time, quadrupoles of the cell could be
varied in the same match. A whole ring can be built this way piece by piece: match the arc cell,
then a dispersion suppressor that follows it (`tl.dx[end] == 0, tl.dpx[end] == 0`), then the
straight section that follows the suppressor, and finally the tunes and chromaticities of the
complete ring.

The Twiss parameters at the start can be knobs too: give `initial` as a `Context`, and vary its
variables. Here, with the quadrupoles as matched above, the beam at the start of the line that
ends in a waist with beta functions of 5 m:

```{code-cell} julia
ini = Context(beta1=10.0, alpha1=0.0, beta2=10.0, alpha2=0.0)
ti = TwissExpr(line; initial=ini)
res = match!(vary    = [@knob(ini.beta1), @knob(ini.alpha1), @knob(ini.beta2), @knob(ini.alpha2)],
             targets = [ti.beta1[end] == 5, ti.beta2[end] == 5,
                        ti.alpha1[end] == 0, ti.alpha2[end] == 0])
```

## Values and Derivatives

`value_and_jacobian` gives the values of expressions and their exact derivatives with respect to
knobs, for your own purposes:

```{code-cell} julia
v, J = value_and_jacobian([tw.beta1[qd], tw.q1], [@knob(qf.Kn1), @knob(qd.Kn1)])
J # [dbeta1/dqf dbeta1/dqd; dq1/dqf dq1/dqd]
```

`deriv(expr, knob)` is the derivative of an expression with respect to a knob as a new expression.
For example, the response of the closed orbit at two elements to a kick of a corrector, i.e. two
elements of the orbit response matrix:

```{code-cell} julia
kicker = HKicker(Kn0L=0.0)
ring = Beamline([qf, sf, d, b, d, qd, sd, d, b, d, kicker],
        species_ref=Species("electron"), pc_ref=18e9)
twr = TwissExpr(ring)
kick = Knob(kicker, :Kn0L)

R = [deriv(twr.x[qf], kick), deriv(twr.x[qd], kick)]
value(R)
```

Such derivatives can be used like any other expression, also in targets, for example to fit a
measured response matrix with quadrupole strengths.

## Using Other Optimizers

`match!` uses the Levenberg-Marquardt method of
[BatchSolve](https://mattsignorelli.github.io/BatchSolve.jl/dev/guide/), which finds the nearest
solution. Like every such method it can stop at a point that cannot be improved by small
changes although a solution exists elsewhere; a better starting point, bounds on the knobs, or a
global optimizer then help. For
anything else, `MatchProblem` provides what the many optimization packages of Julia ask for:

```{code-cell} julia
qf.Kn1 = 0.36
qd.Kn1 = -0.36
prob = MatchProblem(vary    = [@knob(qf.Kn1, lower=0.2, upper=0.6),
                               @knob(qd.Kn1, lower=-0.6, upper=-0.2)],
                    targets = [tw.q1 == 0.25, tw.q2 == 0.25])
```

- `prob.x`, `prob.lower`, `prob.upper`: the current values and the bounds of the knobs
- `value(prob, x)`: the vector of the distances from the targets, at the knob values `x`
- `value_and_jacobian(prob, x)`: these and their exact derivatives with respect to the knobs
- `penalty(prob, x)`: one number, the sum of the squared distances. It is `Inf` where the lattice
  has no solution, for example where it is unstable
- `setknobs!(prob, x)`: sets the knobs

For example, with `NewtonRaphson` of
[`NonlinearSolve`](https://docs.sciml.ai/NonlinearSolve/stable/):

```{code-cell} julia
using NonlinearSolve

f = NonlinearFunction((x, p) -> value(prob, x);
                      jac = (x, p) -> value_and_jacobian(prob, x)[2])
sol = solve(NonlinearProblem(f, prob.x), NewtonRaphson())
setknobs!(prob, sol.u)
twiss(fodo, at=[])
```

With the gradient-based `LBFGS` of [`Optim`](https://github.com/JuliaNLSolvers/Optim.jl):

```julia
import Optim

function g!(G, x) # gradient of the penalty
  r, J = value_and_jacobian(prob, x)
  G .= 2 .* (J'*r)
end
res = Optim.optimize(x -> penalty(prob, x), g!, prob.x, Optim.LBFGS())
setknobs!(prob, Optim.minimizer(res))
```

And with a genetic algorithm that needs no derivatives, here differential evolution of
[`Metaheuristics`](https://github.com/jmejia8/Metaheuristics.jl), within the bounds of the knobs:

```julia
import Metaheuristics

bounds = [prob.lower prob.upper]'
res = Metaheuristics.optimize(x -> penalty(prob, x), bounds, Metaheuristics.DE())
setknobs!(prob, Metaheuristics.minimizer(res))
```

Such a global search can be finished with `match!(prob)`, which then starts from its result.

## How It Works

The knobs are made parameters of the truncated power series that `twiss` computes with (see
[Parametric Normal Form](parametric-nf.md)), with the truncation orders that the quantities and
their derivatives need. One `twiss` calculation per `TwissExpr` then gives the values and all
derivatives, and the lattice is restored afterwards. While this runs, the lattice holds these
parameters, so `match!`, `value` and `value_and_jacobian` must not be used from several threads
at once on the same lattice.

A knob cannot be a parameter that is set to a deferred expression (vary the variable it depends
on instead), and deferred expressions and contexts that knobs act through must not be restricted
to a number type such as `DefExpr{Float64}` or `Context{Float64}`.

```{docstring} match!
```

```{docstring} TwissExpr
```

```{docstring} Knob
```

```{docstring} Target
```

```{docstring} value
```

```{docstring} value_and_jacobian
```

```{docstring} MatchProblem
```
