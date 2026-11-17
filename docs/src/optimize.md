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
takes the same arguments as [`twiss`](twiss.md), but nothing is computed yet.

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

Every summary value and every scalar column of [`twiss`](twiss.md) is available from a `TwissExpr`.
A column is indexed with a location:

| Expression | Meaning |
|---|---|
| `tw.q1`, `tw.q2`, `tw.q3`, `tw.etac`, `tw.alphac`, `tw.qspin` | Summary values |
| `tw.dq1`, `tw.dq2` | Chromaticities, short for `tw.q1[delta=1]` and `tw.q2[delta=1]` |
| `tw.q1[J1=1]` | A term of an amplitude-dependent value, as for `getterm` |
| `tw.beta1[qd]`, `tw.beta1[6]` | At the beginning of an element, or of the element with that beamline index |
| `tw.beta1[qd, :exit]` | At the end of the element |
| `tw.beta1[b, :]` | At every integration step of the element, both ends included |
| `tw.beta1[end]`, `tw.beta1[begin]` | At the end and the beginning of the beamline |
| `tw.beta1["q*"]` | At the elements found by name, see [Searching by Name](beamline.md) |
| `tw.beta1[(2.0, 10.0)]`, `tw.beta1[:]` | At every integration step in an s-range, or of the whole beamline |

Chromatic derivatives (`tw.dbeta1[qd]`, `tw.dx_2[1]`) and the `h` terms (`tw.h3000[1]`) work as
the columns of `twiss` do. The `h` terms are complex numbers, so use `abs`, `real` or `imag` of them.

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

`step` limits how much a knob may change in one iteration (`@knob qf.Kn1 step=0.01`). A knob is
also an expression, so it can appear in targets: `@knob(qf.Kn1) + @knob(qd.Kn1) <= 0.05`.

## Lines That Are Not Periodic

For a transfer line, a linac, or a section of a ring by itself, give the Twiss functions at its
start with `a_initial` (see [Open lattices](twiss.md)). This can be the `a` of another
`TwissExpr`, so that the line continues from another beamline. Here a line of four quadrupoles
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
