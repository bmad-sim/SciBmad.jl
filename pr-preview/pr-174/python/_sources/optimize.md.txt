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

```{code-cell} python
:tags: [remove-cell]
import numpy as np
import scibmad as sb
sb.options.display_width = 100
sb.options.display_height = 30
```

Matching means varying parameters of a lattice (the "knobs") until quantities such as the tunes,
the beta functions at some point, chromaticities, etc have the values you want. In SciBmad all you
need to do is write down the quantities and their goals, say what knobs should be varied to achieve
that goal, and call {jl}`match!`{py}`sb.match`.

Most optimizers need the derivatives of the quantities with respect to the knobs. In
non-differentiable codes, these are computed with finite differences, which are approximate and
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

```{code-cell} python
import numpy as np
import scibmad as sb

with sb.elements():
    qf = sb.Quadrupole(Kn1=0.36, L=0.5)
    sf = sb.Sextupole(Kn2=1.2, L=0.2)
    d = sb.Drift(L=0.2)
    b = sb.SBend(L=6.5, angle=np.pi/132)
    qd = sb.Quadrupole(Kn1=-0.36, L=0.5)
    sd = sb.Sextupole(Kn2=-1.2, L=0.2)

fodo = sb.Beamline([qf, sf, d, b, d, qd, sd, d, b, d],
                   species_ref=sb.Species("electron"), pc_ref=18e9)
```

First we need a way to refer to what `twiss` computes for `fodo`: `TwissExpr` gives exactly that. It is a lazily-evaluated `twiss` call that can be read in the same way that `twiss` can, however nothing is actually computed until it is requested. Furthermore, it will automatically infer the  `at`, `cols`, `base_cols`, `as_taylor_series` and `GTPSA_descriptor` keyword arguments to `twiss` based on what you request. Other than those inputs, `TwissExpr` accepts all the same arguments as [`twiss`](twiss.md).

```{code-cell} julia
tw = TwissExpr(fodo)
```

```{code-cell} python
tw = sb.TwissExpr(fodo)
tw
```

The tunes of the cell are then `tw.q1` and `tw.q2`. A target is written with `==`, and a knob with
{jl}`@knob`{py}`sb.Knob(object, "parameter")`:

```{code-cell} julia
res = match!(vary    = [@knob(qf.Kn1), @knob(qd.Kn1)],
             targets = [tw.q1 == 0.25, tw.q2 == 0.25])
```

```{code-cell} python
res = sb.match(vary=[sb.Knob(qf, "Kn1"), sb.Knob(qd, "Kn1")],
               targets=[tw.q1 == 0.25, tw.q2 == 0.25])
res
```

The quadrupole strengths have been changed, and `twiss` confirms the tunes:

```{code-cell} julia
twiss(fodo, at=[])
```

```{code-cell} python
sb.twiss(fodo, at=[])
```

:::{only} julia
`undo!(res)` would set the knobs back to where they were, and `match!(res)` continues a match
that stopped early.

If only one beamline is involved, it can also be given to `match!` directly. The targets are then
written as a function of its `TwissExpr`. Keyword arguments for `twiss`, if any, are given with
`twiss=(rf_on=false,)`:
:::
:::{only} python
`sb.undo(res)` would set the knobs back to where they were, and `sb.match(res)` continues a match
that stopped early.

If only one beamline is involved, it can also be given to `sb.match` directly. The targets are
then written as a function of its `TwissExpr`. Keyword arguments for `twiss`, if any, are given
with `twiss=dict(rf_on=False)`:
:::

```julia
match!(fodo; vary = [@knob(qf.Kn1), @knob(qd.Kn1)],
             targets = tw -> [tw.q1 == 0.25, tw.q2 == 0.25])
```

```python
sb.match(fodo, vary=[sb.Knob(qf, "Kn1"), sb.Knob(qd, "Kn1")],
         targets=lambda tw: [tw.q1 == 0.25, tw.q2 == 0.25])
```

Now let's make the chromaticities $+1$ in both planes with the sextupoles. The chromaticities are
`tw.dq1` and `tw.dq2`. They are only defined without synchrotron motion, so for a beamline with RF
cavities use `TwissExpr(ring; rf_on=false)`.

```{code-cell} julia
res = match!(vary    = [@knob(sf.Kn2), @knob(sd.Kn2)],
             targets = [tw.dq1 == 1, tw.dq2 == 1])
```

```{code-cell} python
res = sb.match(vary=[sb.Knob(sf, "Kn2"), sb.Knob(sd, "Kn2")],
               targets=[tw.dq1 == 1, tw.dq2 == 1])
res
```

## Quantities and Locations

Every summary value and every numeric column of [`twiss`](twiss.md) is available from a
`TwissExpr`, written in the same way as for the result of `twiss`: `tw.q1`, `tw.dq1`,
{jl}`tw.q1[J1=1]`{py}`tw.q1(J1=1)`, `tw.beta1[qd]`,
{jl}`tw.beta1[qd, :exit]`{py}`tw.beta1[qd, "exit"]`, `tw.beta1[[b], :]`, `tw.beta1["q*"]`,
`tw.dbeta1[qd]`, `tw.h3000[1]`, and so on. Unlike for `twiss`, you do not choose truncation
orders or rows: `TwissExpr` will knows what you need based on what you ask for.

You can form arbitrary expressions with the Twiss output to serve as targets/minimization goals using ordinary arithmetic and functions. For example, the square of the beta function at `qd` minus five times the average of the alpha function through the bends:

```{code-cell} julia
e = tw.beta1[qd]^2 - 5*mean(tw.alpha1[[b], :])
```

```{code-cell} python
e = tw.beta1[qd]**2 - 5*sb.mean(tw.alpha1[[b], :])
e
```

The `h` terms are complex numbers. `value` gives them as such, but a target or `minimize` needs
their `real` and `imag` parts separately.

`value` computes an expression. Only the rows and columns of `twiss` that it needs are computed:

```{code-cell} julia
value(e)
```

```{code-cell} python
sb.value(e)
```

A location with several places gives a vector, and arithmetic and comparisons apply to each of
its entries. {jl}`sum`{py}`sb.sum`, {jl}`mean`{py}`sb.mean`, {jl}`maximum`{py}`sb.maximum`, and
{jl}`minimum`{py}`sb.minimum` reduce it to one
expression{pytext}` (they are Julia's functions, which build an expression, not Python's)`:

```{code-cell} julia
value(tw.beta1["q*"]) # beta1 at qf and at qd
```

```{code-cell} python
sb.value(tw.beta1["q*"])  # beta1 at qf and at qd
```

```{code-cell} julia
value(maximum(tw.beta1[:]))
```

```{code-cell} python
sb.value(sb.maximum(tw.beta1[:]))
```

A phase advance between two points is simply a difference:

```{code-cell} julia
value(tw.phi1[qd] - tw.phi1[qf]) # in units of 2π
```

```{code-cell} python
sb.value(tw.phi1[qd] - tw.phi1[qf])  # in units of 2π
```

Anything else can be included with `lazy`: {jl}`lazy(f)`{py}`sb.lazy(f)` is the result of the
function `f()`, computed
from the lattice itself when it is needed. For example, the field gradient of `qf` in this
beamline:

```{code-cell} julia
value(lazy(() -> fodo[1].Bn1))
```

```{code-cell} python
sb.value(sb.lazy(lambda: fodo[1].Bn1))
```

## Targets

Besides `==`, a target can be an inequality:

```julia
tw.beta1[qd] <= 30
tw.beta1["q*"] <= 30            # at each of these elements
maximum(tw.beta1[:]) <= 100     # everywhere
```

```python
tw.beta1[qd] <= 30
tw.beta1["q*"] <= 30              # at each of these elements
sb.maximum(tw.beta1[:]) <= 100    # everywhere
```

An inequality only acts while it is violated. A target counts as met when it is within its
tolerance, which is `1e-8` unless you set it with `Target`. A `weight` makes a target count more
or less than the others while they cannot all be met:

```julia
Target(tw.q1, 0.25; tol=1e-6)
Target(tw.beta1[qd]; lower=5, upper=30, weight=0.1)
```

```python
sb.Target(tw.q1, 0.25, tol=1e-6)
sb.Target(tw.beta1[qd], lower=5, upper=30, weight=0.1)
```

There may be fewer, as many, or more targets than knobs. With more knobs than needed, many solutions
exist, and `match!` returns one near the starting point: each step is the smallest change of the
knobs that meets the targets to first order. With more targets than can be met, `match!` returns the
closest it can get (in the least-squares sense) and reports the match as not converged.

:::{only} julia
`minimize` is for quantities that should just be as small as possible. For example, we can minimize the driving terms of a ring with four sextupoles while also requiring that the chromaticities are +1 with (using `reim` which returns a tuple of the real and imaginary parts, and then splatting those values with `...` into the vector):
:::
:::{only} python
`minimize` is for quantities that should just be as small as possible. For example, we can minimize
the driving terms of a ring with four sextupoles while also requiring that the chromaticities are +1
with (using `sb.reim` which returns a tuple of the real and imaginary parts, and then unpacking
those values with `*` into the list):
:::

```julia
twc = TwissExpr(ring; rf_on=false)
match!(vary = [@knob(s1.Kn2), @knob(s2.Kn2), @knob(s3.Kn2), @knob(s4.Kn2)],
       targets = [Target(twc.dq1, 1; weight=100, tol=1e-4),
                  Target(twc.dq2, 1; weight=100, tol=1e-4)],
       minimize = [reim(twc.h3000[1])..., reim(twc.h2100[1])..., reim(twc.h1020[1])...])
```

```python
twc = sb.TwissExpr(ring, rf_on=False)
sb.match(vary=[sb.Knob(s, "Kn2") for s in (s1, s2, s3, s4)],
         targets=[sb.Target(twc.dq1, 1, weight=100, tol=1e-4),
                  sb.Target(twc.dq2, 1, weight=100, tol=1e-4)],
         minimize=[*sb.reim(twc.h3000[1]), *sb.reim(twc.h2100[1]), *sb.reim(twc.h1020[1])])
```

The sum of the squares of the expressions in `minimize` and of the distances from the targets
(each times its weight) is made as small as possible. If they cannot all vanish, the result
balances them, so give targets that must hold a larger weight, as here.

(optimize.knobs)=
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

```python
sb.Knob(qf, "Kn1")                          # an element parameter
sb.Knob(qf, "Kn1", lower=0, upper=0.6)      # with bounds
sb.Knob(d, "L")                             # also lengths
sb.Knob(ring.context, "kqf")                # a variable used by deferred expressions
sb.Knob(ring.context, "coilstrengths")      # a vector variable: one knob per entry
sb.Knob(ring.context, "coilstrengths", 2)   # one entry of it (the 2nd, counted from 1)
```

:::{only} julia
`@knob qf.Kn1` is the same as `Knob(qf, :Kn1)`. Setting a parameter of an element changes it
everywhere the element is placed (see [Defining a Beamline](beamline.md)), so `@knob(qf.Kn1)`
varies all `qf`s together.
:::
:::{only} python
`sb.Knob(qf, "Kn1")` is Julia's `@knob qf.Kn1` (Python cannot capture `qf.Kn1` itself: it would
be the number). Setting a parameter of an element changes it everywhere the element is placed
(see [Defining a Beamline](beamline.md)), so `sb.Knob(qf, "Kn1")` varies all `qf`s together.
:::

To vary several different elements together, or to build any other
knob, use a [deferred expression](defexpr.md) and vary the variable it depends on:

```{code-cell} julia
:tags: [remove-cell]
kqf, kqd = qf.Kn1, qd.Kn1
```

```{code-cell} python
:tags: [remove-cell]
kqf, kqd = qf.Kn1, qd.Kn1
```

```{code-cell} julia
qf.Kn1 = DefExpr(c -> c.kq)
qd.Kn1 = DefExpr(c -> -c.kq)
fodo.context.kq = 0.38

res = match!(vary = [@knob(fodo.context.kq)], targets = [tw.q1 == 0.25])
```

```{code-cell} python
qf.Kn1 = sb.DefExpr(lambda c: c.kq)
qd.Kn1 = sb.DefExpr(lambda c: -c.kq)
fodo.context.kq = 0.38

res = sb.match(vary=[sb.Knob(fodo.context, "kq")], targets=[tw.q1 == 0.25])
res
```

```{code-cell} julia
:tags: [remove-cell]
qf.Kn1 = kqf
qd.Kn1 = kqd
```

```{code-cell} python
:tags: [remove-cell]
qf.Kn1 = kqf
qd.Kn1 = kqd
```

:::{only} julia
A variable can also be a vector, e.g. the strengths of many correctors, each set by a deferred
expression such as `DefExpr(c -> c.coilstrengths[3])`. `@knob(ring.context.coilstrengths)` then
varies every entry. The result lists them as `coilstrengths[1]`, `coilstrengths[2]`, and so on,
as many as fit in the terminal. `print(res)` shows all of them.

`step` limits how much a knob may change in one iteration (`@knob qf.Kn1 step=0.01`). A knob is
also an expression, so it can appear in targets: `@knob(qf.Kn1) + @knob(qd.Kn1) <= 0.05`.
:::
:::{only} python
A variable can also be a vector, e.g. the strengths of many correctors, each set by a deferred
expression such as `sb.DefExpr(lambda c: c.coilstrengths[2])` (the third entry: entries of a
vector variable are counted from 0 in Python, as in the NumPy array `ring.context.coilstrengths`).
`sb.Knob(ring.context, "coilstrengths")` then varies every entry. The result lists them as
`coilstrengths[1]`, `coilstrengths[2]`, and so on (the names count from 1, as Julia does, and so
does the third argument of `sb.Knob`, the entry), as many as fit in the terminal.
`print(res)` shows all of them.

`step` limits how much a knob may change in one iteration (`sb.Knob(qf, "Kn1", step=0.01)`). A
knob is also an expression, so it can appear in targets:
`sb.Knob(qf, "Kn1") + sb.Knob(qd, "Kn1") <= 0.05`.
:::

## Orbit Correction

The closed orbit is a column of `twiss` like any other (`tw.x[bpm]`, `tw.y[bpm]`, ...), so
{jl}`match!`{py}`sb.match` can do orbit correction too. Here is a ring of ten cells, each with a
misaligned quadrupole, a
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

```{code-cell} python
ncell = 10
offsets = 1e-4 * np.array([0.8, -1.2, 0.3, 1.5, -0.6, 0.9, -1.1, 0.2, 1.3, -0.4])  # in m
elements = []
bpms = []
for i in range(ncell):
    bpm = sb.Marker()
    bpms.append(bpm)
    elements += [sb.Quadrupole(Kn1=sb.DefExpr(lambda c: 0.36 + c.dk[i]), L=0.5,
                               x_offset=offsets[i]),
                 sb.Multipole(Kn0L=sb.DefExpr(lambda c: c.kicks[i])), sb.Drift(L=0.2), bpm,
                 sb.SBend(L=6.5, angle=np.pi/ncell), sb.Drift(L=0.2),
                 sb.Quadrupole(Kn1=-0.36, L=0.5), sb.Drift(L=0.2),
                 sb.SBend(L=6.5, angle=np.pi/ncell), sb.Drift(L=0.2)]
ring = sb.Beamline(elements, species_ref=sb.Species("electron"), pc_ref=18e9)
ring.context.kicks = np.zeros(ncell)
ring.context.dk = np.zeros(ncell)

twr = sb.TwissExpr(ring)
orbit = [twr.x[bpm] for bpm in bpms]
np.max(np.abs(sb.value(orbit)))  # 5.8e-4 m at the BPMs
```

`minimize` makes the orbit at the BPMs as small as possible (in the least-squares sense), varying
every corrector:

```{code-cell} julia
res = match!(vary = [@knob(ring.context.kicks)], minimize = orbit)
maximum(abs, value(orbit)) # 3e-15 m
```

```{code-cell} python
res = sb.match(vary=[sb.Knob(ring.context, "kicks")], minimize=orbit)
np.max(np.abs(sb.value(orbit)))  # 3e-15 m
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

```{code-cell} python
R = np.array([[sb.deriv(twr.x[bpm], sb.Knob(ring.context, "kicks", j))
               for j in range(1, ncell + 1)] for bpm in bpms], dtype=object)
```

A measured response matrix can then be fitted, as in LOCO. It depends on the focusing of the
ring, and hardly on the misalignments: they shift the orbit, but without sextupoles they barely
change its response to a kick (here by 2e-4 of its size). So it is fitted with gradient errors.

Here the "measurement" is the response matrix of the ring with the gradient errors `dk_true`. The
model starts without them, and finds them again:

```{code-cell} julia
dk_true = 1e-3 .* [1.0, -2.0, 0.5, 1.5, -1.0, 0.8, -0.3, 2.0, -1.2, 0.4]
ring.context.dk = copy(dk_true)
R_measured = value(R)
ring.context.dk = zeros(ncell)

res = match!(vary = [@knob(ring.context.dk)], targets = vec(R .== R_measured))
maximum(abs, ring.context.dk - dk_true) # 5e-16: the errors are found again
```

```{code-cell} python
dk_true = 1e-3 * np.array([1.0, -2.0, 0.5, 1.5, -1.0, 0.8, -0.3, 2.0, -1.2, 0.4])
ring.context.dk = dk_true.copy()
R_measured = sb.value(R)
ring.context.dk = np.zeros(ncell)

targets = [r == m for r, m in zip(R.ravel(order="F"), R_measured.ravel(order="F"))]
res = sb.match(vary=[sb.Knob(ring.context, "dk")], targets=targets)
np.max(np.abs(ring.context.dk - dk_true))  # 5e-16: the errors are found again
```

Further knobs fit the same way, for example quadrupole tilts for the coupling (with the vertical
orbit in the matrix) and the calibrations of the BPMs and correctors.

## Lines That Are Not Periodic

For a transfer line, a linac, or a section of a ring by itself, give the lattice functions at its
start with `initial` or `a_initial` (see [Open lines](#twiss.open)). `a_initial` can be the `a` of
another `TwissExpr`, so that the line continues from another beamline. Here a line of four
quadrupoles follows the FODO cell, and is matched to a waist with beta functions of 10 m at its
end:

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

tl = TwissExpr(line; a_initial=tw.a[end]) # starts with the periodic lattice functions of the cell

res = match!(vary    = [@knob(m1.Kn1), @knob(m2.Kn1), @knob(m3.Kn1), @knob(m4.Kn1)],
             targets = [tl.beta1[end] == 10, tl.beta2[end] == 10,
                        tl.alpha1[end] == 0, tl.alpha2[end] == 0])
```

```{code-cell} python
with sb.elements():
    m1 = sb.Quadrupole(Kn1=0.3, L=0.5)
    m2 = sb.Quadrupole(Kn1=-0.3, L=0.5)
    m3 = sb.Quadrupole(Kn1=0.3, L=0.5)
    m4 = sb.Quadrupole(Kn1=-0.3, L=0.5)
    dm = sb.Drift(L=3.0)

line = sb.Beamline([m1, dm, m2, dm, m3, dm, m4, dm],
                   species_ref=sb.Species("electron"), pc_ref=18e9)

# starts with the periodic lattice functions of the cell:
tl = sb.TwissExpr(line, a_initial=tw.a[-1])

res = sb.match(vary=[sb.Knob(m, "Kn1") for m in (m1, m2, m3, m4)],
               targets=[tl.beta1[-1] == 10, tl.beta2[-1] == 10,
                        tl.alpha1[-1] == 0, tl.alpha2[-1] == 0])
res
```

Because the start of the line is taken from the cell each time, quadrupoles of the cell could be
varied in the same match. A whole ring can be built this way piece by piece: match the arc cell,
then a dispersion suppressor that follows it
({jl}`tl.dx[end] == 0, tl.dpx[end] == 0`{py}`tl.dx[-1] == 0, tl.dpx[-1] == 0`), then the
straight section that follows the suppressor, and finally the tunes and chromaticities of the
complete ring.

The Twiss parameters at the start can be knobs too: give `initial` as a `Context`, and vary its
variables. Here, with the quadrupoles as matched above, they are matched so that the line ends in
a waist with beta functions of 5 m:

```{code-cell} julia
ini = Context(beta1=10.0, alpha1=0.0, beta2=10.0, alpha2=0.0)
ti = TwissExpr(line; initial=ini)
res = match!(vary    = [@knob(ini.beta1), @knob(ini.alpha1),
                          @knob(ini.beta2), @knob(ini.alpha2)],
             targets = [ti.beta1[end] == 5, ti.beta2[end] == 5,
                        ti.alpha1[end] == 0, ti.alpha2[end] == 0])
```

```{code-cell} python
ini = sb.Context(beta1=10.0, alpha1=0.0, beta2=10.0, alpha2=0.0)
ti = sb.TwissExpr(line, initial=ini)
res = sb.match(vary=[sb.Knob(ini, p) for p in ("beta1", "alpha1", "beta2", "alpha2")],
               targets=[ti.beta1[-1] == 5, ti.beta2[-1] == 5,
                        ti.alpha1[-1] == 0, ti.alpha2[-1] == 0])
res
```

## Values and Derivatives

:::{only} julia
`SciBmad.value_and_jacobian` gives the values of expressions and their exact derivatives with
respect to knobs, for your own purposes. It is not exported, because DifferentiationInterface
exports a function of the same name:
:::
:::{only} python
`sb.value_and_jacobian` gives the values of expressions and their exact derivatives with
respect to knobs, for your own purposes (in Julia it is `SciBmad.value_and_jacobian`, which is
not exported):
:::

```{code-cell} julia
v, J = SciBmad.value_and_jacobian([tw.beta1[qd], tw.q1], [@knob(qf.Kn1), @knob(qd.Kn1)])
J # [dbeta1/dqf dbeta1/dqd; dq1/dqf dq1/dqd]
```

```{code-cell} python
v, J = sb.value_and_jacobian([tw.beta1[qd], tw.q1], [sb.Knob(qf, "Kn1"), sb.Knob(qd, "Kn1")])
J  # [[dbeta1/dqf, dbeta1/dqd], [dq1/dqf, dq1/dqd]]
```

`deriv(expr, knob)` is the derivative of an expression with respect to a knob as a new
expression, as for the orbit response matrix above. Such derivatives can be used like any other
expression, also in targets. For more, see [Parametric Normal Form](#pnf).

## Using Other Optimizers

{jl}`match!`{py}`sb.match` takes Gauss-Newton steps, which find a nearby solution. With
{jl}`solver=:levenberg_marquardt`{py}`solver="levenberg_marquardt"`
it uses the Levenberg-Marquardt method of
[BatchSolve](https://mattsignorelli.github.io/BatchSolve.jl/dev/guide/) instead. Like every such
method, it can stop at a point that small changes cannot improve, even though a solution exists
elsewhere. A better starting point, bounds on the knobs, or a global optimizer then help.

For other optimizers, `MatchProblem` provides what the optimization packages of
{jltext}`Julia`{pytext}`Python (e.g. SciPy)` ask for:

```{code-cell} julia
qf.Kn1 = 0.36
qd.Kn1 = -0.36
prob = MatchProblem(vary    = [@knob(qf.Kn1, lower=0.2, upper=0.6),
                               @knob(qd.Kn1, lower=-0.6, upper=-0.2)],
                    targets = [tw.q1 == 0.25, tw.q2 == 0.25])
```

```{code-cell} python
qf.Kn1 = 0.36
qd.Kn1 = -0.36
prob = sb.MatchProblem(vary=[sb.Knob(qf, "Kn1", lower=0.2, upper=0.6),
                             sb.Knob(qd, "Kn1", lower=-0.6, upper=-0.2)],
                       targets=[tw.q1 == 0.25, tw.q2 == 0.25])
prob
```

- `prob.x`, `prob.lower`, `prob.upper`: the current values and the bounds of the knobs
- {jl}`value(prob, x)`{py}`sb.value(prob, x)`: the vector of the distances from the targets, at the
  knob values `x`
- {jl}`SciBmad.value_and_jacobian(prob, x)`{py}`sb.value_and_jacobian(prob, x)`: these and their
  exact derivatives with respect to the knobs
- {jl}`penalty(prob, x)`{py}`sb.penalty(prob, x)`: one number, the sum of the squared distances. It
  is {jl}`Inf`{py}`inf` where the lattice
  has no solution, for example where it is unstable
- {jl}`setknobs!(prob, x)`{py}`sb.setknobs(prob, x)`: sets the knobs

`value`, `value_and_jacobian` and `penalty` set the knobs to `x`, so the lattice is left at the
last `x` an optimizer tried. Finish with {jl}`setknobs!`{py}`sb.setknobs` at the solution, as below.

:::{only} julia
For example, with `NewtonRaphson` of
[`NonlinearSolve`](https://docs.sciml.ai/NonlinearSolve/stable/):
:::
:::{only} python
For example, with the least-squares solver `least_squares` of
[SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html):
:::

```{code-cell} julia
using NonlinearSolve

f = NonlinearFunction((x, p) -> value(prob, x);
                      jac = (x, p) -> SciBmad.value_and_jacobian(prob, x)[2])
sol = solve(NonlinearProblem(f, prob.x), NewtonRaphson())
setknobs!(prob, sol.u)
twiss(fodo, at=[])
```

```{code-cell} python
from scipy.optimize import least_squares

sol = least_squares(lambda x: sb.value(prob, x), prob.x,
                    jac=lambda x: sb.value_and_jacobian(prob, x)[1])
sb.setknobs(prob, sol.x)
sb.twiss(fodo, at=[])
```

:::{only} julia
With the gradient-based `LBFGS` of [`Optim`](https://github.com/JuliaNLSolvers/Optim.jl):
:::
:::{only} python
With the gradient-based `L-BFGS-B` of [SciPy's
`minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html):
:::

```julia
import Optim

function g!(G, x) # gradient of the penalty
  r, J = SciBmad.value_and_jacobian(prob, x)
  G .= 2 .* (J'*r)
end
res = Optim.optimize(x -> penalty(prob, x), g!, prob.x, Optim.LBFGS())
setknobs!(prob, Optim.minimizer(res))
```

```python
from scipy.optimize import minimize

def gradient(x):  # gradient of the penalty
    r, J = sb.value_and_jacobian(prob, x)
    return 2 * J.T @ r

res = minimize(lambda x: sb.penalty(prob, x), prob.x, jac=gradient, method="L-BFGS-B")
sb.setknobs(prob, res.x)
```

:::{only} julia
And with a genetic algorithm that needs no derivatives, here differential evolution of
[`Metaheuristics`](https://github.com/jmejia8/Metaheuristics.jl), within the bounds of the knobs:
:::
:::{only} python
And with a genetic algorithm that needs no derivatives, here
[SciPy's differential
evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html),
within the bounds of the knobs:
:::

```julia
import Metaheuristics

bounds = [prob.lower prob.upper]'
res = Metaheuristics.optimize(x -> penalty(prob, x), bounds, Metaheuristics.DE())
setknobs!(prob, Metaheuristics.minimizer(res))
```

```python
from scipy.optimize import differential_evolution

bounds = list(zip(prob.lower, prob.upper))
res = differential_evolution(lambda x: sb.penalty(prob, x), bounds)
sb.setknobs(prob, res.x)
```

Such a global search can be finished with {jl}`match!(prob)`{py}`sb.match(prob)`, which then starts
from its result.

## How It Works

The knobs are made parameters of the truncated power series that `twiss` computes with (see
[Parametric Normal Form](parametric-nf.md)), with the truncation orders that the quantities and
their derivatives need. One `twiss` calculation per `TwissExpr` then gives the values and all
derivatives, and the lattice is restored afterwards. While this runs, the lattice holds these
parameters, so {jl}`match!`{py}`match`, `value` and `value_and_jacobian` must not be used from
several threads
at once on the same lattice.

Making a parameter a knob sets it, as `qf.Kn1 = 0.38` does. If the parameter was set to a deferred
expression, that dependence is then broken, also by `value` and `value_and_jacobian`. To keep it,
vary the variable the expression depends on. Deferred expressions and contexts that knobs act
through must not be restricted to a number type such as `DefExpr{Float64}` or
`Context{Float64}`.

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

```{docstring} SciBmad.value_and_jacobian
```

```{docstring} MatchProblem
```
