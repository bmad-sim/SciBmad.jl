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

(pnf)=
# Parametric Normal Form

```{code-cell} julia
:tags: [remove-cell]
using SciBmad
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

All analysis (e.g. computing amplitude-dependent tunes, nonlinear Twiss parameters,
invariant spin field, etc.) is done using SciBmad's
[`NonlinearNormalForm` package](https://github.com/bmad-sim/NonlinearNormalForm.jl).

SciBmad also provides parametric normal form calculations: how anything that `twiss` computes
depends on parameters of the lattice, for example how the tunes depend on magnet strengths or how
the beta functions depend on a length. These dependences are computed exactly, by automatic
differentiation, and not by finite differences.

## Derivatives with respect to lattice parameters

Let's compute how the tunes depend on the quadrupole strengths in our FODO cell example.

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-1.2, L=0.2)
end

fodo = Beamline([qf, sf, b, d, qd, sd, b, d],
        species_ref=Species("electron"), pc_ref=18e9)
```

The quantities are written with `TwissExpr`, which takes the same arguments as `twiss` and whose
quantities are read exactly like those of the result of `twiss` (see
[Reading the results](twiss.md)), but nothing is computed yet. The parameters are given as
knobs, `@knob(qf.Kn1)` for the strength of `qf`:

```{code-cell} julia
tw = TwissExpr(fodo)
kf = @knob(qf.Kn1)
kd = @knob(qd.Kn1)
```

`value_and_jacobian` gives the values of the quantities and their derivatives with respect to the
knobs:

```{code-cell} julia
v, J = value_and_jacobian([tw.q1, tw.q2], [kf, kd])
J # [dq1/dkf dq1/dkd; dq2/dkf dq2/dkd]
```

The same goes for any column of `twiss`, at any location. For example, how the beta function and
the dispersion at `qd` depend on the quadrupole strengths:

```{code-cell} julia
v, J = value_and_jacobian([tw.beta1[qd], tw.dx[qd]], [kf, kd])
J # [dbeta1/dkf dbeta1/dkd; ddx/dkf ddx/dkd]
```

and how the chromaticities depend on the sextupole strengths:

```{code-cell} julia
v, J = value_and_jacobian([tw.dq1, tw.dq2], [@knob(sf.Kn2), @knob(sd.Kn2)])
J # [ddq1/dksf ddq1/dksd; ddq2/dksf ddq2/dksd]
```

No truncation order has to be chosen: each quantity knows the order it needs, here the second
order in δ for the chromaticities. Any parameter that can be set with `object.parameter = value`
can be a knob: an element's strength, length or misalignment, or a variable of a
[deferred expression](defexpr.md) (see [Knobs](optimize.md)).

`deriv(quantity, knob)` is the derivative itself as a quantity, so that derivatives can be
taken again. The second derivative of the horizontal tune with respect to the strength of `qf`:

```{code-cell} julia
value(deriv(deriv(tw.q1, kf), kf)) # -5.5676
```

And how the first δ-derivative of the "resonance driving term" $h_{2000}$ at the beginning
depends on the strength of `sf` (the `h` terms are complex numbers):

```{code-cell} julia
value(deriv(tw.dh2000[1], @knob(sf.Kn2))) # 1.209 - 0.1048im
```

These are the same quantities that are used for matching, see
[Matching and Optimization](optimize.md).

## Taylor series in the parameters with GTPSA

The knobs above are made parameters of the truncated power series that `twiss` computes with,
and the lattice is restored afterwards. This can also be done by hand, to get the complete
dependence on the parameters as a Taylor series, using the underlying truncated power series
algebra package [`GTPSA`](https://github.com/bmad-sim/GTPSA.jl). We recommend first going
through the [GTPSA quickstart guide](https://bmad-sim.github.io/GTPSA.jl/stable/quickstart/)
to gain an understanding of the high order autodiff techniques, before going through this
section.

To compute how the tunes depend on the quadrupole strengths, we define a GTPSA `Descriptor`
with two parameters, corresponding to infinitesimal variations in each of the quadrupoles.
Because computing the tunes requires only 1st order in the phase space variables, we can save
computation time by explicitly specifying the truncation orders for the variables to be 1,
truncate the parameters part also at 1, but allow maximum order of 2:

```{code-cell} julia
dnf = Descriptor([1, 1, 1, 1, 1, 1], 2, [1, 1], 1)
dk = params(dnf)
qf.Kn1 = 0.36 + dk[1]
qd.Kn1 = -0.36 + dk[2]
```

Now we can just use `twiss` through the usual machinery, though this time provide the
descriptor explicitly to the `GTPSA_descriptor` keyword argument

```{code-cell} julia
twp = twiss(fodo, GTPSA_descriptor=dnf)
```

By explicitly providing a GTPSA descriptor including parameter dependence, the
`as_taylor_series` keyword argument to `twiss` is automatically set to true. Therefore, the
columns are now Taylor series, and the amplitude-dependent tunes are printed as Taylor series
*including the dependence on the strengths of `qf` and `qd`!*

To extract a gradient w.r.t. parameters as a vector, use the `grad` function:

```{code-cell} julia
grad(twp.q1) # [dq1/dkf, dq1/dkd], as above
```

```{code-cell} julia
grad(twp.beta1[qd]) # [dbeta1/dkf, dbeta1/dkd]
```

Before setting any parameters to be `TPS`'s from a different `Descriptor`, use `scalarize!` on
the beamline, which changes all stored `TPS` values in a `Beamline` to be the "scalar" (0th
order) part:

```{code-cell} julia
scalarize!(fodo)
```

:::{warning}
Mixing two `TPS` objects with different `Descriptor`s will cause the program to crash, by an
error baked in to the GTPSA C library. Please take care to not do this.
:::

`value_and_jacobian`, `deriv` and `match!` do all of this themselves, and restore the lattice
afterwards.
