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

SciBmad also provides parametric normal form calculations, for example computing how the
tunes depend on magnet strengths or how the invariant spin field depends on a misalignment.
To do so, knowledge of the underlying truncated power series algebra package
[`GTPSA`](https://github.com/bmad-sim/GTPSA.jl) is highly useful. We recommend first going
through the [GTPSA quickstart guide](https://bmad-sim.github.io/GTPSA.jl/stable/quickstart/)
to gain an understanding of the high order autodiff techniques, before going through this
section.

Let's compute how the tunes depend on the quadrupole strengths in our FODO cell example.

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=DefExpr(c -> c.kqf), L=0.5)
  sf = Sextupole(Kn2=DefExpr(c-> c.ksf), L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=DefExpr(c -> c.kqd), L=0.5)
  sd = Sextupole(Kn2=DefExpr(c -> c.ksd), L=0.2)
end

fodo = Beamline([qf, sf, b, d, qd, sd, b, d],
        species_ref=Species("electron"), pc_ref=18e9)

fodo.context.kqf = 0.36
fodo.context.kqd = -0.36
fodo.context.ksf = 1.2
fodo.context.ksd = -1.2
```

To do so, we will define a GTPSA `Descriptor` with two parameters, corresponding to
infinitesimal variations in each of the quadrupoles. Because computing the tunes requires
only 1st order in the phase space variables, we can save computation time by explicitly
specifying the truncation orders for the variables to be 1, truncate the parameters part
also at 1, but allow maximum order of 2:

```{code-cell} julia
dnf = Descriptor([1, 1, 1, 1, 1, 1], 2, [1, 1], 1)
dk = params(dnf)
fodo.context.kqf += dk[1]
fodo.context.kqd += dk[2]
```

Now we can just use `twiss` through the usual machinery, though this time provide the
descriptor explicitly to the `GTPSA_descriptor` keyword argument

```{code-cell} julia
tw = twiss(fodo, GTPSA_descriptor=dnf)
```

By explicitly providing a GTPSA descriptor including parameter dependence, the
`as_taylor_series` keyword argument to `twiss` is automatically set to true. Therefore, the
columns are now Taylor series.

We also see the amplitude-dependent tunes are printed as Taylor series *including the
dependence on `kqf` and `kqd`!*

To extract a gradient w.r.t. parameters as a vector, use the `grad` function:

```{code-cell} julia
grad(tw.q1) # [dq1/dkqf, dq1/dkqd]
```

```{code-cell} julia
grad(tw.q2) # [dq2/dkqf, dq2/dkqd]
```

These can then be used in optimizations, for example to optimize the tunes — see
[Optimization with Autodiff](optimize.md).

The same tools can be used on the lattice functions. For example, to see how the periodic
beta functions at the beginning depend on the quadrupole strengths:

```{code-cell} julia
grad(tw.beta1[1]) # [dbeta1/dkf, dbeta1/dkd]
```

```{code-cell} julia
grad(tw.beta2[1]) # [dbeta2/dkf, dbeta2/dkd]
```

This goes for any column we output with Twiss.

We can also see how the chromaticities vary with sextupole strength. To compute the
chromaticity, we require 2nd order in δ. And since we need to see how this 2nd-order in δ
quantity varies with parameters, the maximum truncation order must be increased to 3.

Before setting any parameters to be `TPS`'s from a different `Descriptor`, it is good to use
`scalarize!` on the beamline, which changes all stored `TPS` values in a `Beamline` to be the
"scalar" (0th order) part:

```{code-cell} julia
scalarize!(fodo)
```

:::{warning}
Mixing two `TPS` objects with different `Descriptor`s will cause the program to crash, by an
error baked in to the GTPSA C library. Please take care to not do this.
:::

We then construct our new GTPSA descriptor and set the sextupole strengths as TPSA
parameters:

```{code-cell} julia
dnf2 = Descriptor([1, 1, 1, 1, 1, 2], 3, [1, 1], 1)
dk = params(dnf2)
fodo.context.ksf += dk[1]
fodo.context.ksd += dk[2]

tw = twiss(fodo, GTPSA_descriptor=dnf2)
```

We can extract the horizontal and vertical chromaticities as Taylor series using `getterm`:

```{code-cell} julia
chromx = getterm(tw.q1, delta=1, as_taylor_series=true)
chromy = getterm(tw.q2, delta=1, as_taylor_series=true)
```

And once again use `grad` to compute the gradient w.r.t. parameters

```{code-cell} julia
grad(chromx) # [dchromx/dksf, dchromx/dksd]
```

```{code-cell} julia
grad(chromy) # [dchromy/dksf, dchromy/dksd]
```

As a last example, suppose we want to see how the first δ-derivative of the "resonance
driving term" $h_{2000}$ at the beginning depends on the sextupole strength. This column in
`twiss` can be requested with `"dh2000"` (or equivalently `"dh2000_1"`). We first will
re-`scalarize!` and then construct the appropriate GTPSA `Descriptor`: $h_{2000}$ requires
only first order in the phase space coordinates, but a chromatic derivative adds one order
to δ. However, to accurately compute any resonance driving term/detune coefficient, all
phase space variables must have the same truncation order (thus 2 for all). Then, to see how
this quantity depends on a parameter, we have to bump up the maximum order to 3.

```{code-cell} julia
scalarize!(fodo)

dnf3 = Descriptor([2, 2, 2, 2, 2, 2], 3, [1], 1)
dk = params(dnf3)
fodo.context.ksf += dk[1]

tw = twiss(fodo, GTPSA_descriptor=dnf3, cols=["dh2000"])
```

Finally, we can just use `grad` again to extract
$\partial^2 h_{2000}/\partial\delta\partial k_{sf}$ at the beginning of the beamline:

```{code-cell} julia
grad(tw.dh2000[1])
```
