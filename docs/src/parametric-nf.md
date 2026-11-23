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

Everything that `twiss` computes can also be computed as a function of parameters of the
lattice: for example how the tunes depend on magnet strengths, or how the beta functions depend
on a length. These dependences are computed exactly, by automatic differentiation, and not by
finite differences.

There are two ways to do this. The simplest is to write the quantities and the parameters as
expressions, and let SciBmad compute their derivatives. The second is to make the parameters
Taylor series yourself, with the [GTPSA](https://github.com/bmad-sim/GTPSA.jl) package, and get
the complete dependence on them.

## Derivatives with respect to lattice parameters

As an example, take a FODO cell with sextupoles:

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

The quantities are written with `TwissExpr`. It takes the same arguments as `twiss`, and its
quantities are read like the summary values and columns of a `twiss` result (see
[Twiss](twiss.md)), but nothing is computed yet. The parameters are given as knobs, e.g.
`@knob(qf.Kn1)` for the strength of `qf`:

```{code-cell} julia
tw = TwissExpr(fodo)
kf = @knob(qf.Kn1)
kd = @knob(qd.Kn1)
```

`SciBmad.value_and_jacobian` computes the values of the quantities and their derivatives with
respect to the knobs. (It is not exported, because DifferentiationInterface exports a function of
the same name.) The Jacobian has one row per quantity and one column per knob:

```{code-cell} julia
v, J = SciBmad.value_and_jacobian([tw.q1, tw.q2], [kf, kd])
J # [dq1/dkf dq1/dkd; dq2/dkf dq2/dkd]
```

The same works for any column, at any location. For example, how the beta function and the
dispersion at `qd` depend on the quadrupole strengths:

```{code-cell} julia
v, J = SciBmad.value_and_jacobian([tw.beta1[qd], tw.dx[qd]], [kf, kd])
J # [dbeta1/dkf dbeta1/dkd; ddx/dkf ddx/dkd]
```

And how the chromaticities depend on the sextupole strengths:

```{code-cell} julia
v, J = SciBmad.value_and_jacobian([tw.dq1, tw.dq2], [@knob(sf.Kn2), @knob(sd.Kn2)])
J # [ddq1/dksf ddq1/dksd; ddq2/dksf ddq2/dksd]
```

No truncation order has to be chosen: each quantity knows the order it needs, here the second
order in δ for the chromaticities.

Any parameter that can be set with `object.parameter = value` can be a knob: an element's
strength, length or misalignment, or a variable of a [deferred expression](defexpr.md) (see
[Knobs](#optimize.knobs)).

`value` computes only the values. `deriv(quantity, knob)` is the derivative as a new quantity,
so that derivatives can be taken again. For example, the second derivative of the horizontal
tune with respect to the strength of `qf`:

```{code-cell} julia
value(deriv(deriv(tw.q1, kf), kf)) # -5.5676
```

Or how the first δ-derivative of the resonance driving term $h_{2000}$ at the beginning depends
on the strength of `sf` (the `h` terms are complex numbers):

```{code-cell} julia
value(deriv(tw.dh2000[1], @knob(sf.Kn2))) # 1.209 - 0.1048im
```

These expressions are also what is matched in [Matching and Optimization](optimize.md).

## Taylor series in the parameters with GTPSA

The derivatives above are computed by making the knobs parameters of the truncated power series
that `twiss` computes with, and restoring the lattice afterwards. This can also be done by hand,
to get the complete Taylor series in the parameters. It uses the truncated power series package
[GTPSA](https://github.com/bmad-sim/GTPSA.jl), whose
[quickstart guide](https://bmad-sim.github.io/GTPSA.jl/stable/quickstart/) explains the ideas.
The normal forms themselves are computed by the
[NonlinearNormalForm](https://github.com/bmad-sim/NonlinearNormalForm.jl) package.

First, a GTPSA `Descriptor` defines the variables, the parameters, and their truncation orders.
Here, it has the six phase space variables to order 1, two parameters (the changes of the two
quadrupole strengths) to order 1, and a maximum total order of 2:

```{code-cell} julia
dnf = Descriptor([1, 1, 1, 1, 1, 1], 2, [1, 1], 1)
```

The tunes and the beta functions come from the first-order terms of the map in the phase space
variables. Their first-order dependence on a parameter is therefore in terms of total order 2:
one variable times one parameter. Higher orders work too, but take longer to compute.

The parameters are then added to the strengths:

```{code-cell} julia
dk = params(dnf)
qf.Kn1 = 0.36 + dk[1]
qd.Kn1 = -0.36 + dk[2]
```

and `twiss` is given the Descriptor with `GTPSA_descriptor`:

```{code-cell} julia
twp = twiss(fodo, GTPSA_descriptor=dnf)
```

With parameters in the Descriptor, `twiss` returns its results as Taylor series
(`as_taylor_series=true`). The printed tunes now include their dependence on the strengths of
`qf` and `qd`. `grad` gives the first derivatives with respect to the parameters as a vector:

```{code-cell} julia
grad(twp.q1) # [dq1/dkf, dq1/dkd], as above
```

```{code-cell} julia
grad(twp.beta1[qd]) # [dbeta1/dkf, dbeta1/dkd]
```

Afterwards, the strengths of `qf` and `qd` are still Taylor series, even if numbers are set back
into them. Before using the beamline again, call `scalarize!`. It replaces every Taylor series
in the beamline by its value (its 0th order part):

```{code-cell} julia
scalarize!(fodo)
```

:::{warning}
Mixing Taylor series of two different Descriptors aborts the Julia process, with an error of
the GTPSA C library. For example, `twiss(fodo)` above, before `scalarize!`, would abort Julia,
because `twiss` uses its own Descriptor.
:::
