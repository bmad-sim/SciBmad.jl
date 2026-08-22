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

(defining.lineelement)=
# Defining a LineElement

```{code-cell} julia
:tags: [remove-cell]
using SciBmad
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

To construct a `LineElement`,

```{code-cell} julia
ele = LineElement()
```

As you can see, the `LineElement` by default comes with a "parameter group" structure
called `UniversalParams`. This contains a `kind` as a string, a `name` as a string, a
length `L`, and a `tracking_method`, which defaults to `SciBmadStandard`. For more details
on the tracking methods available, see the [Tracking Methods](tracking-methods.md) section
of the documentation.

To set one of these parameters, use the natural syntax

```{code-cell} julia
ele.name = "ele"
ele.kind = "Quadrupole"
ele.L = 123
```

Alternatively, we could have set these parameters as keyword arguments ("`kwargs`") during
`LineElement` construction:

```{code-cell} julia
ele = LineElement(name="ele", kind="Quadrupole", L=123)
```

It would often be convenient if we can make the variable symbol (`ele` in this case)
automatically fill in the `name` field for each element. We can do exactly this by wrapping
all element definitions in a `@elements` block:

```{code-cell} julia
@elements begin
  ele1 = LineElement()
  ele2 = LineElement()
end
println(ele1.name)
println(ele2.name)
```

In SciBmad, all element "kinds" (e.g. `Quadrupole`, `Sextupole`, `Multipole`, etc.) are one
single type `LineElement` under the hood. That is, the constructor for a `Quadrupole` is
precisely:

```julia
Quadrupole(; kwargs...) = LineElement(; kind="Quadrupole", kwargs...)
```

Therefore,

```{code-cell} julia
@elements begin
  qf = Quadrupole()
  sf = Sextupole()
end
println(qf.kind)
println(sf.kind)
```

Such an implementation provides maximal flexibility, allowing you to define an element with
any combination of parameters you may have. For example, there is nothing stopping you from
doing

```{code-cell} julia
d = Drift(L = 1.2)
d.Ks21 = -200 # Set 21st order skew multipole
```

This flexibility makes it easy to adjust the design on the fly. For example, in the Electron
Storage Ring of the Electron-Ion Collider, we need to add multipoles to the drifts in the
interaction region to simulate field crosstalk from the Hadron Storage Ring. With SciBmad,
one does not need to edit the lattice and change these "drifts" to "multipoles"; just set
the multipole!

How an element is tracked through ultimately depends on the parameters defined within that
`LineElement`. For details, see the [Tracking Methods](tracking-methods.md) section of the
documentation.

## Parameters

SciBmad supports a continually-growing list of parameters to define accelerator elements.
To see a full list of the parameters you can set, look at the docstring for the
`LineElement` type. In a Julia session this is retrieved with `?LineElement` at the REPL, or
with `Docs.doc(LineElement)` in a script:

```julia
?LineElement          # in the REPL
Docs.doc(LineElement) # in a script
```

Note that parameters are split into "parameter groups", for organization and convenience.

(pgs)=
## Parameter Groups

`LineElement` and its parameter groups are defined in
[Beamlines.jl](https://bmad-sim.github.io/Beamlines.jl/stable/), so their full docstrings
live on the Beamlines.jl site rather than being duplicated here. The groups are:

`AlignmentParams`
: Alignment of the element with respect to its nominal position (`x_offset`, `y_offset`,
  `tilt`, `x_rot`, `y_rot`, …). Rotations are applied in the order `tilt`, `x_rot`, `y_rot`.

`ApertureParams`
: A mechanical aperture (shape and extent) for the element.

`BMultipoleParams`
: The magnetic (B) multipoles of the element, normal (`Kn1`, `Bn1`, …) or skew
  (`Ks1`, `Bs1`, …), normalized or unnormalized, integrated or not.

`BeamlineParams`
: Present only on elements that live inside a `Beamline`. Holds the `beamline` itself and
  the `beamline_index`, and provides the `s`/`s_downstream` positions.

`BendParams`
: The curvature of the reference coordinate system through the element and the entrance/exit
  edge angles. Does *not* specify any magnetic field — that is `BMultipoleParams`.

`FourPotentialParams`
: The electromagnetic four-potential of the element, given as a function.

`InitialBeamlineParams`
: Reference species and reference energy (`E_ref`, `pc_ref`, `p_over_q_ref`, …) carried by
  the first element of a `Beamline`. See [Defining a Beamline](beamline.md).

`MapParams`
: An arbitrary user-supplied `transport_map` function for the element — a matrix, a phase
  trombone, a neural network, etc.

`MetaParams`
: Extra string properties (`alias`, `label`, `description`) useful for pattern matching or
  bookkeeping.

`PatchParams`
: Properties of patches, which change the reference coordinate system. Rotations are applied
  in the order `dz_rot`, `dx_rot`, `dy_rot`.

`RFParams`
: Parameters generally associated with radiofrequency cavities (`voltage`, `phi0`,
  `rf_frequency`/`harmon`, …).

`UniversalParams`
: The `kind`, `name`, length `L`, and `tracking_method` carried by every `LineElement`.

The physics conventions behind several of these groups are described in
[Element Parameters](element-parameters.md).
