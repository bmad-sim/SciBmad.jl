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

(ignore.parameters)=
## Ignoring Parameter Groups with `ignore_parameters`

It is often useful to track through an element as if some of its parameter groups were not
there, e.g. to compare tracking with and without misalignments, or with and without
apertures. Instead of removing the parameter groups and later restoring them, put the names
of the parameter groups to ignore in the element's `ignore_parameters` list, which is a
property of its `UniversalParams`:

```{code-cell} julia
qf = Quadrupole(L=0.5, Kn1=0.36, x_offset=1e-3,
                x1_limit=-0.02, x2_limit=0.02, y1_limit=-0.01, y2_limit=0.01)
qf.ignore_parameters = [:AlignmentParams, :ApertureParams]
qf
```

When the `ignore_parameters` list is not empty, it is printed with the element's
`UniversalParams` as above. The parameter groups themselves are left untouched, so to
switch a parameter group back in, just remove it from the list:

```{code-cell} julia
filter!(!=(:ApertureParams), qf.ignore_parameters) # Use the ApertureParams again
qf.ignore_parameters
```

Some other ways of setting `ignore_parameters`:

```{code-cell} julia
push!(qf.ignore_parameters, :BMultipoleParams) # Add to the list
qf.ignore_parameters = :AlignmentParams        # A single parameter group
qf.ignore_parameters = []                      # Use all parameter groups again
sf = Sextupole(L=0.2, Kn2=10, ignore_parameters=[:BMultipoleParams]) # As a keyword argument
sf.ignore_parameters
```

`ignore_parameters` is used by tracking, so it affects everything in SciBmad that is
computed by tracking, e.g. `track`, `twiss`, and `find_closed_orbit`. Other properties of
the element (e.g. its length or bend angle, and the `s` positions in a `Beamline`) are
unaffected.

A parameter group in `ignore_parameters` is treated as if it were absent from the element.
For example, a `Quadrupole` with `:BMultipoleParams` in `ignore_parameters` is tracked as a
drift, and an `SBend` with `:BendParams` in `ignore_parameters` is tracked as a straight
element with the same multipoles.

### `ignore_parameters` in a Beamline

When an element is placed in a `Beamline`, every instance of that element in the `Beamline`
shares the `UniversalParams` of the original element, and therefore also its
`ignore_parameters` list. So setting `ignore_parameters` on the original element, or on any
of its instances in a `Beamline`, switches the parameter groups in or out for all
instances:

```{code-cell} julia
qf.ignore_parameters = [:AlignmentParams]
bl = Beamline([qf, Drift(L=1.0), qf], species_ref=Species("electron"), E_ref=18e9)
bl.line[3].ignore_parameters
```

### Allowed symbols

To catch misspellings, only symbols in the set `Beamlines.IGNORE_PARAMETERS_SYMBOLS` are
allowed in `ignore_parameters`. By default these are the parameter groups used in tracking:

```{code-cell} julia
Beamlines.IGNORE_PARAMETERS_SYMBOLS
```

Any other symbol throws an error, both when `ignore_parameters` is set, and when tracking
through the element (which catches invalid symbols added with e.g. `push!`):

```julia
qf.ignore_parameters = [:AlignmentParam] # Error: Invalid symbol :AlignmentParam in `ignore_parameters`...
```

If you define your own parameter group, e.g. `MyParams <: AbstractParams`, register it so
that it can be switched off too:

```julia
push!(Beamlines.IGNORE_PARAMETERS_SYMBOLS, :MyParams)
```

A registered symbol does not need to be the name of a parameter group. Custom tracking code
can check for any registered symbol with `:MySymbol in ele.ignore_parameters`.

### Checking `ignore_parameters` in tracking code

Tracking code decides whether to use a parameter group with `Beamlines.isactive`. Passing
an element's `ignore_parameters` list as the second argument makes `isactive` return
`false` for a parameter group whose name is in the list:

```{code-cell} julia
qf.ignore_parameters = [:AlignmentParams]
ig = qf.ignore_parameters
Beamlines.isactive(qf.AlignmentParams, ig), Beamlines.isactive(qf.BMultipoleParams, ig)
```

```{docstring} Beamlines.isactive
```

## Parameters

SciBmad supports a continually-growing list of parameters to define accelerator elements.
To see a full list of the parameters you can set, look at the docstring for the
`LineElement` type, reproduced below. In a Julia session it can be retrieved with
`Docs.doc(LineElement)`.

```{docstring} LineElement
```

Note that parameters are split into "parameter groups", for organization and convenience.
They are all documented below.

(pgs)=
## Parameter Groups

(alignment.params)=
(alignment:params)=
### AlignmentParams

```{docstring} AlignmentParams
```

(aperture.params)=
(aperture:params)=
### ApertureParams

```{docstring} ApertureParams
```

(multipole.sol.params)=
(multipole.solenoid:params)=
### BMultipoleParams

```{docstring} BMultipoleParams
```

### BeamlineParams

```{docstring} BeamlineParams
```

(bend.params)=
(bend:params)=
### BendParams

```{docstring} BendParams
```

### FourPotentialParams

```{docstring} FourPotentialParams
```

### InitialBeamlineParams

```{docstring} InitialBeamlineParams
```

### MapParams

```{docstring} MapParams
```

### MetaParams

```{docstring} MetaParams
```

(patch.params)=
(patch:params)=
### PatchParams

```{docstring} PatchParams
```

(rf.params)=
(rf:params)=
### RFParams

```{docstring} RFParams
```

The `zero_phase` parameter takes a `PhaseRef`:

```{docstring} PhaseRef
```

### UniversalParams

```{docstring} UniversalParams
```
