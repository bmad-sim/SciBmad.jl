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

(defining.latelement)=
# Defining a LatElement

```{code-cell} julia
:tags: [remove-cell]
using SciBmad
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

To construct a `LatElement`,

```{code-cell} julia
ele = LatElement()
```

```{code-cell} python
ele = sb.LatElement()
ele
```

As you can see, the `LatElement` by default comes with a "parameter group" structure
called `UniversalParams`. This contains a `kind` as a string, a `name` as a string, a
length `L`, and various settings specifying how the element should be tracked through, including the `tracking_method`. A bare `LatElement` has no tracking method set. For more details on the tracking methods available, see the [Tracking Methods](tracking-methods.md) section of the documentation. All properties of `UniversalParams` and every other parameter group are defined in the last section of this page.

To set one of these parameters, use the natural syntax

```{code-cell} julia
ele.name = "ele"
ele.kind = "MyElementKind"
ele.L = 123
```

```{code-cell} python
ele.name = "ele"
ele.kind = "MyElementKind"
ele.L = 123
```

Alternatively, we could have set these parameters as keyword arguments during
`LatElement` construction:

```{code-cell} julia
ele = LatElement(name="ele", kind="MyElementKind", L=123)
```

```{code-cell} python
ele = sb.LatElement(name="ele", kind="MyElementKind", L=123)
ele
```

It would often be convenient if we can make the variable symbol (`ele` in this case)
automatically fill in the `name` field for each element. We can do exactly this by wrapping
all element definitions in
{jltext}`a `{jl}`@elements`{jltext}` block`{pytext}`a `{py}`with sb.elements():`{pytext}` block`:

```{code-cell} julia
@elements begin
  ele1 = LatElement()
  ele2 = LatElement()
end
println(ele1.name)
println(ele2.name)
```

```{code-cell} python
with sb.elements():
    ele1 = sb.LatElement()
    ele2 = sb.LatElement()
print(ele1.name)
print(ele2.name)
```

In SciBmad, every element kind (`Quadrupole`, `Sextupole`, `Multipole`, etc.) is the same
type, `LatElement`. The kind only does two things: it sets the `kind` string, and it chooses
a suitable tracking method for the element (see [Element Kinds](#element.kinds) below; a
`CrabCavity` also sets `is_crabcavity = true`):

```{code-cell} julia
@elements begin
  qf = Quadrupole(L=0.5)
  sf = Sextupole(L=0.2)
end
println(qf.kind, ": ", qf.tracking_method)
println(sf.kind, ": ", sf.tracking_method)
```

```{code-cell} python
with sb.elements():
    qf = sb.Quadrupole(L=0.5)
    sf = sb.Sextupole(L=0.2)
print(qf.kind, ": ", qf.tracking_method, sep="")
print(sf.kind, ": ", sf.tracking_method, sep="")
```

Because every element is a `LatElement`, any element can have any combination of
parameters. For example, there is nothing stopping you from adding a multipole to a drift:

```{code-cell} julia
d = Drift(L = 1.2)
d.Ks21 = -200 # Set 21st order skew multipole
```

```{code-cell} python
d = sb.Drift(L=1.2)
d.Ks21 = -200  # Set 21st order skew multipole
```

This flexibility makes it easy to adjust the design on the fly. For example, in the Electron
Storage Ring of the Electron-Ion Collider, we need to add multipoles to the drifts in the
interaction region to simulate field crosstalk from the Hadron Storage Ring. With SciBmad,
one does not need to edit the lattice and change these "drifts" to "multipoles"; just set
the multipole!

The drift above still uses `DriftKick`, which can track through any multipole except a
solenoid. If you were to add a solenoid field to this drift, you would also need to set
`d.tracking_method = SolenoidKick()`; if you forget, tracking will stop with an error telling
you so. Note that changing the `kind` string afterwards does not change the tracking method.
For details, see the [Tracking Methods](tracking-methods.md) section of the documentation.

(element.kinds)=
## Element Kinds

Below are all of the element kinds and their default tracking methods. Any
element with `L = 0` uses `Exact()`, and you can always set a different tracking method,
either as a keyword argument or afterwards.

| Kind         | Default tracking method | Notes                                    |
|--------------|-------------------------|------------------------------------------|
| `Drift`      | `DriftKick()`           |                                          |
| `Quadrupole` | `MatrixKick()`          |                                          |
| `Sextupole`  | `DriftKick()`           |                                          |
| `Octupole`   | `DriftKick()`           |                                          |
| `Multipole`  | `DriftKick()`           |                                          |
| `SBend`      | `BendKick()`            |                                          |
| `Solenoid`   | `SolenoidKick()`        |                                          |
| `Kicker`     | `DriftKick()`           |                                          |
| `HKicker`    | `DriftKick()`           |                                          |
| `VKicker`    | `DriftKick()`           |                                          |
| `RFCavity`   | `SolenoidKick()`        |                                          |
| `CrabCavity` | `SolenoidKick()`        | Also sets `is_crabcavity = true`         |
| `Marker`     | `Exact()`               | For any `L`                              |
| `BPM`        | `Exact()`               | For any `L`                              |
| `Patch`      | `Exact()`               | For any `L`                              |
| `Map`        | `Exact()`               | For any `L`                              |

A bare `LatElement()` has no tracking method, so you must set one yourself before tracking
through it.

## Parameters

SciBmad supports a continually-growing list of parameters to define accelerator elements.
To see a full list of the parameters you can set, look at the docstring for the
`LatElement` type, reproduced below. In a Julia session it can be retrieved with
`Docs.doc(LatElement)`{pytext}`, in Python with `{py}`help(sb.LatElement)`.

```{docstring} LatElement
```

Note that parameters are split into "parameter groups", for organization and convenience.
They are all documented below. 

(pgs)=
## Parameter Groups

Parameter groups are automatically added to an element when a corresponding property is set.
Parameter groups can be removed from an element as a whole by setting that parameter group to
{jl}`nothing`{py}`None`; for example, to remove `MapParams` from an element named `ele`, do
{jl}`ele.MapParams = nothing`{py}`ele.MapParams = None`.

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

(emfield.params)=
### EMFieldParams

```{docstring} EMFieldParams
```

### EMultipoleParams

```{docstring} EMultipoleParams
```

### FourPotentialParams

```{docstring} FourPotentialParams
```

(ibs.params)=
### IBSParams

```{docstring} IBSParams
```

The `ibs_model` parameter takes an `IBSModel`:

```{docstring} IBSModel
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

All tracking methods for RF cavities use the same phase convention. A particle of the
reference species that arrives a time $\Delta t$ after the reference particle gains the
energy $|q| V \cos(\phi_0 + \phi_\mathrm{zero} + \omega \Delta t)$, where $\omega$ is the
angular RF frequency and $\phi_\mathrm{zero}$ is $0$ for `PhaseRef.Accelerating`, $+\pi/2$
for `PhaseRef.AboveTransition`, and $-\pi/2$ for `PhaseRef.BelowTransition`. So with
`phi0 = 0`, a particle arriving late loses energy above transition and gains energy below
transition, which is what makes the motion stable in each case.

### UniversalParams

```{docstring} UniversalParams
```

The `fringe_at` parameter takes a `Fringe`:

```{docstring} Fringe
```
