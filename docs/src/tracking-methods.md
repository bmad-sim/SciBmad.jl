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

(track.methods)=
# Tracking Methods

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

Every element has a `tracking_method`, which sets how particles are pushed through it. When
you construct an element with one of the element kinds, a default tracking method is chosen
for you (see the table of [Element Kinds](#element.kinds)), for example `MatrixKick()` for a
`Quadrupole` and `BendKick()` for an `SBend`. Any element with `L = 0` uses `Exact()`.

A bare `LatElement()` has no tracking method, so you must set one yourself before tracking. The default can always be overridden, either during construction or afterwards:

```{code-cell} julia
qf = Quadrupole(L=0.5, Kn1=0.36, tracking_method=DriftKick(order=6))
qf.tracking_method = MatrixKick()
```

```{code-cell} python
qf = sb.Quadrupole(L=0.5, Kn1=0.36, tracking_method=sb.DriftKick(order=6))
qf.tracking_method = sb.MatrixKick()
```

Since any element can contain any combination of parameters, not every tracking method can
handle every element. If it can't, tracking stops with an error telling you which tracking
method to use instead. For example, `DriftKick` cannot track through a solenoid field:

```{code-cell} julia
:tags: [raises-exception]
d = Drift(L=1.0, Ksol=0.1)
bl = Beamline([d], species_ref=Species("electron"), pc_ref=1e9)
track(bl, v0=zeros(1, 6))
```

```{code-cell} python
:tags: [raises-exception]
d = sb.Drift(L=1.0, Ksol=0.1)
bl = sb.Beamline([d], species_ref=sb.Species("electron"), pc_ref=1e9)
sb.track(bl, v0=np.zeros((1, 6)))
```

```{code-cell} julia
d.tracking_method = SolenoidKick()
track(bl, v0=[1e-3 0 0 0 0 0]).v[turn=1, particle=1]
```

```{code-cell} python
d.tracking_method = sb.SolenoidKick()
sb.track(bl, v0=[[1e-3, 0, 0, 0, 0, 0]]).v(turn=1, particle=1)
```

(track.steps)=
## Integration Steps and Slices

The tracking method decides *how* each integration step is taken. How many steps are taken,
and a few other settings, are set in the element's `UniversalParams`:

- `n_steps` or `ds_step`: the number of integration steps, or the largest length of each step.
  Setting one of these sets the other (`n_steps = ceil(L / n_slices / ds_step)`, and the
  steps all have the length `L / n_slices / n_steps`).
- `n_slices`: the number of slices to split the element into, each with `n_steps`
  integration steps. Collective effects (see [Collective Effects](collective.md)) and a
  ramping reference energy are recomputed and applied once per slice. If you only need more
  accuracy, increase `n_steps` instead, which is faster.
- `fringe_at`: which ends of the element include fringe fields. Fringe fields are only
  included at the ends of the whole element, not between its slices.
- `rad_damping`: `true` if tracking should include the deterministic damping effect of radiation
- `rad_fluctuations`: `true` if tracking should include the stochastic kicks of radiation

```{code-cell} julia
sf = Sextupole(L=0.2, Kn2=1.2, n_steps=4, rad_damping=true)
```

```{code-cell} python
sf = sb.Sextupole(L=0.2, Kn2=1.2, n_steps=4, rad_damping=True)
```

Patches, maps, and elements tracked with `SaganCavity` (see [Sagan Cavity](#track.sagan) below,
whose kicks are set by `n_cells`) are always tracked in one step, so `n_steps` must be 1 for
them. Elements with a patch or a map, and presently elements tracked with `SaganCavity`, also
cannot be sliced.

## Symplectic Integrators

`DriftKick`, `MatrixKick`, `BendKick`, `SolenoidKick`, and `Implicit` are the various symplectic
integrators provided in SciBmad. The first four split the motion through the element into a simple part that can be solved directly, and "kicks" with the rest in between. They differ in how the motion is split:

- `DriftKick`: drifts exactly, and kicks with all of the multipoles. This works for any
  straight element without a solenoid field, including RF cavities.
- `MatrixKick`: tracks an approximate energy-dependent quadrupole, and kicks with its
  corrections and other multipoles. This is usually the best choice for straight, aligned elements
  with a strong quadrupole field.
- `SolenoidKick`: tracks through the solenoid field exactly, and kicks with the rest of the
  multipoles. This works for any straight element, with or without a solenoid field,
  including RF cavities.
- `BendKick`: tracks through a bending coordinate system and dipole strength exactly, and kicks with 
  the rest of the multipoles. This is the usual choice for elements that bend the reference trajectory (see [BendParams](#bend.params)).
- `Implicit`: an implicit integrator for arbitrary electromagnetic fields defined by a
  four-potential (see `FourPotentialParams`). All fields in the element must be included in
  the four-potential.

All of these take the same two settings:

```{docstring} DriftKick
```

`Implicit` also has `use_newton` specifying if a Newton iteration should be used for implicit solve (default is `false`, to use fixed point iteration):

```{docstring} Implicit
```

A higher order integrator is more accurate per step, but each step takes longer: with the
optimized schemes, an order 4 step is made of 5 second order steps, and an order 6 step of
13. Whether you get more accuracy for the same speed by increasing the order or by
increasing the number of steps depends on the element, so it is worth checking for your
lattice.

If an element is a pure drift, a pure bend, or a pure solenoid, it is tracked through
exactly, and the order and number of steps don't change the result. The exception is when
radiation is included, because the radiation is applied at each integration step. The steps
are still taken in any case, so [callbacks](#track.callbacks) (e.g. used by `twiss` to
compute its results inside elements) are executed at every step.

## Exact

```{docstring} Exact
```

`Exact` is used for elements that are not integrated in steps: patches, maps, markers, and
thin (zero length) multipoles. It can also track exactly through a nonzero length element
with no field (a drift), only a solenoid field, only a dipole field (also in a bend), or only
an electric dipole field (`EMultipoleParams`). Setting the length of an element to zero
switches its tracking method to `Exact` automatically.

## Runge Kutta

```{docstring} RK4
```

`RK4` integrates the equations of motion through an arbitrary electromagnetic field, defined
as a function of the particle position and time in [EMFieldParams](#emfield.params),
together with any multipoles and bend in the element. It is not symplectic, and does not
support spin tracking at the moment.

(track.sagan)=
## Sagan Cavity

```{docstring} SaganCavity
```

See [SaganCavity Tracking](sagancavity-tracking.md) for how to use it, and
[SaganCavity Physics](sagancavity-physics.md) for the model.
