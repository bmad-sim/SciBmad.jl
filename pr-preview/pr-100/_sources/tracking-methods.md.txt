(track.methods)=
# Tracking Methods

Every `LineElement` carries a `tracking_method`, which determines how particles are pushed
through that element. It is set like any other parameter:

```julia
qf = Quadrupole(Kn1=0.36, L=0.5, tracking_method=Symplectic(order=4))

# or, for a whole beamline:
tm = Symplectic(order=2, radiation_damping_on=true)
foreach(ele -> ele.tracking_method = tm, fodo.line)
```

The default is `SciBmadStandard`, which uses exact transport maps when the element is
exactly solvable, and otherwise a symplectic integrator with a split chosen appropriately
for the element. Which map is ultimately used therefore depends on the parameters set in
the element — see [Defining a LineElement](element.md).

The methods available are:

`SciBmadStandard`
: The default. Exact transport maps where solvable, else `Yoshida(order=4, n_steps=1)`
  with an element-appropriate split. Accepts `radiation_damping_on`,
  `radiation_fluctuations_on`, `ibs_damping_on`, and `ibs_fluctuations_on`.

`Symplectic`
: An explicitly symplectic integrator that automatically selects the splitting for each
  element. Keyword arguments include `order` (2, 4, 6, 8, or 10), the step size as either
  `n_steps` or `ds_step`, `fringe_at`, and the radiation/IBS flags above.

`MatrixKick`, `BendKick`, `SolenoidKick`, `DriftKick`
: The same integrator, but with the splitting fixed explicitly rather than chosen
  automatically. They take the same keyword arguments as `Symplectic`.

`Exact`
: The exact transport map for the element, with `fringe_at` selecting which ends get fringe
  maps.

`SaganCavity`
: RF cavity tracking with both longitudinal energy gain and transverse focusing. Documented
  below.

The tracking methods themselves are implemented in
[BeamTracking.jl](https://github.com/bmad-sim/BeamTracking.jl); their docstrings are
available in a Julia session with e.g. `?Symplectic`.

---

(sagancavity.tracking)=
```{include} tracking/sagancavity-tracking.md
```
