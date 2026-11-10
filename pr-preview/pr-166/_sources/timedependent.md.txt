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

(timedependentramp)=
# Time-Dependent Parameters and Ramping

```{code-cell} julia
:tags: [remove-cell]
using SciBmad
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

## Time-Dependent Parameters

In SciBmad, almost any `LatElement` parameter may be set to have an arbitrary time-dependence
using `Time()`. For example to construct an AC-kicker with amplitude `1e-4` T and angular
frequency `1e6`, we can simply write

```{code-cell} julia
t = Time()
ac_kicker = VKicker(L=0.5, Ks0=1e-4*sin(1e6*t));
```

Here, `Time()` refers to the wall-clock, global reference time, also called the "absolute"
time in classical Bmad parlance. And that's it! We can now track using the same
(GPU-compatible) tracking function described in [Track](track.md). Note that a `Bunch` can
be initialized with a specific `t_ref` or even `p_over_q_ref` if desired.

:::{note}
Time-dependent parameters are evaluated for each particle at its own time, so its z offset
is taken into account. However, they are only evaluated at the entrance of the element, and
are assumed constant through it. If the parameter changes a lot during the time a particle
takes to go through the element, split the element into slices with `n_slices` (see
[Integration Steps and Slices](#track.steps)): the parameters are then evaluated again at
the entrance of each slice.
:::

Note that any expression including `Time()` builds up a function, stored as a
`TimeFunction`. This can be evaluated for time using the natural syntax

```{code-cell} julia
tf = 1e-4*sin(1e6*t)
println(tf(0))
println(tf(pi/(2e6)))
```

As long as the function you write is GPU-compatible, such as in the above example, then it
will work on the GPU.

Really, anything can be a time-dependent parameter. Even misalignments:

```{code-cell} julia
earthquake = Quadrupole(Kn1=0.36, L=0.5, x_offset=0.1*sin(120*t), y_offset=0.1*sin(120*t));
```

## Energy Ramping

In SciBmad, "ramping" refers to ramping the reference energy of a `Bunch`. To ramp, simply
set the `Beamline` reference energy to be a function of time:

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-1.2, L=0.2)
end

fodo = Beamline([qf, sf, b, d, qd, sd, b, d], species_ref=Species("electron"))

fodo.E_ref = 18e9 + 4e11*Time();
```

Now, the unnormalized magnetic field strengths are time-dependent too, as they are derived
from the reference energy (here `Kn1` is the independent variable, so the fields are ramped
with the energy):

```{code-cell} julia
println(fodo[1].Bn1(0))
println(fodo[1].Bn1(1e-3))
```

The reference energy is evaluated at the reference time at the entrance of each element slice.

When ramping, two [configuration settings](track.md) that we can pass
to `track` become relevant:

- `ramp_particle_energy_without_rf`: If `true` when ramping (`Beamline`'s reference energy is
  a `TimeFunction`), then particle energies will be artificially ramped with the reference
  energy. Default is `false`.
- `ramp_update_each_particle`: If `true` when ramping, then the z-offset/time-offset of each
  individual particle at the element entrances is taken into account when tracking particles
  through the time-dependent element parameters. Usually only needed if ramping is fast
  compared to the time scale of bunch passage through elements. Default is `false`.

In this FODO cell example above, we would need to use `ramp_particle_energy_without_rf=true`,
because there is no RF to increase the particle's energy as the fields are ramped. Both flags
are GPU compatible.

Once again, we can now track using the same (GPU-compatible) tracking function described in
[Track](track.md).

## Limitations

- The element length `L` and the bend curvature `g_ref` (and so the bend `angle`) cannot be
  time-dependent: they set the reference coordinate system, which is fixed.
- `SaganCavity` does not support a ramping reference energy.