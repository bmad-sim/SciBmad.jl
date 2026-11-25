(dynamicaperture)=
# Dynamic Aperture

The dynamic aperture is the region of starting amplitudes in which particles survive a given
number of turns in a ring, limited by the nonlinear fields (and apertures) of its elements. The
`dynamic_aperture` function finds its boundary in units of the beam sizes, for several energy
deviations δ, as described in the example below.

As with [`track`](track.md), the parameters of the beamline, including
[deferred expressions](defexpr.md), are evaluated once when `dynamic_aperture` is called, and
kept for all turns of that call. To have parameters change during tracking, use
[time-dependent parameters](timedependent.md).

```{notebook} examples/julia/dynamic-aperture-gpu.ipynb
:skip-heading:
```

```{docstring} dynamic_aperture
```
