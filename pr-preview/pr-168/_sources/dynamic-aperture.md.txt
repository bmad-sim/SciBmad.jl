(dynamicaperture)=
# Dynamic Aperture

As with [`track`](track.md), the parameters of the beamline, including
[deferred expressions](defexpr.md), are evaluated once when `dynamic_aperture` is called, and
kept for all turns of that call. To have parameters change during tracking, use
[time-dependent parameters](timedependent.md).

```{notebook} examples/julia/dynamic-aperture-gpu.ipynb
:skip-heading:
```

```{docstring} dynamic_aperture
```
