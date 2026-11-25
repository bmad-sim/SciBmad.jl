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

(def.tools)=
# Definition Tools

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

## Phase Trombone

A phase trombone is an element that changes the phase advances without changing the Twiss parameters
anywhere. This can be useful for dynamic aperture optimizations where the phase advance between arcs
can have a significant impact on the nonlinear correction scheme.
{jl}`phase_trombone!`{py}`sb.phase_trombone` turns any element in a `Beamline` into a phase
trombone. For example, using a `Marker` at the end of a FODO cell:

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-1.2, L=0.2)
  tr = Marker()
end

fodo = Beamline([qf, sf, b, d, qd, sd, b, d, tr],
        species_ref=Species("electron"), pc_ref=18e9)

tw = twiss(fodo)
println("Before: q1 = ", tw.q1, ", q2 = ", tw.q2)

# Add 0.1 to the horizontal-like tune and subtract 0.05 from the vertical-like one:
phase_trombone!(fodo[tr]; phi1=0.1, phi2=-0.05)

tw = twiss(fodo)
println("After:  q1 = ", tw.q1, ", q2 = ", tw.q2)
```

```{code-cell} python
with sb.elements():
    qf = sb.Quadrupole(Kn1=0.36, L=0.5)
    sf = sb.Sextupole(Kn2=1.2, L=0.2)
    d = sb.Drift(L=0.1)
    b = sb.SBend(L=1.2, angle=np.pi/132)
    qd = sb.Quadrupole(Kn1=-0.36, L=0.5)
    sd = sb.Sextupole(Kn2=-1.2, L=0.2)
    tr = sb.Marker()

fodo = sb.Beamline([qf, sf, b, d, qd, sd, b, d, tr],
                   species_ref=sb.Species("electron"), pc_ref=18e9)

tw = sb.twiss(fodo)
print(f"Before: q1 = {tw.q1}, q2 = {tw.q2}")

# Add 0.1 to the horizontal-like tune and subtract 0.05 from the vertical-like one:
sb.phase_trombone(fodo[tr], phi1=0.1, phi2=-0.05)

tw = sb.twiss(fodo)
print(f"After:  q1 = {tw.q1}, q2 = {tw.q2}")
```

`phi1`, `phi2`, and `phi3` are stored as the first three entries of the element's
`transport_map_params`, so they can be changed at any time without calling
{jl}`phase_trombone!`{py}`phase_trombone` again. `transport_map_params` is a tuple, which cannot be
changed in place, so it's best to let them be deferred expressions stored in the beamline context:

```{code-cell} julia
# Remove the current phase trombone map by setting MapParams = nothing:
fodo[tr].MapParams = nothing

tw = twiss(fodo)
println("Before: q1 = ", tw.q1, ", q2 = ", tw.q2)

# Add variables to the context:
fodo.context.tr_phi1 = 0.1
fodo.context.tr_phi2 = -0.05

# Set the phase trombone angles to be deferred expressions
phase_trombone!(fodo[tr]; phi1=DefExpr(c -> c.tr_phi1), phi2=DefExpr(c -> c.tr_phi2))

tw = twiss(fodo)
println("After: q1 = ", tw.q1, ", q2 = ", tw.q2)

# Change the context variables to adjust the phases
fodo.context.tr_phi1 = 0.2
tw = twiss(fodo)
println("After again: q1 = ", tw.q1, ", q2 = ", tw.q2)
```

```{code-cell} python
# Remove the current phase trombone map by setting MapParams = None:
fodo[tr].MapParams = None

tw = sb.twiss(fodo)
print(f"Before: q1 = {tw.q1}, q2 = {tw.q2}")

# Add variables to the context:
fodo.context.tr_phi1 = 0.1
fodo.context.tr_phi2 = -0.05

# Set the phase trombone angles to be deferred expressions
sb.phase_trombone(fodo[tr], phi1=sb.DefExpr(lambda c: c.tr_phi1),
                  phi2=sb.DefExpr(lambda c: c.tr_phi2))

tw = sb.twiss(fodo)
print(f"After: q1 = {tw.q1}, q2 = {tw.q2}")

# Change the context variables to adjust the phases
fodo.context.tr_phi1 = 0.2
tw = sb.twiss(fodo)
print(f"After again: q1 = {tw.q1}, q2 = {tw.q2}")
```

The phase trombone is built from the closed orbit and the linear optics at the element at the time
{jl}`phase_trombone!`{py}`phase_trombone` is called. As such, if you plan to change the closed orbit
and/or linear optics afterwards, first remove the phase trombone before doing anything, and then
call {jl}`phase_trombone!`{py}`phase_trombone` again when the orbit and linear optics are not
changing anymore.

```{docstring} phase_trombone!
```
