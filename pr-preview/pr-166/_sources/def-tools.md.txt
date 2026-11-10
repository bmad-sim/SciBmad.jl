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

## Phase Trombone

A phase trombone is an element that changes the phase advances, and so the tunes, of a ring
without changing the Twiss parameters anywhere else. This is useful e.g. for scanning the
tunes in a study without having to rematch the optics. `phase_trombone!` turns any element in
a `Beamline` into a phase trombone. For example, using a `Marker` at the end of a FODO cell:

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
phase_trombone!(fodo[tr][1]; phi1=0.1, phi2=-0.05)

tw = twiss(fodo)
println("After:  q1 = ", tw.q1, ", q2 = ", tw.q2)
```

`phi1`, `phi2`, and `phi3` are stored as the first three entries of the element's
`transport_map_params`, so they can be changed at any time without calling
`phase_trombone!` again. `transport_map_params` is a tuple, which cannot be changed in place,
so set a new one with `Base.setindex`:

```{code-cell} julia
t = fodo[tr][1]
# Now add 0.2 to the horizontal-like tune instead (the vertical-like stays -0.05):
t.transport_map_params = Base.setindex(t.transport_map_params, 0.2, 1)

tw = twiss(fodo)
println("Now:    q1 = ", tw.q1, ", q2 = ", tw.q2)
```

The phase trombone is built from the closed orbit and the Twiss parameters at the element at
the time `phase_trombone!` is called. If you change the lattice afterwards, call it again.

```{docstring} phase_trombone!
```
