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

(rd.orbit)=
# 14. Orbit Correction

```{code-cell} julia
:tags: [remove-cell]
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

While operating accelerators, the orbit of the beam is measured with beam position monitors
(BPMs). The accelerator is designed with ideal BPM readings in mind, which are often 0. After
extended operational experience, operators learn good readings for all BPMs, which are referred to
as golden orbits. It then becomes important to change the trajectory of the beam to obtain the
desired BPM readings, with corrector magnets.

## Example: Adding Corrector Coils and BPMs

```{figure} figures/correctors.svg
:width: 90%

FODO cell with corrector coils.
```

Start with the ring with sextupoles of [chapter 12](dynamic-aperture.md):

```{code-cell} julia
include("lattices/12-sextupoles.jl");
```

```{code-cell} python
from lattices.ch12_sextupoles import *
```

1. Corrector coils are placed in the `d1` drifts, 6.4 cm away from the quadrupoles: downstream of the
   quadrupoles in the forward cells and upstream in the reverse cells. Horizontal correctors are put
   next to focusing quadrupoles, where the horizontal beta function is large, and vertical
   correctors next to defocusing quadrupoles. Define the drifts that fit the 0.2 m long correctors,
   and a marker for the BPMs:

```{code-cell} julia
@elements begin
  d1c1 = Drift(L=0.064)
  d1c2 = Drift(L=0.609 - 0.064 - 0.2)
  bpm = Marker()
end;
```

```{code-cell} python
with sb.elements():
    d1c1 = sb.Drift(L=0.064)
    d1c2 = sb.Drift(L=0.609 - 0.064 - 0.2)
    bpm = sb.Marker()
```

2. Each corrector must be an element of its own, so that its kick can be set by itself. So instead of
   defining one corrector and placing it many times, a new corrector is made for each `d1`. Because
   `d1` is used throughout the ring, adding the correctors by editing the lines would be tedious. A
   beamline is built from a Julia vector of elements, so a loop over the elements of the ring can do
   it instead: it replaces each `d1` with the two drifts and a new corrector, depending on the
   quadrupole next to it. The loop also puts a BPM after every bend `b` and `bh`, and after every
   straight section drift `db`:

```{code-cell} julia
line = [sextant1; sextant3; sextant5; sextant7; sextant9; sextant11]
n = length(line)
elements = LatElement[]
hkickers = LatElement[]
vkickers = LatElement[]
for i in 1:n
  ele = line[i]
  if ele === d1
    # The quadrupole next to d1: before it in forward cells, after it in reverse cells (the
    # next element of the ring after the last one is the first one)
    forward = line[i-1].kind == "Quadrupole"
    j = forward ? i - 1 : mod1(i + 1, n)
    while line[j].kind != "Quadrupole"
      j = mod1(j + 1, n)
    end
    if line[j].Kn1 > 0
      kicker = HKicker(name="ch_$(length(hkickers) + 1)", L=0.2)
      push!(hkickers, kicker)
    else
      kicker = VKicker(name="cv_$(length(vkickers) + 1)", L=0.2)
      push!(vkickers, kicker)
    end
    append!(elements, forward ? [d1c1, kicker, d1c2] : [d1c2, kicker, d1c1])
  else
    push!(elements, ele)
  end
  if ele === b || ele === bh || ele === db
    push!(elements, bpm)
  end
end

ring = Beamline(elements, species_ref=electron, pc_ref=pc_ref, context=ring.context)
length(hkickers), length(vkickers), length(ring[[bpm]])
```

```{code-cell} python
line = sextant1 + sextant3 + sextant5 + sextant7 + sextant9 + sextant11
n = len(line)
elements = []
hkickers = []
vkickers = []
for i in range(n):
    ele = line[i]
    if ele is d1:
        # The quadrupole next to d1: before it in forward cells, after it in reverse cells (the
        # next element of the ring after the last one is the first one)
        forward = line[i-1].kind == "Quadrupole"
        j = i - 1 if forward else (i + 1) % n
        while line[j].kind != "Quadrupole":
            j = (j + 1) % n
        if line[j].Kn1 > 0:
            kicker = sb.HKicker(name=f"ch_{len(hkickers) + 1}", L=0.2)
            hkickers.append(kicker)
        else:
            kicker = sb.VKicker(name=f"cv_{len(vkickers) + 1}", L=0.2)
            vkickers.append(kicker)
        elements += [d1c1, kicker, d1c2] if forward else [d1c2, kicker, d1c1]
    else:
        elements.append(ele)
    if ele is b or ele is bh or ele is db:
        elements.append(bpm)

ring = sb.Beamline(elements, species_ref=electron, pc_ref=pc_ref, context=ring.context)
len(hkickers), len(vkickers), len(ring[[bpm]])
```

   The new ring keeps the context of the previous one, which holds the variables of the sextupoles
   and the phase trombones.

3. A phase trombone is a matrix applied around the closed orbit at its place, built from the
   closed orbit and the lattice functions at the time {jl}`phase_trombone!`{py}`sb.phase_trombone`
   is called. The radiation
   and the correctors change the closed orbit, so the phase trombones are taken out now, and put
   back at the very end, after the orbit correction:

```{code-cell} julia
for tr in (tr_f, tr_r, tr_m)
  ring[tr].MapParams = nothing
end
```

```{code-cell} python
for tr in (tr_f, tr_r, tr_m):
    ring[tr].MapParams = None
```

## Example: Radiation-Induced Sawtooth Orbit Correction

```{figure} figures/sawtooth.svg
:width: 90%

A) The sawtooth orbit due to synchrotron radiation (in Bmad). B) The orbit after correction using
kickers and BPMs.
```

1. A sawtooth orbit is caused by the energy loss to synchrotron radiation around the ring. The
   energy of the particles changes along the ring (it decreases in the bends and is restored in the
   RF cavities), which makes the horizontal closed orbit a sawtooth through the dispersion. Turn on
   radiation damping in all elements:

```{code-cell} julia
for ele in ring.line
  ele.rad_damping = true
end
tw = twiss(ring, cols=["x", "pz"])
using CairoMakie
fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1], ylabel="x [mm]")
lines!(ax, tw.s, 1e3 .* tw.x)
ax2 = Axis(fig[2, 1], xlabel="s [m]", ylabel="pz")
lines!(ax2, tw.s, tw.pz)
fig
```

```{code-cell} python
for ele in ring.line:
    ele.rad_damping = True
tw = sb.twiss(ring, cols=["x", "pz"])
from matplotlib.figure import Figure
fig = Figure(figsize=(8, 4), layout="constrained")
ax, ax2 = fig.subplots(2, 1)
ax.plot(tw.s, 1e3*tw.x)
ax.set_ylabel("x [mm]")
ax2.plot(tw.s, tw.pz)
ax2.set(xlabel="s [m]", ylabel="pz")
fig
```

2. Correct the orbit with the horizontal correctors: make the orbit at the BPMs as small as possible
   (in the least-squares sense), with `minimize` (see [Orbit Correction](../optimize.md)). There
   are more BPM readings than correctors, so the orbit cannot be zero at all of them. Each corrector
   is a knob, and its strength is its normal dipole field `Kn0L` (a kick of `-Kn0L`).

```{code-cell} julia
tw_orbit = TwissExpr(ring)
res = match!(vary=[Knob(kicker, :Kn0L) for kicker in hkickers], minimize=tw_orbit.x[[bpm]])
res.converged, res.iterations
```

```{code-cell} python
tw_orbit = sb.TwissExpr(ring)
res = sb.match(vary=[sb.Knob(kicker, "Kn0L") for kicker in hkickers], minimize=tw_orbit.x[[bpm]])
res.converged, res.iterations
```

```{code-cell} julia
tw2 = twiss(ring, cols=["x"])
lines!(ax, tw2.s, 1e3 .* tw2.x)
fig
```

```{code-cell} python
tw2 = sb.twiss(ring, cols=["x"])
ax.plot(tw2.s, 1e3*tw2.x)
fig
```

   The correctors kill the orbit at the BPMs, but between them, in the bends, the sawtooth
   remains, because the energy changes along each bend.

## Example: Orbit Correction with Quadrupole Misalignments

1. Every placement of a quadrupole such as `qf` is the same element, with the same parameters. To
   misalign each quadrupole by itself, first make a copy of the ring in which every element is an
   element of its own. {jl}`deepcopy`{py}`copy.deepcopy` copies an element of a beamline into a new,
   independent element,
   with its parameters (including deferred expressions). The copy keeps the context of the ring:

```{code-cell} julia
ring_err = Beamline(deepcopy.(ring.line), species_ref=electron, pc_ref=pc_ref,
                    context=ring.context)
```

```{code-cell} python
import copy
ring_err = sb.Beamline([copy.deepcopy(ele) for ele in ring.line], species_ref=electron,
                       pc_ref=pc_ref, context=ring.context)
```

   Its elements are new, so they are found by their names, e.g. `ring_err["bpm"]` for all BPMs.

2. Add random horizontal offsets to all quadrupoles, from a Gaussian distribution with
   $\sigma = 10$ µm, cut at $3\sigma$:

```{code-cell} julia
using Random, Statistics
Random.seed!(1)
for q in ring_err["Quadrupole::*"]
  dx = 1e-5*randn()
  while abs(dx) > 3e-5
    dx = 1e-5*randn()
  end
  q.x_offset = dx
end
tw_err = TwissExpr(ring_err)
maximum(abs, value(tw_err.x["bpm"])) # The largest BPM reading [m]
```

```{code-cell} python
rng = np.random.default_rng(1)
for q in ring_err["Quadrupole::*"]:
    dx = 1e-5*rng.standard_normal()
    while abs(dx) > 3e-5:
        dx = 1e-5*rng.standard_normal()
    q.x_offset = dx
tw_err = sb.TwissExpr(ring_err)
np.max(np.abs(sb.value(tw_err.x["bpm"])))  # The largest BPM reading [m]
```

3. Correct the orbit with the same process:

```{code-cell} julia
res = match!(vary=[Knob(kicker, :Kn0L) for kicker in ring_err["HKicker::*"]],
             minimize=tw_err.x["bpm"])
maximum(abs, value(tw_err.x["bpm"]))
```

```{code-cell} python
res = sb.match(vary=[sb.Knob(kicker, "Kn0L") for kicker in ring_err["HKicker::*"]],
               minimize=tw_err.x["bpm"])
np.max(np.abs(sb.value(tw_err.x["bpm"])))
```

4. If the misalignments are large, the closed orbit can become unstable, and then there is no closed
   orbit to correct. Try larger misalignments, with $\sigma = 100$ µm. One approach to avoid unstable
   orbits is to correct the orbit of an open line first: the trajectory of one turn through the ring,
   starting on the design orbit. A `TwissExpr` of the ring with `initial` (the lattice functions at
   the start, e.g. a row of the periodic `twiss` table of the design) treats the ring as an open
   line. After the trajectory is corrected, the orbit excursions are much smaller, the periodic orbit
   is far away from the unstable region, and the closed orbit can be corrected.

## Example: Putting the Phase Trombones Back

The orbit of `ring` is corrected, so its phase trombones can be put back, with the phase advances
of [chapter 12](dynamic-aperture.md). They are now built around the closed orbit with the radiation
and the correctors, so they do not change it:

```{code-cell} julia
reading = value(tw_orbit.x[[bpm]]) # The BPM readings without the phase trombones
phase_trombone!(ring[tr_f]; phi1=DefExpr(c -> c.x1), phi2=DefExpr(c -> c.y1))
phase_trombone!(ring[tr_r]; phi1=DefExpr(c -> c.x2), phi2=DefExpr(c -> c.y2))
phase_trombone!(ring[tr_m]; phi1=DefExpr(c -> -c.x1 - c.x2), phi2=DefExpr(c -> -c.y1 - c.y2))
maximum(abs, value(tw_orbit.x[[bpm]]) .- reading) # The change of the BPM readings [m]
```

```{code-cell} python
reading = sb.value(tw_orbit.x[[bpm]])  # The BPM readings without the phase trombones
sb.phase_trombone(ring[tr_f], phi1=sb.DefExpr(lambda c: c.x1), phi2=sb.DefExpr(lambda c: c.y1))
sb.phase_trombone(ring[tr_r], phi1=sb.DefExpr(lambda c: c.x2), phi2=sb.DefExpr(lambda c: c.y2))
sb.phase_trombone(ring[tr_m], phi1=sb.DefExpr(lambda c: -c.x1 - c.x2),
                  phi2=sb.DefExpr(lambda c: -c.y1 - c.y2))
np.max(np.abs(sb.value(tw_orbit.x[[bpm]]) - reading))  # The change of the BPM readings [m]
```

If the orbit is corrected again later, the phase trombones are taken out first, in the same way.

## Exercises

1. **Zero orbit at the IP:** Use the 4 horizontal and 4 vertical correctors closest to the IP (on both
   sides) to make the orbit and its angle exactly 0 at the IP, without affecting the rest of the ring.

2. **Vertical orbit:** Misalign the quadrupoles vertically as well, and correct the vertical orbit
   with the vertical correctors.

3. **Tapering:** Instead of correcting the sawtooth with correctors, its effect on the optics can be
   removed by "tapering": scaling the strength of every magnet with the energy of the closed orbit at
   it, so that particles on the closed orbit see the design fields. Taper the elements of `ring_err`
   (without misalignments): multiply the `Kn0`, `Kn1` and `Kn2` of each element by $1 + p_z$ of the
   closed orbit at it. What happens to the orbit?

:::{only} julia
The lattice file of this chapter is
[`lattices/14-orbit-correction.jl`](https://github.com/bmad-sim/SciBmad.jl/tree/main/docs/src/ring-design/lattices/14-orbit-correction.jl).
:::
:::{only} python
The lattice file of this chapter is
[`lattices/ch14_orbit_correction.py`](https://github.com/bmad-sim/SciBmad.jl/tree/main/docs/src/ring-design/lattices/ch14_orbit_correction.py).
:::
