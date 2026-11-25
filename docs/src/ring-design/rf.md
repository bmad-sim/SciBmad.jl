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

(rd.rf)=
# 9. RF Cavities

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

(rd.refenergy)=
## Reference Energy

Every beamline has a reference particle (`species_ref`), a reference energy (`E_ref`, the total
energy in eV), and a reference momentum (`pc_ref`, in eV). The three are interrelated, so knowing
the reference particle and either the reference momentum or the reference energy, the third can
be calculated. The magnetic rigidity `p_over_q_ref` (in T·m) is another way to give the reference
momentum. The reference momentum is used to normalize the phase space momenta, as well as the
normalized strengths of the elements.

For example, for a quadrupole, the field gradient `Bn1` (in T/m) can be used to give the linear
field gradient, and `Kn1` (in 1/m²) is the field gradient normalized by the magnetic rigidity of the
reference particle: `Kn1 = Bn1/p_over_q_ref`. Whichever of the two was set last is stored, and the
other is computed from it, in the beamline the element is placed in:

```{code-cell} julia
using SciBmad

q1 = Quadrupole(L=0.1, Kn1=0.14)
q2 = Quadrupole(L=0.1)
lat = Beamline([q1, q2], species_ref=Species("proton"), pc_ref=1e9)
lat[q2].Bn1 = lat.p_over_q_ref*lat[q1].Kn1 # The same field gradient as q1 [T/m]
lat[q1].Bn1, lat[q2].Kn1
```

```{code-cell} python
import numpy as np
import scibmad as sb

q1 = sb.Quadrupole(L=0.1, Kn1=0.14)
q2 = sb.Quadrupole(L=0.1)
lat = sb.Beamline([q1, q2], species_ref=sb.Species("proton"), pc_ref=1e9)
lat[q2].Bn1 = lat.p_over_q_ref*lat[q1].Kn1  # The same field gradient as q1 [T/m]
lat[q1].Bn1, lat[q2].Kn1
```

If the reference momentum of `lat` is changed, the field gradient of `q1` changes (`Kn1` is
stored), and the normalized strength of `q2` changes (`Bn1` is stored). See
[Defining a Beamline](../beamline.md) for more.

In a ring, the reference energy is the same everywhere. The RF cavities of a ring (`RFCavity`)
change the energy of the particles that arrive at another time than the reference particle, but
not the reference energy: the reference particle arrives at the zero crossing of the RF voltage,
and keeps its energy. (These are the `rfcavity` elements of Bmad.)

In a linac, the cavities accelerate the reference particle, and the reference energy changes from
one cavity to the next. In SciBmad, such a cavity is an `RFCavity` with the change of the reference
energy `dE_ref`, tracked with the `SaganCavity` tracking method (see
[SaganCavity Tracking](../sagancavity-tracking.md)). (These are the `lcavity` elements of Bmad.) A
change of the reference energy starts a new beamline: an element with `dE_ref` is the first element
of a `Beamline`, and the beamlines of a linac are joined in a `Branch`. This tutorial only needs
the cavities of a ring.

The phase of an RF cavity is set by `phi0` and `zero_phase` (see [RFParams](#rf.params)). A particle
that arrives a time $\Delta t$ after the reference particle gains the energy
$|q| V \cos(\phi_0 + \phi_\mathrm{zero} + \omega \Delta t)$. With `zero_phase =
PhaseRef.AboveTransition` and `phi0 = 0`, the reference particle arrives at the zero crossing, and a
particle arriving late ($z < 0$) loses energy. This is what keeps the longitudinal motion stable
above transition, where particles with more energy take longer around the ring. For example, an
electron 1 mm ahead of the reference particle in a 1 MV, 500 MHz cavity:

```{code-cell} julia
cav = RFCavity(L=0.5, voltage=1e6, rf_frequency=500e6, zero_phase=PhaseRef.AboveTransition)
lat = Beamline([cav], species_ref=Species("electron"), E_ref=18e9)
res = track(lat, v0=[0 0 0 0 1e-3 0])
res.pz[turn=1] # ≈ (1 MeV/18 GeV) sin(2π (1 mm)/(0.6 m))
```

```{code-cell} python
cav = sb.RFCavity(L=0.5, voltage=1e6, rf_frequency=500e6, zero_phase=sb.PhaseRef.AboveTransition)
lat = sb.Beamline([cav], species_ref=sb.Species("electron"), E_ref=18e9)
res = sb.track(lat, v0=[[0, 0, 0, 0, 1e-3, 0]])
res.pz(turn=1)  # ≈ (1 MeV/18 GeV) sin(2π (1 mm)/(0.6 m))
```

## Adding RF Cavities

In a real machine, electrons emit synchrotron radiation as they bend around the ring, so the energy
needs to be replenished with an RF system. Let's put RF cavities into the 10 o'clock straight
section of our ring. We will design a special FODO cell with RF cavities, `fodorf`, and replace the
4 FODO cells of the 10 o'clock straight section with it.

In the ESR, there are two RF cavities between each pair of quadrupoles, with equal drift spaces
between the elements. For our ring, the cavities are 3.41 m long, as shown in the figure.

```{figure} figures/rf.svg
:width: 90%

FODO cell with RF cavities.
```

## Example: Constructing the FODO Cell with RF Cavities

1. Start with the ring with the tune cell of [chapter 7](tune-cell.md):

```{code-cell} julia
include("lattices/07-tune-cell.jl");
```

```{code-cell} python
from lattices.ch07_tune_cell import *
```

2. Define the RF cavities and the drifts. The drift length is computed from the length of the
   straight section cell. The harmonic number `harmon` is the RF frequency in units of the
   revolution frequency (the RF frequency is then computed from the length of the ring). Start
   with the voltage of the ESR, 68 MV divided among 18 cavities:

```{code-cell} julia
@elements begin
  rf0 = RFCavity(L=3.41, harmon=7560, voltage=68e6/18,
                 zero_phase=PhaseRef.AboveTransition)
  drf = Drift(L=((1.241 + 5.855 + 0.609) - 2*3.41)/3)
end;
```

```{code-cell} python
with sb.elements():
    rf0 = sb.RFCavity(L=3.41, harmon=7560, voltage=68e6/18,
                      zero_phase=sb.PhaseRef.AboveTransition)
    drf = sb.Drift(L=((1.241 + 5.855 + 0.609) - 2*3.41)/3)
```

3. Define the `fodorf` cell:

```{code-cell} julia
fodorf = [qfss, drf, rf0, drf, rf0, drf, qdss, drf, rf0, drf, rf0, drf];
```

```{code-cell} python
fodorf = [qfss, drf, rf0, drf, rf0, drf, qdss, drf, rf0, drf, rf0, drf]
```

4. Replace the 4 FODO cells of the 10 o'clock section by `fodorf` cells, in the sextants 9 and 11:

```{code-cell} julia
sextant9 = [repeat(fodossf, 4); ss_to_arcf; repeat(fodoaf, 20); arc_to_ssf;
            repeat(fodossf, 2); repeat(fodorf, 2)]
sextant11 = [repeat(fodorf, 2); repeat(fodossr, 2); ss_to_arcr; repeat(fodoar, 20);
             arc_to_ssr; repeat(fodossr, 4)]

ring = Beamline([sextant1; sextant3; sextant5; sextant7; sextant9; sextant11],
                species_ref=electron, pc_ref=pc_ref)
```

```{code-cell} python
sextant9 = fodossf*4 + ss_to_arcf + fodoaf*20 + arc_to_ssf + fodossf*2 + fodorf*2
sextant11 = fodorf*2 + fodossr*2 + ss_to_arcr + fodoar*20 + arc_to_ssr + fodossr*4

ring = sb.Beamline(sextant1 + sextant3 + sextant5 + sextant7 + sextant9 + sextant11,
                   species_ref=electron, pc_ref=pc_ref)
```

With RF cavities, the longitudinal motion oscillates too, and `twiss` computes it as the third
normal mode: the synchrotron tune is `q3`.

```{code-cell} julia
twiss(ring, at=[])
```

```{code-cell} python
sb.twiss(ring, at=[])
```

The synchrotron tune is negative above transition, because the longitudinal motion turns the other
way in the $z$-$p_z$ plane than below transition (see [Twiss](#twiss.rf)).

5. Set the synchrotron tune to 0.05 by varying the voltage of the cavities. All 16 cavities are
   placements of `rf0`, so one knob varies them all:

```{code-cell} julia
tw_ring = TwissExpr(ring)
match!(vary=[@knob(rf0.voltage)], targets=[tw_ring.q3 == -0.05])
```

```{code-cell} python
tw_ring = sb.TwissExpr(ring)
sb.match(vary=[sb.Knob(rf0, "voltage")], targets=[tw_ring.q3 == -0.05])
```

The momentum compaction $\alpha_c$, the relative change of the length of the closed orbit with
$\delta$, is defined without synchrotron motion: compute it with the cavities turned off, with
`rf_on=false`.

```{code-cell} julia
tw = twiss(ring, rf_on=false, at=[])
tw.alphac
```

```{code-cell} python
tw = sb.twiss(ring, rf_on=False, at=[])
tw.alphac
```

## Exercises

1. **Synchrotron tune:** For small oscillations, the synchrotron tune is
   $Q_s = \sqrt{h\,\eta_c\, e V_\mathrm{tot} / (2\pi \beta^2 E)}$, where $h$ is the harmonic number,
   $\eta_c$ the slip factor (`etac`), $V_\mathrm{tot}$ the total voltage and $E$ the energy. Compute
   it with the matched voltage, and compare with `q3`.

2. **Velocity and z:** For a particle slower than light, $z = -\beta c\,(t - t_\mathrm{ref})$ depends
   on its velocity. Make a beamline of a short cavity (`L = 1e-6`) for protons with the reference
   momentum `pc_ref = 1e9`, with the voltage `1e9` and `zero_phase = PhaseRef.Accelerating` (on
   crest), and track a particle that starts with `z = 1e-3`. Calculate its velocity before and after
   the cavity, and verify that the change of $z$ is consistent with what `track` calculates.

3. **RF phase:** Change `phi0` of the cavity of the previous exercise. How does the energy gain
   change? Does the reference energy change?

The lattice file of this chapter is
[`lattices/09-rf.jl`](https://github.com/bmad-sim/SciBmad.jl/tree/main/docs/src/ring-design/lattices/09-rf.jl).
