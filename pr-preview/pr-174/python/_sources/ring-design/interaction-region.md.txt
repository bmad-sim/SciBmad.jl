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

(rd.ir)=
# 6. Low-Beta Interaction Region Insertion

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

## Introduction

In this chapter an interaction region (IR) with a low-beta interaction point (IP) will be
constructed, by modifying the quadrupoles of the 6 o'clock straight section as shown in the
figures A and B below.

```{figure} figures/ip6-layout.svg
:width: 90%

A) Layout before modification. B) Layout with the IR constructed. C) Beta functions in the IR.
```

```{figure} figures/irspecs.svg
:width: 90%

Interaction region dimensions. IPF: forward section before the IP. IPR: reverse section after the
IP.
```

The IR is divided into two parts, which are constructed separately as shown above. IPF is the
upstream "forward" section before the IP (labeled IP6), and IPR is the downstream "reverse" section
after IP6. At the IP, the beta functions are to be matched to $\beta_1^* = 0.6$ m and
$\beta_2^* = 0.06$ m and, as is standard for many IPs, the alpha functions $\alpha_1^*$ and
$\alpha_2^*$ to zero (a "waist"). The surrounding quadrupoles are adjusted to match these IP
parameters, while still matching to the periodic solution of the straight section at the ends of
the IR furthest from the IP.

Because this region is particularly sensitive, the match is done in two steps:

1. Match the forward line IPF from the straight section to the IP.
2. Match the reverse line IPR from the IP, with the lattice functions that IPF actually gives
   there, to the periodic lattice functions of the straight section.

(In the Bmad version of this tutorial, IPR was matched backwards from the straight section to the
IP, and then once more forward to correct the small mismatch at the IP. Because IPR starts with
the lattice functions of IPF here, the second match is already exact.)

For IR simulations, SciBmad has crab cavities (`CrabCavity`) to simulate crabbing with a nonzero
crossing angle. A discussion of this is beyond the scope of this tutorial.

## Example: Constructing and Matching the IR

1. Start with the ring of [chapter 5](ring.md):

```{code-cell} julia
include("lattices/05-ring.jl");
```

```{code-cell} python
from lattices.ch05_ring import *
```

2. Define the drifts and quadrupoles of the IR, with the dimensions of the figure above. The
   strengths are to be determined: start with the straight section strengths. Define also the
   IP itself, as a `Marker`. Markers have zero length and no effect on the orbit or spin.

```{code-cell} julia
@elements begin
  qef1 = Quadrupole(L=0.5, Kn1=k_ss)
  qef2 = Quadrupole(L=0.5, Kn1=-k_ss)
  def1 = Drift(L=20.46)
  qef3 = Quadrupole(L=1.6, Kn1=k_ss)
  def2 = Drift(L=3.76)
  qef4 = Quadrupole(L=1.2, Kn1=-k_ss)
  def3 = Drift(L=5.8)
  ip6 = Marker()
  der3 = Drift(L=5.3)
  qer4 = Quadrupole(L=1.8, Kn1=-k_ss)
  der2 = Drift(L=0.5)
  qer3 = Quadrupole(L=1.4, Kn1=k_ss)
  der1 = Drift(L=23.82)
  qer2 = Quadrupole(L=0.5, Kn1=k_ss)
  qer1 = Quadrupole(L=0.5, Kn1=-k_ss)
end;
```

```{code-cell} python
with sb.elements():
    qef1 = sb.Quadrupole(L=0.5, Kn1=k_ss)
    qef2 = sb.Quadrupole(L=0.5, Kn1=-k_ss)
    def1 = sb.Drift(L=20.46)
    qef3 = sb.Quadrupole(L=1.6, Kn1=k_ss)
    def2 = sb.Drift(L=3.76)
    qef4 = sb.Quadrupole(L=1.2, Kn1=-k_ss)
    def3 = sb.Drift(L=5.8)
    ip6 = sb.Marker()
    der3 = sb.Drift(L=5.3)
    qer4 = sb.Quadrupole(L=1.8, Kn1=-k_ss)
    der2 = sb.Drift(L=0.5)
    qer3 = sb.Quadrupole(L=1.4, Kn1=k_ss)
    der1 = sb.Drift(L=23.82)
    qer2 = sb.Quadrupole(L=0.5, Kn1=k_ss)
    qer1 = sb.Quadrupole(L=0.5, Kn1=-k_ss)
```

3. Define the upstream and downstream lines. The IP is at the end of `ipf`. (It is not repeated
   at the beginning of `ipr`, so that it is placed once in the ring.)

```{code-cell} julia
ipf = [qef1, d1, db, d2, qef2, d1, db, d2, def1, qef3, def2, qef4, def3, ip6]
ipr = [der3, qer4, der2, qer3, der1, qer2, d2, db, d1, qer1, d2, db, d1]
ir_f = Beamline(ipf, species_ref=electron, pc_ref=pc_ref)
ir_r = Beamline(ipr, species_ref=electron, pc_ref=pc_ref)
```

```{code-cell} python
ipf = [qef1, d1, db, d2, qef2, d1, db, d2, def1, qef3, def2, qef4, def3, ip6]
ipr = [der3, qer4, der2, qer3, der1, qer2, d2, db, d1, qer1, d2, db, d1]
ir_f = sb.Beamline(ipf, species_ref=electron, pc_ref=pc_ref)
ir_r = sb.Beamline(ipr, species_ref=electron, pc_ref=pc_ref)
```

4. IPF follows a straight section cell, so it starts with the periodic lattice functions of the
   forward straight section cell. IPR starts with the lattice functions at the end of IPF:

```{code-cell} julia
tw_ipf = TwissExpr(ir_f, a_initial=tw_ssf.a[end])
tw_ipr = TwissExpr(ir_r, a_initial=tw_ipf.a[end])
```

```{code-cell} python
tw_ipf = sb.TwissExpr(ir_f, a_initial=tw_ssf.a[-1])
tw_ipr = sb.TwissExpr(ir_r, a_initial=tw_ipf.a[-1])
```

   With the straight section strengths, the beta functions at the IP are far from the goal:

```{code-cell} julia
value([tw_ipf.beta1[end], tw_ipf.beta2[end]])
```

```{code-cell} python
sb.value([tw_ipf.beta1[-1], tw_ipf.beta2[-1]])
```

5. Match IPF. The beta functions here range from 0.06 m at the IP to several hundred meters in
   the final focus quadrupoles. A difference of 1 m from the goal means little at 600 m, but a lot
   at 0.06 m. It is therefore better to match the logarithm of β: then a relative error counts the
   same everywhere. Targets can be any expression of the quantities of `twiss`:

```{code-cell} julia
match!(vary=[@knob(qef1.Kn1), @knob(qef2.Kn1), @knob(qef3.Kn1), @knob(qef4.Kn1)],
       targets=[log(tw_ipf.beta1[end]) == log(0.6), log(tw_ipf.beta2[end]) == log(0.06),
                tw_ipf.alpha1[end] == 0, tw_ipf.alpha2[end] == 0])
```

```{code-cell} python
sb.match(vary=[sb.Knob(q, "Kn1") for q in (qef1, qef2, qef3, qef4)],
         targets=[np.log(tw_ipf.beta1[-1]) == np.log(0.6),
                  np.log(tw_ipf.beta2[-1]) == np.log(0.06),
                  tw_ipf.alpha1[-1] == 0, tw_ipf.alpha2[-1] == 0])
```

6. Match IPR to the periodic lattice functions of the reverse straight section cell, which
   follows it:

```{code-cell} julia
match!(vary=[@knob(qer1.Kn1), @knob(qer2.Kn1), @knob(qer3.Kn1), @knob(qer4.Kn1)],
       targets=[log(tw_ipr.beta1[end]) == log(tw_ssr.beta1[begin]),
                log(tw_ipr.beta2[end]) == log(tw_ssr.beta2[begin]),
                tw_ipr.alpha1[end] == tw_ssr.alpha1[begin],
                tw_ipr.alpha2[end] == tw_ssr.alpha2[begin]])
```

```{code-cell} python
sb.match(vary=[sb.Knob(q, "Kn1") for q in (qer1, qer2, qer3, qer4)],
         targets=[np.log(tw_ipr.beta1[-1]) == np.log(tw_ssr.beta1[sb.begin]),
                  np.log(tw_ipr.beta2[-1]) == np.log(tw_ssr.beta2[sb.begin]),
                  tw_ipr.alpha1[-1] == tw_ssr.alpha1[sb.begin],
                  tw_ipr.alpha2[-1] == tw_ssr.alpha2[sb.begin]])
```

7. Add the IR into the ring. It replaces the last three straight section cells of sextant 5 and
   the first three of sextant 7 (each half of the IR is as long as three cells):

```{code-cell} julia
sextant5 = [repeat(fodossf, 4); ss_to_arcf; repeat(fodoaf, 20); arc_to_ssf; fodossf; ipf]
sextant7 = [ipr; fodossr; ss_to_arcr; repeat(fodoar, 20); arc_to_ssr; repeat(fodossr, 4)]

ring = Beamline([sextant1; sextant3; sextant5; sextant7; sextant9; sextant11],
                species_ref=electron, pc_ref=pc_ref)
tw = twiss(ring)
tw.beta1[ip6], tw.beta2[ip6], tw.alpha1[ip6], tw.alpha2[ip6]
```

```{code-cell} python
sextant5 = fodossf*4 + ss_to_arcf + fodoaf*20 + arc_to_ssf + fodossf + ipf
sextant7 = ipr + fodossr + ss_to_arcr + fodoar*20 + arc_to_ssr + fodossr*4

ring = sb.Beamline(sextant1 + sextant3 + sextant5 + sextant7 + sextant9 + sextant11,
                   species_ref=electron, pc_ref=pc_ref)
tw = sb.twiss(ring)
tw.beta1[ip6], tw.beta2[ip6], tw.alpha1[ip6], tw.alpha2[ip6]
```

   The periodic lattice functions of the whole ring have the IP values: the IR is matched into
   the rest of the ring. The beta functions around the IP:

```{code-cell} julia
using CairoMakie
s0 = tw.s[ip6]
fig = Figure(size=(800, 350))
ax = Axis(fig[1, 1], xlabel="s - s(IP6) [m]", ylabel="β [m]")
rows = (s0 - 120, s0 + 120) # An s-range around the IP
lines!(ax, tw.s[rows] .- s0, tw.beta1[rows], label="β₁")
lines!(ax, tw.s[rows] .- s0, tw.beta2[rows], label="β₂")
axislegend(ax, position=:ct)
fig
```

```{code-cell} python
from matplotlib.figure import Figure
s0 = tw.s[ip6]
fig = Figure(figsize=(8, 3.5), layout="constrained")
ax = fig.subplots()
rows = slice(s0 - 120, s0 + 120)  # An s-range around the IP
ax.plot(tw.s[rows] - s0, tw.beta1[rows], label="β₁")
ax.plot(tw.s[rows] - s0, tw.beta2[rows], label="β₂")
ax.set_xlabel("s - s(IP6) [m]")
ax.set_ylabel("β [m]")
ax.legend(loc="upper center")
fig
```

## Exercises

1. **Initial lattice functions for a reflected line:** A line can be matched backwards, from its
   end to its start, by computing the optics of the reflected line `reverse(ipr)`. (This is only
   correct for elements that are symmetric front to back, such as drifts and quadrupoles.) How must
   the initial Twiss parameters $\beta$ and $\alpha$ be changed for the reflected line? Hint: the
   Twiss parameters are the same as computed in earlier chapters.

2. **Matching backwards:** Match IPR backwards: start the reflected line with the periodic lattice
   functions of the reverse straight section cell (given with `initial`, as numbers of a `twiss`
   table), and match `qer1` to `qer4` so that $\beta_1 = 0.6$ m, $\beta_2 = 0.06$ m and
   $\alpha_1 = \alpha_2 = 0$ at its end, the IP. Start with the strengths found above, each
   multiplied by 1.01. Do you get the same strengths back? What happens if the initial alpha
   functions are not changed for the reflected line? (Starting with the straight section strengths,
   this match stops at a point that is not a solution.)

3. **Logarithms:** Start again with the straight section strengths, and match IPR with targets for
   β instead of log β. What happens? Several matches in this tutorial stop at a point that is not a
   solution, because small changes of the knobs cannot improve the merit function there. How the
   targets are written decides which points these are.

4. **No step size:** In Bmad, the derivatives for the optimizer are computed with finite differences,
   with a step size for each variable, and a step size of $10^{-4}$ instead of $10^{-6}$ for the
   quadrupole strengths keeps this match from converging. Why does the step size matter? Why does
   SciBmad not need one?

The lattice file of this chapter is
[`lattices/06-interaction-region.jl`](https://github.com/bmad-sim/SciBmad.jl/tree/main/docs/src/ring-design/lattices/06-interaction-region.jl).
