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

(rd.dispsup)=
# 2. Dispersion Suppressor

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

To connect an arc to the following straight section in a sextant, the dispersion in the straight
section must be suppressed to zero while minimally changing the beta functions. For a 90°
lattice, this can be approximately achieved by using two arc FODO cells with the bends at half
strength. For FODO cells with a different phase advance, dispersion suppressors can also be
constructed from the last two cells by changing their bend strengths. For 60° FODO cells, for
example, the first cell has zero bend strength, and the last has the full bend strength. To get
the dispersion and its derivative at the end of the dispersion suppressor exactly zero, the last
two quadrupoles can be minimally varied.

```{figure} figures/dispsup.svg
:width: 95%

Dispersion suppressor.
```

## Example: Forward Dispersion Suppressor

The procedure for creating the forward dispersion suppressor is outlined below. The reverse
dispersion suppressor is left as an exercise.

1. Start with the lattice of [chapter 1](fodo.md), which contains the arc cells with 90° phase
   advance. If you did not do the exercises of chapter 1, include the lattice file of that
   chapter:

```{code-cell} julia
include("lattices/01-fodo.jl");
```

```{code-cell} python
from lattices.ch01_fodo import *
```

2. Define a new half-strength bend `bh`, and the two special quadrupoles that will be varied to
   make the dispersion $\eta_x$ and its derivative $\eta_x'$ zero. Because these special
   quadrupoles are in the forward sextants (F), and are the first special quadrupoles (1), they are
   named `qff1` and `qdf1`:

```{code-cell} julia
@elements begin
  bh = SBend(L=Lc*(θ/2)/(2*sin(θ/4)), angle=θ/2, e1=θ/4, e2=θ/4) # Half-strength bend
  qff1 = Quadrupole(L=0.5, Kn1=0.3)
  qdf1 = Quadrupole(L=0.5, Kn1=-0.3)
end;
```

```{code-cell} python
with sb.elements():
    bh = sb.SBend(L=Lc*(θ/2)/(2*np.sin(θ/4)), angle=θ/2, e1=θ/4, e2=θ/4)  # Half-strength bend
    qff1 = sb.Quadrupole(L=0.5, Kn1=0.3)
    qdf1 = sb.Quadrupole(L=0.5, Kn1=-0.3)
```

3. Define the line of the figure above, with the half-strength bends and the special
   quadrupoles, and a `Beamline` with it:

```{code-cell} julia
dispsupf = [qf, d1, bh, d2, qd, d1, bh, d2, qff1, d1, bh, d2, qdf1, d1, bh, d2]
ds_f = Beamline(dispsupf, species_ref=electron, pc_ref=pc_ref)
```

```{code-cell} python
dispsupf = [qf, d1, bh, d2, qd, d1, bh, d2, qff1, d1, bh, d2, qdf1, d1, bh, d2]
ds_f = sb.Beamline(dispsupf, species_ref=electron, pc_ref=pc_ref)
ds_f
```

4. The dispersion suppressor is not periodic: it is an open line, and its lattice functions are
   computed from those at its start. Its start is the end of a periodic arc cell. In SciBmad,
   the lattice functions at a location are all contained in the normalizing map `a` there (see
   [Twiss](../twiss.md)), and a `TwissExpr` of an open line can start with the `a` of another
   `TwissExpr`. `tw_af` is the `TwissExpr` of the forward arc cell `cell_af` of chapter 1, so:

```{code-cell} julia
tw_dsf = TwissExpr(ds_f, a_initial=tw_af.a[end])
```

```{code-cell} python
tw_dsf = sb.TwissExpr(ds_f, a_initial=tw_af.a[-1])
tw_dsf
```

   The dispersion suppressor now always starts with the periodic lattice functions of the arc
   cell, also when the strengths of `qf` and `qd` change.

5. The targets are the dispersion `dx` and its derivative `dpx` (the derivative of `px` with
   respect to δ) at the end of the line. Before matching, they are:

```{code-cell} julia
value([tw_dsf.dx[end], tw_dsf.dpx[end]])
```

```{code-cell} python
sb.value([tw_dsf.dx[-1], tw_dsf.dpx[-1]])
```

   `value` computes the values of expressions. To see the dispersion through the line, compute
   it at every location, `tw_dsf.dx[:]`:

```{code-cell} julia
using CairoMakie
fig = Figure(size=(700, 300))
ax = Axis(fig[1, 1], xlabel="s [m]", ylabel="η [m]")
lines!(ax, value(tw_dsf.s[:]), value(tw_dsf.dx[:]), label="before")
fig
```

```{code-cell} python
from matplotlib.figure import Figure
fig = Figure(figsize=(7, 3), layout="constrained")
ax = fig.subplots()
ax.plot(sb.value(tw_dsf.s[:]), sb.value(tw_dsf.dx[:]), label="before")
ax.set_xlabel("s [m]")
ax.set_ylabel("η [m]")
fig
```

6. The match varies `qff1` and `qdf1`:

```{code-cell} julia
match!(vary=[@knob(qff1.Kn1), @knob(qdf1.Kn1)],
       targets=[tw_dsf.dx[end] == 0, tw_dsf.dpx[end] == 0])
```

```{code-cell} python
sb.match(vary=[sb.Knob(qff1, "Kn1"), sb.Knob(qdf1, "Kn1")],
         targets=[tw_dsf.dx[-1] == 0, tw_dsf.dpx[-1] == 0])
```

   The dispersion and its derivative are now zero to within the tolerance of the targets
   (`1e-8`), and the strengths of `qff1` and `qdf1` changed only by a little:

```{code-cell} julia
lines!(ax, value(tw_dsf.s[:]), value(tw_dsf.dx[:]), label="after")
axislegend(ax, position=:lb)
fig
```

```{code-cell} python
ax.plot(sb.value(tw_dsf.s[:]), sb.value(tw_dsf.dx[:]), label="after")
ax.legend(loc="lower left")
fig
```

## Exercises

1. **Reverse dispersion suppressor:** Construct the reverse dispersion suppressor, `dispsupr`, with
   the quadrupoles `qfr1` and `qdr1`, starting with the periodic lattice functions of the reverse
   arc cell (`tw_ar`). Match `qfr1` and `qdr1` for $\eta_x = \eta_x' = 0$ at the end. How do the two
   quadrupole strengths compare to those of the forward suppressor?

2. **Forward and reverse cells:** Check that your forward and reverse cells, which both start with
   focusing quadrupoles, have different periodic beta and alpha functions. Check also that both
   cells, for the same phase advance of 90°, have exactly the same quadrupole strengths. Explain
   how this can be the correct solution.

3. **Dispersion derivative:** The column `dpx` is the derivative of the momentum `px` with respect
   to δ (Bmad's `etap`), and not the derivative of the dispersion with respect to $s$ (Bmad's
   `deta_ds`). What is the difference between the two? Show that with the lattice being designed
   they are the same: compare `dpx` with the slope of `dx` in a drift, e.g. between the beginning
   and the end of `d1`. Hint: the phase space coordinates are defined in
   [Particle Phase Space Coordinates](phase-space.md).

4. **Second-order dispersion:** The suppressor makes the dispersion $\partial x/\partial\delta$ zero,
   but not the second-order dispersion $\partial^2 x/\partial\delta^2$, the column `dx_2`. Compute it at the
   end of the suppressor, as a `TwissExpr` quantity:
   {jl}`value(tw_dsf.dx_2[end])`{py}`sb.value(tw_dsf.dx_2[-1])`. Is it zero? Why
   not?

5. **Setting optimization parameters:** {jl}`match!`{py}`sb.match` stops after at most 50 iterations
   by default.
   Which keyword argument of {jl}`match!`{py}`sb.match` changes this? Hint: look at the
   documentation with
   {jl}`?match!`{py}`help(sb.match)`.

6. **Strength of bends:** [Warning! This is a difficult problem!] To a good approximation, a
   dispersion suppressor from an arc FODO line to a straight FODO line (where the straight FODO
   cells have the same layout as the arc FODO cells, except that the bends are replaced by drifts)
   can be constructed using two arc FODO cells, with the first cell having the bend strengths
   reduced by a factor $\alpha$ and the second cell with the bend strengths reduced by a factor
   $1 - \alpha$. In the case of 90° FODO cells, $\alpha$ is 0.5. Note: the approximation is only
   valid in the case of weak bends. Problem: Prove that this is the case, and show that for 60° FODO
   cells the value of $\alpha$ is zero.

The solution of exercise 1 is in the lattice file of this chapter,
[`lattices/02-dispersion-suppressor.jl`](https://github.com/bmad-sim/SciBmad.jl/tree/main/docs/src/ring-design/lattices/02-dispersion-suppressor.jl).
