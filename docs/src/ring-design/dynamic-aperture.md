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

(rd.da)=
# 12. Dynamic Aperture

```{code-cell} julia
:tags: [remove-cell]
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

The dynamic aperture (DA) is a measure of the stable phase space in an accelerator over a specified
number of turns. Particles outside of the DA will be driven to large amplitudes, where they are
lost. In the following sections, strategies for increasing the DA will be shown for our ring.

## Calculating the Dynamic Aperture

SciBmad's `dynamic_aperture` tracks a polar grid of particles in the $x$-$y$ plane, in units of the
beam sizes $\sigma_x$ and $\sigma_y$ at the start of the ring, for several energy deviations δ. All
particles are tracked at once (in parallel on a GPU, if one is used). The boundary of the DA along
each angle is the first particle of the grid that is lost. (Bmad instead searches along each angle
for the boundary between stable and unstable starting points.) See
[Dynamic Aperture](../dynamic-aperture.md) for all of its options.

Start with the ring with the RF cavities of [chapter 9](rf.md):

```{code-cell} julia
include("lattices/09-rf.jl");
```

The DA for the emittances $\varepsilon_1 = 2\cdot10^{-8}$ m and $\varepsilon_2 = 10^{-8}$ m, with
200 turns, for δ from 0 to 1%:

```{code-cell} julia
deltas = 0:0.002:0.01
da_settings = (n_turns=200, emit_1=2e-8, emit_2=1e-8, n_r=21, n_theta=9, max_sig_x=30,
               max_sig_y=60, deltas=deltas, verbose=false)
x0, y0 = dynamic_aperture(ring; da_settings...);
```

`x0[j, i]` and `y0[j, i]` are the boundary along the angle `j` for `deltas[i]`, in units of the beam
sizes. A column of zeros means that already the particle on the closed orbit is lost. Radiation
is left out here (see exercise 4). DA
calculations can be slow, so it is sometimes advisable to reduce the accuracy (here 1.5σ in $x$ and
3σ in $y$), the number of turns, or the number of angles to get faster results between
optimizations, and then come back with a more accurate run to verify.

```{code-cell} julia
using CairoMakie
fig = Figure(size=(700, 380))
ax = Axis(fig[1, 1], xlabel="x/σx", ylabel="y/σy", title="No sextupoles", limits=(-30, 30, 0, 60))
for i in eachindex(deltas)
  lines!(ax, x0[:, i], y0[:, i], label="δ = $(100*deltas[i])%")
end
Legend(fig[1, 2], ax)
fig
```

Without sextupoles, the chromaticities of the ring are large (about −80 and −90), and particles with
an energy deviation have very different tunes. At δ = 0.6%, the horizontal tune changes by about
−0.5, and already the particle on the closed orbit is lost.

## Example: Adding Sextupoles

Sextupoles are commonly used for DA optimization. They correct the chromaticity, and "harmonic"
sextupoles minimize resonance driving terms. Sextupoles can be added to the lattice by following the
steps below.

1. We need to make room for the sextupoles, so the drift `d2` is split into two smaller drifts.
   Harmonic sextupoles are defined for the straight sections, one family in each plane. They are
   turned off here:

```{code-cell} julia
@elements begin
  d2s1 = Drift(L=0.501)
  d2s2 = Drift(L=0.156)
  sfss = Sextupole(L=0.584, Kn2=0.0)
  sdss = Sextupole(L=0.584, Kn2=0.0)
end
fodossf = [qfss, d1, db, d2s1, sdss, d2s2, qdss, d1, db, d2s1, sfss, d2s2]
fodossr = [qfss, d2s2, sfss, d2s1, db, d1, qdss, d2s2, sdss, d2s1, db, d1];
```

2. Define the chromatic sextupoles: these control the chromaticity, and therefore have to be placed
   where there is dispersion, in the arcs. Two families per plane in each arc will be used: the
   sextupoles `sf1` and `sf2` near the focusing quadrupoles alternate from one FODO cell to the
   next, and so do `sd1` and `sd2` near the defocusing quadrupoles. Their strengths are deferred
   expressions of variables in the context of the ring (see
   [Deferred Expressions and Contexts](control.md)): `k2f` and `k2d` for all focusing and all
   defocusing arc sextupoles, and the vectors `xf` and `xd`, which make the two families of each
   arc stronger and weaker by the same amount. They are used in the next section. The arcs are
   built in a loop: arc `i` is that of sextant `2i-1`.

```{code-cell} julia
arcs = Vector{LatElement}[]
for i in 1:6
  sf1 = Sextupole(name="sf1_$(2i-1)", L=0.584, Kn2=DefExpr(c -> c.k2f + c.xf[i]))
  sf2 = Sextupole(name="sf2_$(2i-1)", L=0.584, Kn2=DefExpr(c -> c.k2f - c.xf[i]))
  sd1 = Sextupole(name="sd1_$(2i-1)", L=0.584, Kn2=DefExpr(c -> c.k2d + c.xd[i]))
  sd2 = Sextupole(name="sd2_$(2i-1)", L=0.584, Kn2=DefExpr(c -> c.k2d - c.xd[i]))
  if isodd(i) # Forward sextants 1, 5, 9
    cells = [qf, d1, b, d2s1, sd1, d2s2, qd, d1, b, d2s1, sf1, d2s2,
             qf, d1, b, d2s1, sd2, d2s2, qd, d1, b, d2s1, sf2, d2s2]
  else        # Reverse sextants 3, 7, 11
    cells = [qf, d2s2, sf1, d2s1, b, d1, qd, d2s2, sd1, d2s1, b, d1,
             qf, d2s2, sf2, d2s1, b, d1, qd, d2s2, sd2, d2s1, b, d1]
  end
  push!(arcs, repeat(cells, 10))
end
```

3. Finally, the sextants and the ring are built with the new cells. Also added are markers for the
   phase trombones and for the end of arc 7, which are used in the next section:

```{code-cell} julia
@elements begin
  tr_m = Marker()
  tr_f = Marker()
  tr_r = Marker()
  end_7 = Marker()
end

sextant1 = [tr_m; repeat(fodossf, 4); ss_to_arcf; arcs[1]; arc_to_ssf_2; repeat(fodossf_2, 4)]
sextant3 = [repeat(fodossr_2, 4); ss_to_arcr_2; arcs[2]; arc_to_ssr; repeat(fodossr, 4)]
sextant5 = [repeat(fodossf, 4); ss_to_arcf; arcs[3]; arc_to_ssf; fodossf; tr_f; ipf]
sextant7 = [ipr; tr_r; fodossr; ss_to_arcr; arcs[4]; end_7; arc_to_ssr; repeat(fodossr, 4)]
sextant9 = [repeat(fodossf, 4); ss_to_arcf; arcs[5]; arc_to_ssf; repeat(fodossf, 2);
            repeat(fodorf, 2)]
sextant11 = [repeat(fodorf, 2); repeat(fodossr, 2); ss_to_arcr; arcs[6]; arc_to_ssr;
             repeat(fodossr, 4)]

ring = Beamline([sextant1; sextant3; sextant5; sextant7; sextant9; sextant11],
                species_ref=electron, pc_ref=pc_ref)

ring.context.k2f = 1.0
ring.context.k2d = -2.0
ring.context.xf = zeros(6)
ring.context.xd = zeros(6);
```

## Example: Chromaticity Correction

The first step in chromatic correction is to use the chromatic sextupoles to correct the linear
chromaticity. There is a chromaticity in each plane, so two knobs are needed: the variables `k2f`
and `k2d` of the context, which vary all focusing and all defocusing arc sextupoles together.
Instead of setting the chromaticity to zero, it is usually set to a small positive number above
transition (or a small negative number below transition), to avoid the head-tail instability. Here
it is set to 1 in both planes.

The chromaticities are only defined without synchrotron motion, so the `TwissExpr` is computed with
the RF cavities off:

```{code-cell} julia
tw_chrom = TwissExpr(ring, rf_on=false)
match!(vary=[@knob(ring.context.k2f), @knob(ring.context.k2d)],
       targets=[tw_chrom.dq1 == 1, tw_chrom.dq2 == 1])
```

The DA now:

```{code-cell} julia
x1, y1 = dynamic_aperture(ring; da_settings...);
fig = Figure(size=(700, 380))
ax = Axis(fig[1, 1], xlabel="x/σx", ylabel="y/σy", title="Chromaticity corrected", limits=(-30, 30, 0, 60))
for i in eachindex(deltas)
  lines!(ax, x1[:, i], y1[:, i], label="δ = $(100*deltas[i])%")
end
Legend(fig[1, 2], ax)
fig
```

### Sextupoles of a family cancel their resonance driving terms

Sextupoles also drive resonances: their fields are nonlinear. The first-order resonance driving
terms of a sextupole, e.g. $h_{3000}$ (see [Twiss](#twiss.h)), are proportional to its integrated
strength $K_2L$ and depend on the phase advance $\varphi_1$ from the point of observation to the
sextupole as $e^{-3i\varphi_1}$:

$$
h_{3000} = \frac{K_2 L\, \beta_1^{3/2}}{12\sqrt{2}}\, e^{-3i\varphi_1}
$$

In our arcs, the sextupoles of one family are placed every second cell, and are therefore 180° apart
in both planes. Their contributions $e^{-3i\varphi_1}$ have opposite signs, and cancel pairwise. The
contributions of the focusing sextupoles of arc 1 (both families, which have the same strength
here) to $h_{3000}$ at the beginning of the ring, added one after the other in their order along
the arc:

```{code-cell} julia
tw = twiss(ring, rf_on=false, cols=["beta1", "phi1"])
families = ["sf1_1", "sf2_1"]
s = vcat([tw.s[name] for name in families]...)
β = vcat([tw.beta1[name] for name in families]...)
φ = 2pi .* vcat([tw.phi1[name] for name in families]...) # Phase advance from the start [rad]
K2L = ring["sf1_1"][1].Kn2*0.584
h = K2L .* β.^1.5 ./ (12*sqrt(2)) .* cis.(-3 .* φ)
h = h[sortperm(s)] # In the order of the sextupoles along the arc
total = cumsum([0; h])
fig = Figure(size=(500, 450))
ax = Axis(fig[1, 1], xlabel="Re h₃₀₀₀", ylabel="Im h₃₀₀₀", aspect=DataAspect())
lines!(ax, real(total), imag(total))
scatter!(ax, real(total), imag(total))
fig
```

Each sextupole moves the sum by the same amount (the beta functions at all of them are equal), in a
direction that turns by $3 \cdot 90° = 270°$ from one cell to the next. After four cells the sum is
back at zero: the next sextupole of the same family, two cells later, takes back what the first one
added. The 20 sextupoles add up to zero. Their contributions to the chromaticity, which depend on
the dispersion and not on the phase, add up instead. This is one reason why the phase advance of
90° per cell was chosen.

The resonance driving terms of the whole ring are small for the same reason. Compare $|h_{3000}|$ at
the beginning of the ring with the contribution of one sextupole:

```{code-cell} julia
twh = twiss(ring, rf_on=false, cols=["h3000"], at=[1])
abs(twh.h3000[1]), ring["sf1_1"][1].Kn2*0.584*tw.beta1["sf1_1"][1]^1.5/(12*sqrt(2))
```

## Advanced Chromatic Correction

In addition to the linear chromaticity, several other quantities that affect the DA can be
corrected. One of them is the chromatic beta beat, described by the Montague W function:

$$
W = \sqrt{A^2 + B^2}, \qquad
A = \frac{\partial\alpha}{\partial p_z} - \frac{\alpha}{\beta}\frac{\partial\beta}{\partial p_z},
\qquad B = \frac{1}{\beta}\frac{\partial\beta}{\partial p_z}
$$

$W$ is the amplitude of the linear energy-dependent beta beat. Like the linear chromaticity, it can
be corrected with sextupoles. The W functions of our ring (the columns `w1` and `w2`; $A$ and $B$
are `w1a` and `w1b`):

```{code-cell} julia
tww = twiss(ring, rf_on=false, chrom=2, cols=["w1", "w2"])
fig = Figure(size=(800, 300))
ax = Axis(fig[1, 1], xlabel="s [m]", ylabel="W", title="Chromaticity corrected")
lines!(ax, tww.s, tww.w1, label="W₁")
lines!(ax, tww.s, tww.w2, label="W₂")
axislegend(ax, position=:lt)
fig
```

The W functions receive a large contribution from the IR, where the beta functions are large,
which then travels around the ring. The correction is more complex than for the chromaticity, as
the phase advance between the sextupoles and the quadrupoles that create the W function is
important. Since the beta beat goes as twice the betatron phase advance, sextupoles that are 90°
apart work against each other. The solution is the two families per plane in each arc: varying them
in opposite directions (the variables `xf` and `xd`) leaves the chromaticity unchanged, while their
effects on the W function do not cancel.

1. Without proper phasing between the arcs, the sextupoles may not be efficient for the correction.
   To change the phases between the arcs and the IR, phase trombones are used: elements that change
   the phase advances without changing the Twiss parameters (see
   [Definition Tools](../def-tools.md)). We need three: two surrounding the IR, and one at the
   beginning of the ring, whose phase advance is the negative of their sum, so that the tunes stay
   the same. Their phase advances are deferred expressions of the variables `x1`, `x2`, `y1`, `y2`.
   They are built from the optics without synchrotron motion (`rf_on=false`), in which the W
   function is computed:

```{code-cell} julia
ring.context.x1 = 0.0
ring.context.x2 = 0.0
ring.context.y1 = 0.0
ring.context.y2 = 0.0
phase_trombone!(ring[tr_f]; phi1=DefExpr(c -> c.x1), phi2=DefExpr(c -> c.y1), rf_on=false)
phase_trombone!(ring[tr_r]; phi1=DefExpr(c -> c.x2), phi2=DefExpr(c -> c.y2), rf_on=false)
phase_trombone!(ring[tr_m]; phi1=DefExpr(c -> -c.x1 - c.x2), phi2=DefExpr(c -> -c.y1 - c.y2),
                rf_on=false);
```

2. The W function is often corrected by making it zero at specific locations in the ring. We make it
   zero at the IP and at the end of arc 7 (`end_7`), with the families of the arcs 5 and 7, on both
   sides of the IP, and the phase trombones. $W$ is zero when both $A$ and $B$ are, so these are the
   targets: 8 targets, and 8 knobs.

```{code-cell} julia
match!(vary=[@knob(ring.context.xf[3]), @knob(ring.context.xd[3]),
             @knob(ring.context.xf[4]), @knob(ring.context.xd[4]),
             @knob(ring.context.x1), @knob(ring.context.x2),
             @knob(ring.context.y1), @knob(ring.context.y2)],
       targets=[tw_chrom.w1a[ip6] == 0, tw_chrom.w1b[ip6] == 0,
                tw_chrom.w2a[ip6] == 0, tw_chrom.w2b[ip6] == 0,
                tw_chrom.w1a[end_7] == 0, tw_chrom.w1b[end_7] == 0,
                tw_chrom.w2a[end_7] == 0, tw_chrom.w2b[end_7] == 0],
       solver=:gauss_newton)
```

The W function is now corrected in the IR and over one arc:

```{code-cell} julia
tww = twiss(ring, rf_on=false, chrom=2, cols=["w1", "w2"])
fig = Figure(size=(800, 300))
ax = Axis(fig[1, 1], xlabel="s [m]", ylabel="W", title="W corrected")
lines!(ax, tww.s, tww.w1, label="W₁")
lines!(ax, tww.s, tww.w2, label="W₂")
axislegend(ax, position=:lt)
fig
```

and the DA is much larger for particles with an energy deviation:

```{code-cell} julia
x2, y2 = dynamic_aperture(ring; da_settings...);
fig = Figure(size=(700, 380))
ax = Axis(fig[1, 1], xlabel="x/σx", ylabel="y/σy", title="Chromaticity and W corrected", limits=(-30, 30, 0, 60))
for i in eachindex(deltas)
  lines!(ax, x2[:, i], y2[:, i], label="δ = $(100*deltas[i])%")
end
Legend(fig[1, 2], ax)
fig
```

## Exercises

1. **Other phase advances:** With the phase advance $\mu$ per cell in both planes, the $k$-th
   sextupole of a family (every second cell) has the phase $2k\mu$, and its contribution to
   $h_{3000}$ the factor $e^{-6ik\mu}$. For which $\mu$ do the contributions of the ten sextupoles of a
   family add up to zero? For which do they add up instead? Which values of $\mu$ also make the
   contributions to $h_{2100}$ and $h_{1020}$ add up to zero (see [Twiss](#twiss.h))?

2. **Phases around the IP:** Instead of the phase trombones, the phase advances from the arcs to
   the IP can be changed with quadrupoles. For example, the phase advance from the end of arc 5 to
   the IP can be set with the quadrupoles of the IR (`qef1` to `qef4`) and the last three quadrupoles
   of the matching section before it, if these get their own copies in sextant 5 (as the tune cell
   did in [chapter 7](tune-cell.md)), while keeping the Twiss parameters at the IP. Set up this match.
   Note that the W correction is then slightly disrupted; this sometimes requires an iterative
   process.

3. **Resonance driving terms:** The harmonic sextupoles `sfss` and `sdss` in the straight sections
   are off. Use them, with deferred expressions for their strengths, to make $|h_{3000}|$ and
   $|h_{2100}|$ at the IP smaller, while keeping the chromaticities: e.g. with `minimize` (see
   [Matching and Optimization](../optimize.md)).

4. **Radiation:** Turn on radiation damping in all elements (`ele.rad_damping = true` for every
   `ele` in `ring.line`), and compute the DA again. How does it change? Why must radiation damping
   be off for the chromaticity match?

The lattice file of this chapter is
[`lattices/12-sextupoles.jl`](https://github.com/bmad-sim/SciBmad.jl/tree/main/docs/src/ring-design/lattices/12-sextupoles.jl).
