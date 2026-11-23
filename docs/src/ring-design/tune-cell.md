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

(rd.tunecell)=
# 7. Tune Cell

```{code-cell} julia
:tags: [remove-cell]
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

## Introduction

In order to adjust the tunes of the ring, the phase advance per FODO cell in one of the straight
sections can be adjusted, while ensuring that it is still matched to the periodic solution in each
of the surrounding arcs. We'll put this "tune cell" in the 2 o'clock straight section, between the
sextants 1 and 3, and name its quadrupoles by appending `_2` to the names of the quadrupoles they
replace.

```{figure} figures/tunecell.svg
:width: 100%

The tune cell in the 2 o'clock straight section.
```

Required for the match is:

1. Setting the phase advances in the `fodossf_2`/`fodossr_2` cells of the straight section for the
   desired tunes $Q_1 = 54.08$ and $Q_2 = 54.14$ (2 constraints).
2. Matching from the dispersion suppressor to the new periodic Twiss parameters of the
   `fodossf_2`/`fodossr_2` cells (4 constraints).
3. Matching the `fodossf_2`/`fodossr_2` cells to the dispersion creator (4 more constraints, but
   these are already met by the mirror symmetry).

These 6 constraints can be met using the four quadrupoles of the forward matching section
(`qff2_2`, `qdf2_2`, `qff3_2`, and `qdf3_2`) and the two quadrupoles of the straight section cells
(`qfss_2` and `qdss_2`).

## Example: Matching the Tune Cell

1. Start with the ring with the IR of [chapter 6](interaction-region.md):

```{code-cell} julia
include("lattices/06-interaction-region.jl");
twiss(ring, at=[]) # The tunes before
```

2. Define the new quadrupoles, starting with the strengths of those they replace:

```{code-cell} julia
@elements begin
  qfss_2 = Quadrupole(L=0.5, Kn1=qfss.Kn1)
  qdss_2 = Quadrupole(L=0.5, Kn1=qdss.Kn1)
  qff2_2 = Quadrupole(L=0.5, Kn1=qff2.Kn1)
  qdf2_2 = Quadrupole(L=0.5, Kn1=qdf2.Kn1)
  qff3_2 = Quadrupole(L=0.5, Kn1=qff3.Kn1)
  qdf3_2 = Quadrupole(L=0.5, Kn1=qdf3.Kn1)
end;
```

3. Define the lines of the tune cell, and put them into the sextants 1 and 3. The dispersion
   suppressor and the dispersion creator keep their quadrupoles:

```{code-cell} julia
fodossf_2 = [qfss_2, d1, db, d2, qdss_2, d1, db, d2]
fodossr_2 = [qfss_2, d2, db, d1, qdss_2, d2, db, d1]
mssf_2 = [qff2_2, d1, db, d2, qdf2_2, d1, db, d2, qff3_2, d1, db, d2, qdf3_2, d1, db, d2]
mdcr_2 = [qfss_2, d2, db, d1, qdf3_2, d2, db, d1, qff3_2, d2, db, d1, qdf2_2, d2, db, d1]
dispcrer_2 = [qff2_2, d2, bh, d1, qdf1, d2, bh, d1, qff1, d2, bh, d1, qd, d2, bh, d1]
arc_to_ssf_2 = [dispsupf; mssf_2]
ss_to_arcr_2 = [mdcr_2; dispcrer_2]

sextant1 = [repeat(fodossf, 4); ss_to_arcf; repeat(fodoaf, 20); arc_to_ssf_2;
            repeat(fodossf_2, 4)]
sextant3 = [repeat(fodossr_2, 4); ss_to_arcr_2; repeat(fodoar, 20); arc_to_ssr;
            repeat(fodossr, 4)]

ring = Beamline([sextant1; sextant3; sextant5; sextant7; sextant9; sextant11],
                species_ref=electron, pc_ref=pc_ref)
```

4. The targets need three `TwissExpr`s: of the new straight section cell (its periodic Twiss
   parameters), of the line from the arc into the new straight section (the first half of the tune
   cell), and of the ring (its tunes):

```{code-cell} julia
cell_ss2 = Beamline(fodossf_2, species_ref=electron, pc_ref=pc_ref)
line_2 = Beamline(arc_to_ssf_2, species_ref=electron, pc_ref=pc_ref)

tw_ss2 = TwissExpr(cell_ss2)
tw_2 = TwissExpr(line_2, a_initial=tw_af.a[end])
tw_ring = TwissExpr(ring)
```

   All three depend on the knobs, and `match!` computes each of them once per evaluation.

5. Run the match:

```{code-cell} julia
match!(vary=[@knob(qfss_2.Kn1), @knob(qdss_2.Kn1), @knob(qff2_2.Kn1), @knob(qdf2_2.Kn1),
             @knob(qff3_2.Kn1), @knob(qdf3_2.Kn1)],
       targets=[tw_2.beta1[end] == tw_ss2.beta1[begin],
                tw_2.alpha1[end] == tw_ss2.alpha1[begin],
                tw_2.beta2[end] == tw_ss2.beta2[begin],
                tw_2.alpha2[end] == tw_ss2.alpha2[begin],
                tw_ring.q1 == 54.08, tw_ring.q2 == 54.14],
       solver=:gauss_newton)
```

   The tunes of the ring are now 54.08 and 54.14, and the beta functions are only changed in the
   tune cell:

```{code-cell} julia
tw = twiss(ring)
using CairoMakie
fig = Figure(size=(800, 300))
ax = Axis(fig[1, 1], xlabel="s [m]", ylabel="β [m]")
rows = (0.0, 1300.0) # The sextants 1 and 3
lines!(ax, tw.s[rows], tw.beta1[rows], label="β₁")
lines!(ax, tw.s[rows], tw.beta2[rows], label="β₂")
axislegend(ax, position=:rt)
fig
```

## Exercises

1. **Periodicity inside the tune cell:** In the Bmad version of this tutorial, the periodicity of
   the Twiss parameters inside the tune cell was a target of its own: the Twiss parameters at the
   fourth `qfss_2` of the ring must be equal to those at the fifth. Write these targets with
   `tw_ring` only. Hint: `ring[[qfss_2]]` is the vector of all placements of `qfss_2` in the
   ring, and each of them is a location by itself.

2. **Other tunes:** Change the tunes to 54.12 and 54.18. How much do the quadrupole strengths of the
   tune cell change? Which tunes can the tune cell reach?

The lattice file of this chapter is
[`lattices/07-tune-cell.jl`](https://github.com/bmad-sim/SciBmad.jl/tree/main/docs/src/ring-design/lattices/07-tune-cell.jl).
