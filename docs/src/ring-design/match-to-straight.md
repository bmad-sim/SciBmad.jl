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

(rd.match)=
# 3. Matching the Dispersion Suppressor to the Straight Section

```{code-cell} julia
:tags: [remove-cell]
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

## Introduction

The periodic beta functions in the arcs are different from the periodic beta functions in the
straight sections. After suppressing the dispersion to zero, four quadrupoles after the
dispersion suppressor are needed to match $\alpha_1$, $\alpha_2$, $\beta_1$, and $\beta_2$ to the
periodic ones of the following straight section.

```{figure} figures/match.svg
:width: 100%

Dispersion suppressor and matching section.
```

## Example: Forward Matching Section

The procedure for creating the forward matching section is shown below. Matching the reverse
matching section is left as an exercise.

1. Start with the lattice of [chapter 2](dispersion-suppressor.md), which contains the
   dispersion suppressors, and the straight section cells of chapter 1 (exercises 2 and 3):

```{code-cell} julia
include("lattices/02-dispersion-suppressor.jl");
```

2. Define four quadrupoles for matching to the straight section. Because this is the forward
   matching section (F), and these are the 2nd and 3rd special focusing and defocusing quadrupoles,
   they are named `qff2`, `qdf2`, `qff3`, and `qdf3`. They start with the strength of the straight
   section quadrupoles:

```{code-cell} julia
k_ss = qfss.Kn1 # The straight section strength, to start with

@elements begin
  qff2 = Quadrupole(L=0.5, Kn1=k_ss)
  qdf2 = Quadrupole(L=0.5, Kn1=-k_ss)
  qff3 = Quadrupole(L=0.5, Kn1=k_ss)
  qdf3 = Quadrupole(L=0.5, Kn1=-k_ss)
end;
```

3. Define the matching section `mssf` of the figure, with the straight section drifts `db`, and the
   line `arc_to_ssf` from the end of the arc to the straight section, which contains the
   dispersion suppressor and the matching section. In Julia, `[a; b]` joins the vectors `a` and `b`:

```{code-cell} julia
mssf = [qff2, d1, db, d2, qdf2, d1, db, d2, qff3, d1, db, d2, qdf3, d1, db, d2]
arc_to_ssf = [dispsupf; mssf]
line_f = Beamline(arc_to_ssf, species_ref=electron, pc_ref=pc_ref)
tw_f = TwissExpr(line_f, a_initial=tw_af.a[end])
```

4. The last four quadrupoles need to be matched so that at the end of `arc_to_ssf` the Twiss
   parameters are equal to the periodic ones of the straight section FODO cells. These are
   computed by the `TwissExpr` `tw_ssf` of the straight section cell (chapter 1, exercise 2): its
   Twiss parameters at the beginning of the cell are `tw_ssf.beta1[begin]` and so on. The targets
   therefore compare two quantities, one of each `TwissExpr`:

```{code-cell} julia
targets = [tw_f.beta1[end] == tw_ssf.beta1[begin], tw_f.alpha1[end] == tw_ssf.alpha1[begin],
           tw_f.beta2[end] == tw_ssf.beta2[begin], tw_f.alpha2[end] == tw_ssf.alpha2[begin]]
```

   No number has to be copied from one lattice to another: if the straight section cells change,
   these targets follow.

5. Run the match. A matching problem like this one can have several solutions: the knobs that make
   the targets hold are not unique. Which solution is found depends on the starting point and on
   the method of the optimizer. Here we use the Gauss-Newton method of `match!` (`solver=:gauss_newton`),
   which finds the solution of the Bmad version of this tutorial:

```{code-cell} julia
match!(vary=[@knob(qff2.Kn1), @knob(qdf2.Kn1), @knob(qff3.Kn1), @knob(qdf3.Kn1)],
       targets=targets, solver=:gauss_newton)
```

   The default method, Levenberg-Marquardt, finds another solution (exercise 3).

6. The betas through the line, from the arc into the straight section:

```{code-cell} julia
using CairoMakie
s = value(tw_f.s[:])
fig = Figure(size=(700, 350))
ax = Axis(fig[1, 1], xlabel="s [m]", ylabel="β [m]")
lines!(ax, s, value(tw_f.beta1[:]), label="β₁")
lines!(ax, s, value(tw_f.beta2[:]), label="β₂")
axislegend(ax, position=:lt)
ax2 = Axis(fig[2, 1], xlabel="s [m]", ylabel="η [m]")
lines!(ax2, s, value(tw_f.dx[:]))
fig
```

   and at the end, together with the periodic values of the straight section:

```{code-cell} julia
value([tw_f.beta1[end], tw_ssf.beta1[begin], tw_f.alpha2[end], tw_ssf.alpha2[begin]])
```

## Exercises

1. **Reverse matching section:** Construct the reverse matching section `mssr` with the quadrupoles
   `qfr2`, `qdr2`, `qfr3`, `qdr3`, following the same steps as in the example but for the reverse
   cells (it follows the reverse dispersion suppressor, and matches into the reverse straight
   section cell). How do your quadrupole strengths compare to those of the forward matching
   section?

2. **Weights:** Optimization success or failure can depend heavily on the relative weights of the
   targets. As an example of this, start again with the unoptimized quadrupoles, and this time give
   the first target the weight `1e5`, by writing it as
   `Target(tw_f.beta1[end] - tw_ssf.beta1[begin], 0; weight=1e5)`. With this, how well does the
   optimization do?

3. **Several solutions:** Start again with the unoptimized quadrupoles, and run the match with
   the default method (without `solver`). Check that the solution is different, and that it meets
   the targets. Compare the maximum beta functions in the matching section for both solutions, for
   example with `value(maximum(tw_f.beta1["qff2:qdf3"]))`. Which would you choose?

The solution of exercise 1 is in the lattice file of this chapter,
[`lattices/03-match-to-straight.jl`](https://github.com/bmad-sim/SciBmad.jl/tree/main/docs/src/ring-design/lattices/03-match-to-straight.jl).
