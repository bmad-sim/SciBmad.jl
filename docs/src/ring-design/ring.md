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

(rd.ring)=
# 5. Constructing the Ring

```{code-cell} julia
:tags: [remove-cell]
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

## Ring Construction

```{figure} figures/ring.png
:width: 100%

Mirror symmetry about the centers of each straight section.
```

In the previous chapters, forward and reverse arc cells, forward and reverse straight section
cells, and forward and reverse lines to connect the arcs to the straight sections (with dispersion
suppression and matching of the beta functions) were constructed. In this chapter, the entire ring
is created, based on the figure above. All that is left is a section to match the straight section
to the arc, including dispersion "creation" and matching of the beta functions, as shown in the
figure of a sextant below.

```{figure} figures/sextant.svg
:width: 80%

A sextant of the ring.
```

Does this mean that a dispersion creator for both the forward and reverse sextants needs to be
matched? Fortunately, no! Considering the mirror symmetry shown above, the quadrupole settings for
the forward sextant dispersion suppression and matching are equal to the quadrupole settings for
the reverse dispersion *creation* and matching. Likewise, the reverse dispersion suppression and
matching quadrupole settings are equal to those of the forward dispersion creation and matching.
Therefore, no matching is necessary here. All that is needed is to connect the lines together.

## Example: Forward Straight Section to Arc

Start with the lattice of [chapter 3](match-to-straight.md):

```{code-cell} julia
include("lattices/03-match-to-straight.jl");
```

As an example, the forward line to connect the straight section to the arc, `ss_to_arcf`, will be
constructed. Following the figure of the sextant, both the matching section to the dispersion
creator (`mdcf`) and the dispersion creator itself (`dispcref`) need to be constructed. By the
mirror symmetry, `ss_to_arcf` uses the same quadrupoles as `arc_to_ssr`, in reverse order. Keeping
with our convention of starting each line with a full horizontally focusing quadrupole, this is

```{code-cell} julia
# Match the forward straight section to the dispersion creator (reverse quadrupoles)
mdcf = [qfss, d1, db, d2, qdr3, d1, db, d2, qfr3, d1, db, d2, qdr2, d1, db, d2]

# Forward dispersion creator (reverse quadrupoles)
dispcref = [qfr2, d1, bh, d2, qdr1, d1, bh, d2, qfr1, d1, bh, d2, qd, d1, bh, d2]

ss_to_arcf = [mdcf; dispcref];
```

Note that the elements in a forward line keep the forward order of the drifts (`d1` after a
quadrupole, `d2` before it); only the quadrupoles come from the reverse sextant.

After matching the straight sections to the arcs, it is now time to construct the full ring. The
reverse line `ss_to_arcr` is constructed in the same way (exercise 1). In Julia,
`repeat(fodoaf, 20)` repeats the cell 20 times, and `[a; b; c]` joins the lines `a`, `b` and `c`:

```{code-cell} julia
:tags: [remove-cell]
mdcr = [qfss, d2, db, d1, qdf3, d2, db, d1, qff3, d2, db, d1, qdf2, d2, db, d1]
dispcrer = [qff2, d2, bh, d1, qdf1, d2, bh, d1, qff1, d2, bh, d1, qd, d2, bh, d1]
ss_to_arcr = [mdcr; dispcrer];
```

```{code-cell} julia
sextant1 = [repeat(fodossf, 4); ss_to_arcf; repeat(fodoaf, 20); arc_to_ssf; repeat(fodossf, 4)]
sextant3 = [repeat(fodossr, 4); ss_to_arcr; repeat(fodoar, 20); arc_to_ssr; repeat(fodossr, 4)]
sextant5 = [repeat(fodossf, 4); ss_to_arcf; repeat(fodoaf, 20); arc_to_ssf; repeat(fodossf, 4)]
sextant7 = [repeat(fodossr, 4); ss_to_arcr; repeat(fodoar, 20); arc_to_ssr; repeat(fodossr, 4)]
sextant9 = [repeat(fodossf, 4); ss_to_arcf; repeat(fodoaf, 20); arc_to_ssf; repeat(fodossf, 4)]
sextant11 = [repeat(fodossr, 4); ss_to_arcr; repeat(fodoar, 20); arc_to_ssr; repeat(fodossr, 4)]

ring = Beamline([sextant1; sextant3; sextant5; sextant7; sextant9; sextant11],
                species_ref=electron, pc_ref=pc_ref)
```

The optics of the whole ring:

```{code-cell} julia
tw = twiss(ring)
```

```{code-cell} julia
using CairoMakie
fig = Figure(size=(800, 450))
ax = Axis(fig[1, 1], ylabel="β [m]")
lines!(ax, tw.s, tw.beta1, label="β₁")
lines!(ax, tw.s, tw.beta2, label="β₂")
axislegend(ax, position=:rt)
ax2 = Axis(fig[2, 1], xlabel="s [m]", ylabel="η [m]")
lines!(ax2, tw.s, tw.dx)
linkxaxes!(ax, ax2)
fig
```

The beta functions are periodic in the cells of the arcs and of the straight sections, and the
dispersion is zero in the straight sections:

```{code-cell} julia
maximum(abs, tw.dx[1:33]) # Through the first straight section: the elements 1 to 32 [m]
```

Here `1:33` are beamline indexes: the beginnings of the elements 1 to 32 of the ring, and of the
element 33, which is the end of the straight section.

## Exercises

1. **Reverse straight section to arc:** Construct `ss_to_arcr`, using the example above and the
   figures of the ring and of the sextant.

2. **Construct the ring:** Construct the ring yourself from the lines of the previous chapters, as
   above. What are its tunes? Why are they not multiples of the phase advances of the cells?

3. **Check that the ring closes geometrically:** The sum of the bend angles must be $2\pi$:
   `sum(ele.g_ref*ele.L for ele in ring.line)`. Compute also the global coordinates at the end of the
   ring, with the loop of [Machine Coordinates](coordinates.md): they must be those at the beginning.
   What is the circumference of the ring?

4. **Floor plan:** Plot the global coordinates $Z$ against $X$ at the exit ends of all elements, to
   draw the floor plan of the ring. Can you see the six arcs and straight sections? Draw the
   floor plan of one sextant.

The lattice file of this chapter is
[`lattices/05-ring.jl`](https://github.com/bmad-sim/SciBmad.jl/tree/main/docs/src/ring-design/lattices/05-ring.jl).
