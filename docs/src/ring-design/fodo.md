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

(rd.fodo)=
# 1. FODO Cells

```{code-cell} julia
:tags: [remove-cell]
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

```{figure} figures/fodoaf.svg
:width: 95%

Forward arc FODO cell. QF, QD: focusing and defocusing quadrupoles. SF, SD: sextupoles near the
focusing and defocusing quadrupoles.
```

```{figure} figures/fodoar.svg
:width: 95%

Reverse arc FODO cell.
```

```{figure} figures/sextant.svg
:width: 80%

A sextant of the ring.
```

## Introduction

In the arc FODO cells of the Electron Storage Ring (ESR), a sextupole, a bend, and a corrector
coil all need to fit in between each pair of quadrupoles. To accommodate the sextupoles, the
quadrupoles cannot be centered between the bends. As a result, there are two types of FODO cells
in the ESR: one which has the sextupoles placed before the quadrupoles and one which has the
sextupoles placed after the quadrupoles. We'll call these the "forward" and "reverse" cells
respectively, shown in the figures above. This tutorial will use the convention that all FODO
cells start with a full-length horizontally focusing ($K_1 > 0$) quadrupole.

The ring is made up of 6 sextants, as discussed in [Constructing the Ring](ring.md). There are
two types of sextants: forward sextants with forward FODO cells and reverse sextants with reverse
FODO cells. The two types of sextants have mirror symmetry with respect to one another. Each
sextant consists of two half-length straight sections with an arc section in between, as shown
above.

The first straight section, in which the dispersion is zero, is followed by a dispersion
"creator" to match to the periodic dispersion in the arc. After the arc, there is a dispersion
"suppressor" to match to the dispersion-free second straight section.

## Example: Forward Arc FODO Cell

The phase advance per FODO cell for the lattice to be designed is chosen to be 90° to achieve
the emittance requirements. Construction of the forward arc FODO cell without sextupoles is
demonstrated below. Later in the tutorial the sextupoles and corrector coils will be added in.
Left as an exercise is the construction of the reverse arc FODO cell without sextupoles, along
with the forward and reverse straight section FODO cells.

```{figure} figures/fodoaf-dim.svg
:width: 95%

Forward arc FODO cell with lengths specified.
```

The figure shows the lengths of each element in the forward arc FODO cell. The ESR uses
rectangular bends in the arcs. A rectangular bend has straight, parallel pole faces, and the
length given for it ($L_\mathrm{chord}$ = 6.86 m) is the length of the straight line between the
centers of its pole faces. SciBmad has only sector bends (`SBend`), whose length `L` is the length
of the arc that the reference trajectory follows. A rectangular bend is a sector bend with the
edge angles `e1` and `e2` both half the bend angle $\theta$, and with the arc length
$L = L_\mathrm{chord}\,\theta / (2\sin(\theta/2))$.

Our collider ring will have 132 arc FODO cells, and so each bend must bend the orbit by an angle
of $2\pi/(2 \cdot 132)$. Choosing initially arbitrary quadrupole strengths `Kn1` (the normal
quadrupole strength $K_1$, in 1/m²), the lattice for this cell is:

```{code-cell} julia
using SciBmad

θ = pi/132   # Bend angle of the arc bends [rad]: 264 of them make 2π
Lc = 6.86    # Chord length of the arc bends [m]

@elements begin
  qf = Quadrupole(L=0.5, Kn1=0.4)
  qd = Quadrupole(L=0.5, Kn1=-0.4)
  b = SBend(L=Lc*θ/(2*sin(θ/2)), angle=θ, e1=θ/2, e2=θ/2) # Rectangular bend
  d1 = Drift(L=0.609)
  d2 = Drift(L=1.241)
end

fodoaf = [qf, d1, b, d2, qd, d1, b, d2] # Forward arc FODO cell

electron = Species("electron")
pc_ref = 17.846262612447e9 # Reference momentum [eV/c]

cell_af = Beamline(fodoaf, species_ref=electron, pc_ref=pc_ref)
```

The `@elements` block names each element after its variable (`qf.name` is `"qf"`). `fodoaf` is a
plain Julia vector of the elements in the order of the cell, and the `Beamline` `cell_af` places
them along a reference trajectory for an electron with the reference momentum `pc_ref`. A
`Beamline` is what SciBmad analyzes and tracks through. The same element can be placed in many
beamlines and many times in one: changing a parameter of `qf` changes it everywhere it is placed.
For more details, see [Defining a LatElement](../element.md) and
[Defining a Beamline](../beamline.md).

The periodic optics of the cell are computed with `twiss`:

```{code-cell} julia
tw = twiss(cell_af)
```

The output shows first the summary values for the cell. `q1` and `q2` are the phase advances
through the periodic cell, in units of 2π. The numbers 1 and 2 refer to the two transverse normal
modes, which are the horizontal and the vertical motion when there is no coupling, as here. (In
Bmad these are the modes `a` and `b`.) `etac` and `alphac` are the slip factor and the momentum
compaction. The table has one row per location: here the beginning of every element (each element
is integrated in one step) and the end of the cell. The columns are the position `s`, the beta
functions, the phase advances `phi1` and `phi2` (in units of 2π), the dispersion `dx`, and so on.

The columns are read at a location, which can be given by an element (placed once) or by its
index in the beamline. `end` is the end of the beamline. For example, at the end of the cell:

```{code-cell} julia
tw.beta1[end], tw.alpha1[end], tw.dx[end], tw.phi1[end]
```

Several placements, or several elements, give a vector:

```{code-cell} julia
tw.beta1[[d1]] # At the beginning of both placements of d1
```

All the ways to give a location are described in [Twiss](../twiss.md). The whole table is
`tw.df`, a [`DataFrame`](https://dataframes.juliadata.org/stable/), from which any columns can be
selected:

```{code-cell} julia
tw.df[:, [:name, :s, :beta1, :alpha1, :phi1, :dx]]
```

To plot the optics, use a plotting package such as [Makie](https://docs.makie.org/stable/):

```{code-cell} julia
using CairoMakie

fig = Figure(size=(700, 450))
ax = Axis(fig[1, 1], ylabel="β [m]")
lines!(ax, tw.s, tw.beta1, label="β₁")
lines!(ax, tw.s, tw.beta2, label="β₂")
axislegend(ax, position=:ct)
ax2 = Axis(fig[2, 1], xlabel="s [m]", ylabel="η [m]")
lines!(ax2, tw.s, tw.dx)
fig
```

The curves are drawn through the rows of the table. With more integration steps per element
(`b.n_steps = 10`, see [Tracking Methods](#track.steps)), the table has more rows and the curves
follow the optics inside the elements.

### Matching

Given our arbitrary choice of quadrupole strengths, the phase advance through our FODO cell is
not 90°. In order to determine which strengths produce the desired phase advance, the optimizer
of SciBmad, `match!`, can be used. It needs to know the targets (what should be achieved) and the
knobs (what may be changed).

The targets are written with the quantities of `twiss`. A `TwissExpr` gives access to all of them
in the same way as the result of `twiss` does, but computes nothing until it is needed:

```{code-cell} julia
tw_af = TwissExpr(cell_af)
```

`tw_af.q1` is the horizontal phase advance through the cell (in units of 2π), and a target is
written with `==`:

```{code-cell} julia
targets = [tw_af.q1 == 0.25, tw_af.q2 == 0.25]
```

The knobs are the parameters to vary, given with `@knob`. Here they are the strengths `Kn1` of
`qf` and `qd`:

```{code-cell} julia
vary = [@knob(qf.Kn1), @knob(qd.Kn1)]
```

Optimization involves the minimization of a merit function, which `match!` evaluates as

$$
\mathcal{M} = \sum_i \left( w_i \, \delta D_i \right)^2
$$

where the sum is over the targets, $w_i$ are their weights (1 by default) and $\delta D_i$ is the
difference of the value of target $i$ from its goal. For `tw_af.q1 == 0.25`, this is the value of
`tw_af.q1` (computed from the lattice) minus 0.25. A target counts as met when $\delta D_i$ is within
its tolerance, `1e-8` by default. Weights and tolerances are set by writing a target as, e.g.,
`Target(tw_af.q1, 0.25; weight=10, tol=1e-10)`. Weights can have a large effect on optimizations
when the targets cannot all be met. Often the best way to choose them is by trial and error: is
the optimization converging well or not?

Now the match is run:

```{code-cell} julia
match!(vary=vary, targets=targets)
```

`match!` reports whether the match converged, the value of each target and its goal, and the
values of the knobs before and after. The knobs are now set to their new values:

```{code-cell} julia
qf.Kn1, qd.Kn1
```

and the phase advances through the cell are 90°:

```{code-cell} julia
tw = twiss(cell_af)
tw.q1, tw.q2
```

Most optimizers need the derivatives of the targets with respect to the knobs. Other codes, such
as Tao, compute them with finite differences, which are approximate, need a step size, and need
one more calculation for each knob. SciBmad computes the exact derivatives with respect to all
knobs at once, with automatic differentiation (see
[Matching and Optimization](../optimize.md)). `match!` takes Gauss-Newton steps (the smallest
change of the knobs that meets the targets to first order), and shorter or damped steps where those
do not improve the match. It finds a nearby solution, and converges to a local optimum. For a global search, see
[Using Other Optimizers](../optimize.md).

There is no need to save the optimized lattice to a file: the strengths of `qf` and `qd` have
been changed, and every beamline in which they are placed now has them. In your lattice file,
keep the `match!` call after the definitions. It runs each time the file is included, so the file
records how the strengths were found.

## Exercises

1. **Reverse arc FODO:** Construct the reverse arc FODO cell `fodoar` without sextupoles, with
   the elements in the order of the figure of the reverse arc cell, and determine the quadrupole
   strengths for a 90° phase advance.

2. **Forward straight section FODO:** Replace the bends in the forward cell by a drift (empty
   space) of length 5.855 m, named `db`. This length is chosen to produce the same straight
   section length as in the RHIC tunnel. Additionally, use new horizontally focusing and
   defocusing quadrupoles `qfss` and `qdss`. Which quadrupole strengths lead to 90° phase advance
   in both planes? Name the line `fodossf`.

3. **Reverse straight section FODO:** Replace the bends in the reverse cell by the drift `db`, and
   use the quadrupoles `qfss` and `qdss`. Which quadrupole strengths lead to 90° phase advance in
   both planes? Name the line `fodossr`. These quadrupole strengths should be exactly the same as
   for `fodossf`.

4. **Analytical FODO cell:** Take a FODO model with the same total length as constructed above,
   but with zero length quadrupoles and no bends. Calculate by hand the quadrupole strengths that
   lead to a 90° phase advance in $x$ and $y$. What are the beta functions at the thin
   quadrupoles? How do the quadrupole strengths and beta functions compare to above? (A thin
   quadrupole is a `Multipole` with an integrated strength `Kn1L`; you can check your answer with
   `twiss`.)

5. **Forward and reverse cells:** Check that your forward and reverse arc FODO cells, which both
   start with focusing quadrupoles, have different periodic beta and alpha functions. Check also
   that both cells, for the same phase advance of 90°, have exactly the same quadrupole strengths.
   Explain why this is so.

6. **Arc and chord lengths:** What is the difference between the arc length `L` of the bends and
   their chord length? Check with the geometry: the chord of an arc of length $L$ and radius
   $\rho = L/\theta$ is $2\rho\sin(\theta/2)$. The curvature of the reference trajectory in a bend is
   `b.g_ref` (in 1/m).

7. **Finding elements:** What do the following lines give? See
   [Searching by Name](#search:name).
   ```julia
   cell_af["Quadrupole::*"]
   [ele.Kn1 for ele in cell_af["q*"]]
   ```

8. **Targets and knobs:** A target and a knob are objects of their own. Look at their
   documentation with `?Target` and `?Knob` in the REPL, and find out how to give a knob lower
   and upper bounds.

9. **Fewer targets:** Run the match with only the target `tw_af.q1 == 0.25`, starting from
   `qf.Kn1 = 0.4` and `qd.Kn1 = -0.4` again. Is the match successful? What is the vertical phase
   advance afterwards? With more knobs than targets, many solutions exist, and `match!` returns one
   near the starting point.

The solutions of exercises 1 to 3 are in the lattice file of this chapter,
[`lattices/01-fodo.jl`](https://github.com/bmad-sim/SciBmad.jl/tree/main/docs/src/ring-design/lattices/01-fodo.jl).
