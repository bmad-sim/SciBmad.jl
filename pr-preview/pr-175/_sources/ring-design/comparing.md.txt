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

(rd.comparing)=
# 13. Comparing Lattices

```{code-cell} julia
:tags: [remove-cell]
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

Bmad's program Tao keeps three copies of a lattice: the "design" lattice as read from the lattice
file, which never changes, the "model" lattice, which the commands of Tao change, and a "base"
lattice, a copy of the model made at some point to compare with. In SciBmad, a lattice is a set of
Julia objects, which you change directly. To compare the lattice before and after a change, keep
the result of `twiss` from before the change: it does not change when the lattice does.

## Example

An open line with a drift, a bend and a quadrupole, for electrons with a total energy of 10 MeV,
starting with beta functions of 10 m:

```{code-cell} julia
using SciBmad

@elements begin
  d = Drift(L=0.5)
  b = SBend(L=0.5, g_ref=1.0, Kn0=1.001, e1=0.1)
  q = Quadrupole(L=0.6, Kn1=0.23)
end
for ele in (d, b, q)
  ele.n_steps = 10 # For smooth plots
end
simple = Beamline([d, b, q], species_ref=Species("electron"), E_ref=10e6)
initial = (beta1=10.0, beta2=10.0)
tw_design = twiss(simple, initial=initial)
```

The field of the bend `Kn0` is a little stronger than the curvature `g_ref` of the reference
trajectory, so the orbit is not zero. Now the lattice is changed: a vertical kick of −0.5 mrad is
added to the drift (a skew dipole field, `Ks0L`), and a horizontal kick of 1 mrad to the quadrupole
(a normal dipole field, `Kn0L`: a positive normal dipole field kicks towards $-x$):

```{code-cell} julia
d.Ks0L = -0.0005
q.Kn0L = -0.001
tw = twiss(simple, initial=initial)
```

The orbit of the changed lattice, and its difference from the design:

```{code-cell} julia
using CairoMakie
fig = Figure(size=(700, 500))
ax1 = Axis(fig[1, 1], ylabel="orbit [mm]")
lines!(ax1, tw.s, 1e3 .* tw.x, label="x")
lines!(ax1, tw.s, 1e3 .* tw.y, label="y")
axislegend(ax1, position=:lt)
ax2 = Axis(fig[2, 1], xlabel="s [m]", ylabel="orbit - design [mm]")
lines!(ax2, tw.s, 1e3 .* (tw.x .- tw_design.x), label="x")
lines!(ax2, tw.s, 1e3 .* (tw.y .- tw_design.y), label="y")
fig
```

To compare with another state than the design, keep another `twiss` result, as Tao's "base"
lattice:

```{code-cell} julia
tw_base = tw
q.Ks0L = 0.0005
tw = twiss(simple, initial=initial)
maximum(abs, tw.y .- tw_base.y), maximum(abs, tw.x .- tw_base.x)
```

To set the lattice back, set the parameters back (`d.Ks0L = 0` etc.). A match can be undone with
`undo!` of its result, e.g. `res = match!(...)`, then `undo!(res)`.

## Exercises

1. **Orbit plots:** Plot the horizontal orbit of the design lattice, and that of the changed
   lattice, in one plot.

2. **Functions:** To save typing, define a function `setit(k)` that sets the vertical kick of `q` to
   `k`, so that `setit(1e-6)` is the same as `q.Ks0L = 1e-6`.

3. **Changing the tunes with the arcs:** Start with the ring of [chapter 9](rf.md), and keep its
   `twiss` result as the design. Then change the tunes to 54.10 and 54.12 by varying the arc
   quadrupoles `qf` and `qd` instead of the tune cell. Plot the relative difference of the beta functions from the design,
   `tw.beta1 ./ tw_design.beta1 .- 1`, around the ring. How much "beta beating" is there? Why does
   the tune cell avoid it?
