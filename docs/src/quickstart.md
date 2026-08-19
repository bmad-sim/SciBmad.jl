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

# Quickstart

SciBmad is a high-performance, CPU/GPU-compatible, fully-differentiable accelerator
physics simulation ecosystem, usable from both Julia and Python. This page gets you
from a fresh install to your first tracking run.

:::{note}
Every Julia block on this page is executed when the documentation is built, and the
output shown underneath it is the real result — nothing is pasted in by hand. See
[Runnable pages](https://github.com/bmad-sim/SciBmad.jl/blob/main/docs/README.md)
in the docs README for how to write one.
:::

```{code-cell} julia
:tags: [remove-cell]
# Hidden setup: keep printed tables narrow enough for the page.
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

## Installation

SciBmad runs on Windows, macOS, and Linux. Follow the detailed, per-platform
instructions:

- [Windows](https://github.com/bmad-sim/SciBmad.jl/blob/main/WINDOWS.md)
- [macOS](https://github.com/bmad-sim/SciBmad.jl/blob/main/MAC.md)
- [Linux](https://github.com/bmad-sim/SciBmad.jl/blob/main/LINUX.md)

Once Julia is set up, install the package from the Julia REPL:

```julia
import Pkg
Pkg.add("SciBmad")
```

## Your first lattice

Everything starts from `using SciBmad`, which re-exports the lattice element types
from [Beamlines.jl](https://bmad-sim.github.io/Beamlines.jl/stable/):

```{code-cell} julia
using SciBmad
```

Build the elements, then collect them into a `Beamline`. Wrapping the definitions in
an `@elements` block makes each element pick up its variable name automatically:

```{code-cell} julia
@elements begin
    qf = Quadrupole(Kn1=0.36, L=0.5)
    qd = Quadrupole(Kn1=-0.36, L=0.5)
    d  = Drift(L=1.0)
end

ring = Beamline([qf, d, qd, d], species_ref=Species("electron"), E_ref=18e9)
```

Each cell shares one Julia session with the cells before it, so `ring` is still
available further down the page — exactly like a Documenter `@example` block:

```{code-cell} julia
ring.line[1].L, ring.line[1].Kn1
```

Larger machines are usually defined in a lattice file that you `include`. Several
ready-to-run lattices live in the [`lattices/`](https://github.com/bmad-sim/SciBmad.jl/tree/main/lattices)
directory:

```julia
include("lattices/esr-v6.3.1-tapered.jl")   # defines `ring`
```

## Computing Twiss functions

`twiss` returns the periodic (nonlinear) Twiss functions of the lattice, including
the tunes:

```{code-cell} julia
tw = twiss(ring)
tw.tunes
```

The per-element table is a `TypedTables.Table`, so the usual column access works:

```{code-cell} julia
tw.table.beta_1
```

## Tracking

Choose a tracking method and assign it to the elements:

```{code-cell} julia
tm = Yoshida(order=2, radiation_damping_on=true)
foreach(t -> t.tracking_method = tm, ring.line)
ring.line[1].tracking_method
```

From here you can run dynamic-aperture scans, normal-form analysis, parameter sweeps,
and more — all differentiable and GPU-portable. The [Examples](examples-index.md)
notebooks demonstrate each of these.

## Where to go next

- **[Examples](examples-index.md)** — full Jupyter notebooks (with outputs) for
  Twiss, dynamic aperture, autodifferentiation, spin tracking, and fitting.
- **[Overview](overview.md)** — the SciBmad data model and concepts.
- **{external:doc}`API Reference <index>`** — docstrings for every type and function.
