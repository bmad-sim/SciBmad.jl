# Quickstart

SciBmad is a high-performance, CPU/GPU-compatible, fully-differentiable accelerator
physics simulation ecosystem, usable from both Julia and Python. This page gets you
from a fresh install to your first tracking run. For runnable, end-to-end examples
see the [Examples](examples-index.md) section, whose notebooks are shown below with
their outputs.

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
from [Beamlines.jl](https://bmad-sim.github.io/Beamlines.jl/stable/). Build elements,
collect them into a `Beamline`, and you have a ring:

```julia
using SciBmad

qf = Quadrupole(Kn1=0.36, L=0.5)
qd = Quadrupole(Kn1=-0.36, L=0.5)
d  = Drift(L=1.0)

ring = Beamline([qf, d, qd, d])
```

Larger machines are usually defined in a lattice file that you `include`. Several
ready-to-run lattices live in the [`lattices/`](https://github.com/bmad-sim/SciBmad.jl/tree/main/lattices)
directory:

```julia
include("lattices/esr-v6.3.1-tapered.jl")   # defines `ring`
```

## Computing Twiss functions and tracking

```julia
# Periodic (nonlinear) Twiss functions, including the tunes
tw = twiss(ring)
tw.tunes

# Choose a tracking method, e.g. 2nd-order Yoshida with radiation damping
tm = Yoshida(order=2, radiation_damping_on=true)
foreach(t -> t.tracking_method = tm, ring.line)
```

From here you can run dynamic-aperture scans, normal-form analysis, parameter sweeps,
and more — all differentiable and GPU-portable. The [Examples](examples-index.md)
notebooks demonstrate each of these.

## Where to go next

- **[Examples](examples-index.md)** — full Jupyter notebooks (with outputs) for
  Twiss, dynamic aperture, autodifferentiation, spin tracking, and fitting.
- **[Overview](overview.md)** — the SciBmad data model and concepts.
- **{external:doc}`API Reference <index>`** — docstrings for every type and function.
```

