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
from a fresh install to your first Twiss calculation and tracking run.

:::{note}
Except where noted, every Julia block on this page is executed when the documentation is
built, and the output shown underneath it is the real result — nothing is pasted in by hand.
See [Runnable pages](https://github.com/bmad-sim/SciBmad.jl/blob/main/docs/README.md)
in the docs README for how to write one.
:::

```{code-cell} julia
:tags: [remove-cell]
# Hidden setup: keep printed tables narrow enough for the page.
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

## Installation

SciBmad runs on Windows, macOS, and Linux. Once Julia is set up, install the package from
the Julia REPL:

```julia
import Pkg
Pkg.add("SciBmad")
```

See [Installation](installation.md) for the full instructions, including installing Julia
itself and using SciBmad from Python.

## A simple copy-paste example

The following is a complete tour of SciBmad in one block — build a FODO cell, retune it,
compute linear, nonlinear, chromatic and spin Twiss parameters, and track particles with
spin. The rest of this page walks through it step by step.

```julia
using SciBmad

# Define lattice elements using @elements
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-1.2, L=0.2)
end

# Construct a Beamline and define the reference species + momentum
fodo = Beamline([qf, sf, b, d, qd, sd, b, d],
        species_ref=Species("electron"), pc_ref=18e9)

# Change the quadrupole strengths
qf.Kn1 = 0.37
qd.Kn1 = -0.37

# Find the drifts `d` in `fodo`
ds = fodo[d];

# Get the first element of `fodo`
first_ele = fodo[1];

# Gets all quadrupoles
quadrupoles = fodo[x -> x.kind == "Quadrupole"];

# Compute the periodic Twiss parameters
tw = twiss(fodo)

#=
  Compute only specific periodic Twiss parameters
  "beta1" = horizontal-like beta
  "phi2" = vertical-like phase advance,
  "dx" = horizontal dispersion
=#
tw = twiss(fodo, cols=["beta1", "phi2", "dx"])

#=
  Compute higher order chromatic quantities such as
  the chromaticity, Montague W functions ("w1", "w2"),
  and 2nd order dispersions ("dx_2", "dy_2") by setting
  the order of δ (chrom) equal to 2
=#
tw = twiss(fodo, cols=["w1", "w2", "dx_2", "dy_2"], chrom=2)
print(tw.q2)
#= Prints:
  AmplitudeDependentValue:
   0.05253999423230923 - 0.00471066037657978 δ
=#
# Get the y-chromaticity
chromy = getterm(tw.q2, delta=1)

# Resonance driving terms h3000, h2100, require order=2
tw = twiss(fodo, cols=["h3000", "h2100"], order=2)

#=
  Spin analysis (invariant spin field as Taylor series,
  amplitude-dependent spin tune).
=#
tw = twiss(fodo, cols=["nx", "ny", "nz"], spin=true,
      as_taylor_series=true, order=2)
print(tw.qspin)
#= Prints:
  AmplitudeDependentValue:
   -0.3094612779434053 - 0.309461277627552 δ - 7.243608429370996e-10 J₁
    + 27.379264609438568 J₂ - 4.343783185854744e-10 δ²
=#

#=
  Track a particle with [x, px, y, py, z, pz]
  = [1e-3 0. 0. 0. 0. 1e-3] for n_turns=10
=#
res = track(fodo, v0=[1e-3 0. 0. 0. 0. 1e-3], n_turns=10)

# Also do spin quaternion tracking:
res = track(fodo, v0=[1e-3 0. 0. 0. 0. 1e-3], n_turns=10, spin=true)
# Apply spin quaternion to horizontal initial spin direction (FAST):
s = track_spin(res.q, [1, 0, 0])

# Track a bunch, give matrix of size n_particles x 6:
n_particles = 10
res = track(fodo, v0=rand(n_particles, 6).*1e-5, n_turns=10, spin=true)
```

## Constructing a Beamline

Everything starts from `using SciBmad`, which re-exports the lattice element types from
[Beamlines.jl](https://bmad-sim.github.io/Beamlines.jl/stable/):

```{code-cell} julia
using SciBmad
```

Let's start by constructing a simple FODO cell `Beamline`.

To do this, we will define six `LineElement`s corresponding to each of these objects. The
lengths of each object are specified by `L` (in meters), and the quadrupole strengths can be
set using the property `Kn1`, where `n` means the "normal" multipole (`s` would be "skew")
and `1` means 1st order multipole (quadrupole). The normal sextupole strength is thus `Kn2`.
For the bend, the `angle` parameter sets both the curvature of the coordinate system and the
corresponding magnetic field so that a "reference particle" will follow the coordinate
system curvature exactly. The coordinate system curvature and normalized field strength can
be set independently using `g_ref` and `Kn0` respectively if desired.

Wrapping the definitions in an `@elements` block makes each element pick up its variable
name automatically:

```{code-cell} julia
# Define lattice elements using @elements
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-1.2, L=0.2)
end

# Construct a Beamline and define the reference species + momentum
fodo = Beamline([qf, sf, b, d, qd, sd, b, d],
        species_ref=Species("electron"), pc_ref=18e9)
```

Note that, under the hood, all element "kinds" (`Sextupole`, `Quadrupole`, etc.) are one
single type `LineElement`. Therefore, there is nothing preventing you from writing e.g.
`Drift(L=1.2, Ks8=123)`. This structure provides maximal flexibility for defining and
modifying elements — see [Defining a LineElement](element.md).

The [`AtomicAndPhysicalConstants`](https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl)
package is used for specifying particle species, and so any species defined by that package
may be provided. Also, instead of specifying the reference momentum `pc_ref`, we
alternatively could have specified the total reference energy `E_ref`, or the *signed*
magnetic rigidity `p_over_q_ref`.

We can find instances of elements in a `Beamline` by "indexing" it in the following manner:

```{code-cell} julia
# Find the drifts `d` in `fodo`
ds = fodo[d];

# Get the first element of `fodo`
first_ele = fodo[1];

# Gets all quadrupoles
quadrupoles = fodo[x -> x.kind == "Quadrupole"];
```

Note that `LineElement`s in a beamline simply inherit all properties from the original
element definition. For example, retuning the quadrupole strengths

```{code-cell} julia
# Change the quadrupole strengths
qf.Kn1 = 0.37
qd.Kn1 = -0.37
```

is reflected everywhere that element has been placed in a `Beamline`, without needing to
rebuild `fodo` or search-and-replace through its element list. More on this in
[Defining a Beamline](beamline.md).

Larger machines are usually defined in a lattice file that you `include`. Several
ready-to-run lattices live in the
[`lattices/`](https://github.com/bmad-sim/SciBmad.jl/tree/main/lattices) directory:

```julia
include("lattices/esr-v6.3.1-tapered.jl")   # defines `ring`
```

## Twiss

Once a `Beamline` is defined, we can compute the Twiss parameters with `twiss`:

```{code-cell} julia
# Compute the periodic Twiss parameters
tw = twiss(fodo)
```

By default this computes the tunes and, in the Sagan-Rubin/Edwards-Teng coupling formalism,
the beta functions, alpha functions, coupling matrix, phase advances/slip, closed orbit and
(crab) dispersions at *all integration steps* in a
[`DataFrame`](https://dataframes.juliadata.org/stable/) struct.

The summary quantities are properties of the result — `q1` and `q2` are the tunes:

```{code-cell} julia
tw.q1, tw.q2
```

If only a subset of quantities is needed, the `cols` keyword restricts the computation (and
the columns returned) to exactly what you ask for, which can save computation time:

```{code-cell} julia
#=
  Compute only specific periodic Twiss parameters
  "beta1" = horizontal-like beta
  "phi2" = vertical-like phase advance,
  "dx" = horizontal dispersion
=#
tw = twiss(fodo, cols=["beta1", "phi2", "dx"])
```

The per-element table is a `DataFrame`, reachable as `tw.df`, and its columns are forwarded
onto `tw` itself:

```{code-cell} julia
tw.df.beta1
```

`twiss` does much more than the linear optics: chromatic and amplitude-dependent quantities,
resonance driving terms, the invariant spin field and the spin tune, and derivatives with
respect to element parameters. All of that is covered in [Twiss](twiss.md).

## Track

To do particle tracking, a single function `track` is provided, which accepts initial phase
space coordinates as an `n_particle x 6` matrix:

```{code-cell} julia
#=
  Track a particle with [x, px, y, py, z, pz]
  = [1e-3 0. 0. 0. 0. 1e-3] for n_turns=10
=#
res = track(fodo, v0=[1e-3 0. 0. 0. 0. 1e-3], n_turns=10)
```

Spin quaternion tracking is enabled with `spin=true`, and `track_spin` then applies the
tracked quaternions to any initial spin direction:

```{code-cell} julia
# Also do spin quaternion tracking:
res = track(fodo, v0=[1e-3 0. 0. 0. 0. 1e-3], n_turns=10, spin=true)
# Apply spin quaternion to horizontal initial spin direction (FAST):
s = track_spin(res.q, [1, 0, 0])
```

Tracking a bunch just means giving `track` more rows:

```{code-cell} julia
# Track a bunch, give matrix of size n_particles x 6:
n_particles = 10
res = track(fodo, v0=rand(n_particles, 6).*1e-5, n_turns=10, spin=true)
```

On the CPU, all tracking will automatically be parallelized using single instruction,
multiple data (SIMD) if your hardware supports it. CPU multithreading can also be enabled by
starting Julia with threads, and setting `use_cpu_multithreading=true`. For more details,
see the [CPU Parallelization](#cpuparallel) section of the manual.

For GPU parallelized tracking, simply initialize your initial particle coordinates as a GPU
array. For example:

```julia
using CUDA
n_particles = 100000 # 100,000 particles
v0 = CUDA.rand(Float64, n_particles, 6) .* 1e-5
res = track(fodo, v0=v0, n_turns=10, spin=true)
```

For more details, see the [GPU Parallelization](#gpuparallel) section of the manual.

## Choosing a tracking method

Every element carries a `tracking_method`, which defaults to `SciBmadStandard`. To use a
different one — for example an explicitly symplectic integrator with radiation damping
turned on — assign it to the elements:

```{code-cell} julia
tm = Symplectic(order=2, radiation_damping_on=true)
foreach(t -> t.tracking_method = tm, fodo.line)
fodo.line[1].tracking_method
```

See [Tracking Methods](tracking-methods.md) for the available methods.

## Where to go next

- **[Defining a LineElement](element.md)** and **[Defining a Beamline](beamline.md)** — the
  lattice data model in full.
- **[Twiss](twiss.md)** — linear, nonlinear, chromatic, parametric and spin lattice functions.
- **[Track](track.md)** — the `TrackingResult`, configuration settings, callbacks, and
  CPU/GPU parallelization.
- **[Examples](examples-index.md)** — full Jupyter notebooks (with outputs) for Twiss,
  dynamic aperture, autodifferentiation, spin tracking, and fitting.
- **{external:doc}`API Reference <index>`** — docstrings for every type and function.
