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

(batchparams)=
# Batch Parameters

```{code-cell} julia
:tags: [remove-cell]
using SciBmad
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

## Basics

In SciBmad, it is possible to not only CPU/GPU parallelize over particles in a bunch, but
also to parallelize over many different **accelerator parameters**. This is possible using
SciBmad's `BatchParam` type.

As a first example, say we want to simulate 1 particle travelling through 10 different FODO
cells in parallel. We first define our FODO cell, using a
[deferred expression](defexpr.md) for the quadrupole strength:

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=DefExpr(c -> c.K1), L=0.4)
  d = Drift(L=0.5)
  qd = Quadrupole(Kn1=DefExpr(c -> -c.K1), L=0.4)
end

fodo = Beamline([qf, d, qd, d], E_ref=18e9,
        species_ref=Species("electron"))
```

We can then set the `K1` property of `fodo.context` to be a `BatchParam` with 10 different
quadrupole strengths

```{code-cell} julia
fodo.context.K1 = BatchParam(collect(0.1:0.05:0.55))
```

Now when we call `track` with this `Beamline`, the 1st particle will see `K1=0.1`, the 2nd
particle will see `K1=0.15`, the third particle `K1=0.2`, etc. So we'll make all initial
particle coordinates be the same, and observe the difference in the output $x$-coordinates:

```{code-cell} julia
v0 = repeat([1e-3 0 0 0 0 1e-3], 10, 1)
res = track(fodo, v0=v0)
res.v[:,1,end] # output x-coordinates
```

In one parallelized tracking run, we have effectively simulated 10 different beamlines. This
functionality is extremely powerful, as it enables us to do parameter scans *within a single
process!* In fact, SciBmad's tracking kernels have been carefully developed so that single
instruction, multiple data parallelization is still applied over different parameters per
particle.

Let's say we want to track 2 particles in each of our 10 different FODO cells instead.
Because each particle in a bunch sees a corresponding parameter at the same index in the
`BatchParam`, we need to initialize the `BatchParam` so that each quadrupole strength is
repeated twice:

```{code-cell} julia
n_particles = 2
fodo.context.K1 = BatchParam(repeat(collect(0.1:0.05:0.55), inner=n_particles))
```

And `track` again:

```{code-cell} julia
v0 = repeat([1e-3 0 0 0 0 1e-3], n_particles*10, 1)
res = track(fodo, v0=v0)
res.v[:,1,end] # output x-coordinates
```

## GPU Batch Parameter Simulation

In the same way that a GPU enables parallelization over a huge number of particles, we can
GPU-parallelize over a huge number of parameters with `BatchParam`s. Simply set the
parameter to be a `GPUArray`, just as we do with the particle phase space coordinates:

```julia
using CUDA

# (100 quad strengths x 1,000 particles per quad strength) = 100,000 particles tracked in total
n_particles_per_beamline = 1000
n_quad_strengths = 100

fodo.context.K1 = CuArray(range(start=0.1, stop=0.55, length=n_quad_strengths))

# Random initial phase space coordinates:
v0 = CUDA.rand(Float64, n_particles_per_beamline * n_quad_strengths, 6) .* 1e-5

res = track(fodo, v0=v0, spin=true)
```

