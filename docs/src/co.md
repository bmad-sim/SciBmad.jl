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

(closedorbit)=
# (GPU-)Batched Closed Orbit Finder

```{code-cell} julia
:tags: [remove-cell]
using SciBmad
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

The `find_closed_orbit` function can be used to find the fixed point of a periodic beamline.
It uses the Newton method implemented in the
[`BatchSolve`](https://github.com/mattsignorelli/BatchSolve.jl) package. By default,
automatic differentiation via
[`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl) is used to compute the
Jacobian(s) each iteration.

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  d = Drift(L=1.2)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  kick = HKicker(Kn0L=1e-5)
end

fodo = Beamline([qf, d, qd, d, kick],
        species_ref=Species("electron"), E_ref=18e9) # some Beamline
co_sol = find_closed_orbit(fodo);
```

A `NamedTuple` is returned containing the properties:

- `v0`: The closed orbit
- `coasting_beam`: `true` if no longitudinal oscillations (4D closed orbit), `false` otherwise
- `sol`: Another `NamedTuple` of the solution, outputted from `BatchSolve`, which contains:
  - `u`: Solution array (same object as `x`, mutated in-place).
  - `f`: Final residual vector (same object as `y`, mutated in-place).
  - `jac`: Final Jacobian.
  - `retcode`: `RETCODE_SUCCESS = 0x0`, `RETCODE_FAILURE = 0x1`, or `RETCODE_MAXITER = 0x2`.
  - `iters`: Number of iterations taken (scalar, or array when `batchdim` is set).

An initial "guess" for the closed orbit can be provided by specifying a `1 x 6` matrix of
particle phase space coordinates to the keyword argument `v0`. By default, it is
`zeros(1, 6)`.

```{code-cell} julia
co_sol = find_closed_orbit(fodo; v0=rand(1, 6) .* 1e-5)
```

:::{note}
When `v0` is provided, it **will be mutated in place with the result!**
:::

## δ-Dependent Closed Orbits

For beamlines with coasting beam, δ-dependent closed orbits can be computed. To do this, we
set `v0` equal to a matrix of `n_particles x 6`:

```{code-cell} julia
# To get delta-dependent closed orbits:
v0 = [0. 0. 0. 0. 0. 0.1e-2; # δ = 0.1e-2
      0. 0. 0. 0. 0. 0.2e-2] # δ = 0.2e-2

co_sol = find_closed_orbit(fodo, v0=v0)
co_sol.v0
```

If `v0` is a CUDA array type `CuArray`, then the Newton method used to find the closed
orbits will be GPU-parallelized, with bindings directly to cuBLAS's batched linear system
solvers.

```julia
using CUDA
n_particles = 100000
v0 = CUDA.zeros(n_particles, 6)
v0[:,6] .= CuArray(range(start=0, stop=1e-2, length=n_particles))

co_sol = find_closed_orbit(fodo, v0=v0)
```

## Closed Orbits with Batch Parameters

If any element parameters are defined with [`BatchParam`s](batch.md), then as with `track`,
each particle in the initial guess matrix will see that corresponding parameter.

Below, we find two closed orbits: one where `kick.Kn0L=1e-5` and `δ = 0.1e-2`, and another
where `kick.Kn0L=2e-5` and `δ = 0.2e-2`:

```{code-cell} julia
# With BatchParams
kick.Kn0L = BatchParam([1e-5, 2e-5])
v0 = zeros(2, 6)
co_sol = find_closed_orbit(fodo, v0=v0)

# BatchParams + delta-dependent closed orbits:
v0 = [0. 0. 0. 0. 0. 0.1e-2; # δ = 0.1e-2
      0. 0. 0. 0. 0. 0.2e-2] # δ = 0.2e-2
co_sol = find_closed_orbit(fodo, v0=v0)
```

Again, by making these `CuArray`s, the closed orbit finding will be GPU-parallelized.

```julia
using CUDA

n_kick_strengths = 100
n_deltas = 1000
n_particles = n_kick_strengths * n_deltas
v0 = CUDA.zeros(n_particles, 6)
deltas = repeat(range(start=0, stop=1e-2, length=n_deltas), outer=n_kick_strengths)
v0[:,6] .= CuArray(deltas)

kick_strengths = repeat(range(start=1e-5, stop=2e-5, length=n_kick_strengths), inner=n_deltas)
kick.Kn0L = BatchParam(CuArray(kick_strengths))
co_sol = find_closed_orbit(fodo, v0=v0)
```

:::{seealso}
The full `find_closed_orbit` docstring is in the {external:doc}`API Reference <index>`.
:::
