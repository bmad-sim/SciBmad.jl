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

```{code-cell} python
:tags: [remove-cell]
import numpy as np
import scibmad as sb
sb.options.display_width = 100
sb.options.display_height = 30
```

The `find_closed_orbit` function can be used to find the fixed point of a periodic beamline. It uses 
the Newton  method implemented in the [`BatchSolve`](https://github.com/mattsignorelli/BatchSolve.jl) 
package. By default, automatic differentiation via [`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl) 
is used to compute the Jacobian(s) each iteration.

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

```{code-cell} python
with sb.elements():
    qf = sb.Quadrupole(Kn1=0.36, L=0.5)
    d = sb.Drift(L=1.2)
    qd = sb.Quadrupole(Kn1=-0.36, L=0.5)
    kick = sb.HKicker(Kn0L=1e-5)

fodo = sb.Beamline([qf, d, qd, d, kick],
                   species_ref=sb.Species("electron"), E_ref=18e9)  # some Beamline
co_sol = sb.find_closed_orbit(fodo)
```

A `NamedTuple` is returned containing the properties:

- `v0`: The closed orbit
- `coasting_beam`: `true` if no longitudinal oscillations (4D closed orbit), `false` otherwise
- `sol`: Another `NamedTuple` with details of the solution from `BatchSolve`:
  - `u`: The coordinates the Newton method solved for: all six, or for a coasting beam only the
    four transverse ones (δ is fixed), one row per particle.
  - `f`: The final residual: the change of these coordinates over one turn, which is zero on the
    closed orbit.
  - `jac`: The Jacobian of the residual at the last iteration (of all particles together).
  - `retcode`: `0x0` if the solution converged, `0x1` if the solver failed, `0x2` if it reached the
    maximum number of iterations (one per particle with several particles).
  - `iters`: The number of iterations taken (one per particle with several particles).

An initial "guess" for the closed orbit can be provided by specifying a `1 x 6` matrix of
particle phase space coordinates to the keyword argument `v0`. By default, it is
`zeros(1, 6)`.

```{code-cell} julia
co_sol = find_closed_orbit(fodo; v0=[1e-5 2e-6 -1e-5 -2e-6 0.0 1e-5])
co_sol.v0
```

```{code-cell} python
co_sol = sb.find_closed_orbit(fodo, v0=[[1e-5, 2e-6, -1e-5, -2e-6, 0.0, 1e-5]])
co_sol.v0
```

This beamline has no RF, so the beam is coasting (`co_sol.coasting_beam` is `true`), and the
last coordinate δ of the guess did not change: the result is the closed orbit for δ = 1e-5
(see δ-Dependent Closed Orbits below).

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

```{code-cell} python
# To get delta-dependent closed orbits:
v0 = [[0, 0, 0, 0, 0, 0.1e-2],  # δ = 0.1e-2
      [0, 0, 0, 0, 0, 0.2e-2]]  # δ = 0.2e-2

co_sol = sb.find_closed_orbit(fodo, v0=v0)
co_sol.v0
```

If `v0` is a CUDA array type `CuArray`{pytext}` (sb.device("cuda").asarray(...) in Python)`, then
the Newton method used to find the closed
orbits will be GPU-parallelized, with bindings directly to cuBLAS's batched linear system
solvers.

```julia
using CUDA
n_particles = 100000
v0 = CUDA.zeros(Float64, n_particles, 6)
v0[:,6] .= CuArray(range(start=0, stop=1e-2, length=n_particles))

co_sol = find_closed_orbit(fodo, v0=v0)
```

```python
gpu = sb.device("cuda")
n_particles = 100000
v0 = np.zeros((n_particles, 6))
v0[:, 5] = np.linspace(0, 1e-2, n_particles)

co_sol = sb.find_closed_orbit(fodo, v0=gpu.asarray(v0))
```

## Closed Orbits with Batch Parameters

If any element parameters are defined with [`BatchParam`s](batch.md), then as with `track`,
each particle in the initial guess matrix will see that corresponding parameter (with a guess of
one row, the default `v0`, the first value). Element lengths and the reference energy can be
batched too.

Below, we first find two closed orbits, one for `kick.Kn0L=1e-5` and one for `kick.Kn0L=2e-5`,
and then the same with `δ = 0.1e-2` for the first and `δ = 0.2e-2` for the second:

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

```{code-cell} python
# With BatchParams
kick.Kn0L = sb.BatchParam([1e-5, 2e-5])
v0 = np.zeros((2, 6))
co_sol = sb.find_closed_orbit(fodo, v0=v0)

# BatchParams + delta-dependent closed orbits:
v0 = [[0, 0, 0, 0, 0, 0.1e-2],  # δ = 0.1e-2
      [0, 0, 0, 0, 0, 0.2e-2]]  # δ = 0.2e-2
co_sol = sb.find_closed_orbit(fodo, v0=v0)
co_sol
```

Again, by making these `CuArray`s, the closed orbit finding will be GPU-parallelized.

```julia
using CUDA

n_kick_strengths = 100
n_deltas = 1000
n_particles = n_kick_strengths * n_deltas
v0 = CUDA.zeros(Float64, n_particles, 6)
deltas = repeat(range(start=0, stop=1e-2, length=n_deltas), outer=n_kick_strengths)
v0[:,6] .= CuArray(deltas)

kick_strengths = repeat(range(start=1e-5, stop=2e-5, length=n_kick_strengths), inner=n_deltas)
kick.Kn0L = BatchParam(CuArray(kick_strengths))
co_sol = find_closed_orbit(fodo, v0=v0)
```

```python
gpu = sb.device("cuda")

n_kick_strengths = 100
n_deltas = 1000
n_particles = n_kick_strengths * n_deltas
v0 = np.zeros((n_particles, 6))
v0[:, 5] = np.tile(np.linspace(0, 1e-2, n_deltas), n_kick_strengths)

kick_strengths = np.repeat(np.linspace(1e-5, 2e-5, n_kick_strengths), n_deltas)
kick.Kn0L = sb.BatchParam(gpu.asarray(kick_strengths))
co_sol = sb.find_closed_orbit(fodo, v0=gpu.asarray(v0))
```

```{docstring} find_closed_orbit
```
