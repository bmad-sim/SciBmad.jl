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

(optimize)=
# Optimization with Autodiff

```{code-cell} julia
:tags: [remove-cell]
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

When optimizing quantities such as beta functions, tunes, chromaticities, resonance driving
terms, etc., many optimizers require gradients of these quantities w.r.t. the knobs you are
turning to achieve the goal. In non-differentiable codes, these quantities must be computed
with finite differences, which has errors that grow exponentially with the order of
derivative you are computing. For example, to optimize the resonance driving term
$h_{2020}$ requires 3rd order derivatives in the phase space variables, and to see how this
quantity varies with a parameter you have to add another order, so 4th order. Finite
differencing may struggle with such an optimization. However, because SciBmad is fully
differentiable, we can compute such a derivative _exactly!_

This section assumes that you have gone through the
[Parametric Normal Form](parametric-nf.md) section of the documentation and have basic
familiarity with [`GTPSA`](https://github.com/bmad-sim/GTPSA.jl).

Let's optimize the phase advance through a FODO cell to be $90^\circ$ in both planes by
varying the quadrupoles, and then optimize the chromaticity to be $+1$ in both planes by
varying the sextupoles. Per the [Parametric Normal Form](parametric-nf.md) section, we know
how to compute these quantities and their gradients w.r.t. such parameters. Thus, all we
need now is an optimization algorithm.

Julia has a plethora of optimization packages/ecosystems with different optimizers. Some
various options available are:

- [`NonlinearSolve`](https://docs.sciml.ai/NonlinearSolve/stable/): For root-finding and solving (in a least squares sense) nonlinear systems
- [`Optimization`](https://docs.sciml.ai/Optimization/stable/): For minimizing scalar-valued functions, a unified interface to many optimizers in Julia (heavyweight package)
- [`Optim`](https://github.com/JuliaNLSolvers/Optim.jl): Lightweight package for finding local minima of scalar-valued functions
- [`Metaheuristics`](https://docs.sciml.ai/Optimization/stable/): Global, multi-objective optimization algorithms
- [`JuMP`](https://jump.dev/): A modeling language/ecosystem for optimizing many different problem classes
- [`BatchSolve`](https://github.com/mattsignorelli/BatchSolve.jl): Lightweight package including a Newton root finder and Brent's optimizer, optionally GPU-batchable with CUDA

In this example we will use `NewtonRaphson` in `NonlinearSolve`. Make sure to add it using

```julia
import Pkg; Pkg.add("NonlinearSolve")
```

We start with our FODO cell

```{code-cell} julia
using SciBmad
using NonlinearSolve

@elements begin
  qf = Quadrupole(Kn1=DefExpr(c -> c.kqf), L=0.5)
  sf = Sextupole(Kn2=DefExpr(c-> c.ksf), L=0.2)
  d = Drift(L=0.2)
  b = SBend(L=6.5, angle=pi/132)
  qd = Quadrupole(Kn1=DefExpr(c -> c.kqd), L=0.5)
  sd = Sextupole(Kn2=DefExpr(c -> c.ksd), L=0.2)
end

fodo = Beamline([qf, sf, d, b, d, qd, sd, d, b, d],
        species_ref=Species("electron"), pc_ref=18e9)

# Initial values
fodo.context.kqf = 0.36
fodo.context.kqd = -0.36
fodo.context.ksf = 1.2
fodo.context.ksd = -1.2
```

We now want to vary `kqf` and `kqd` so that the tunes are $90^\circ$. To do this with
`NonlinearSolve`, we need to write two functions: one that returns the residual vector of
the tunes given some `u = [kqf, kqd]`, and another that returns the Jacobian of the tunes
w.r.t. `kqf` and `kqd` given some `u = [kqf, kqd]`. We'll let both functions receive a
second positional argument that is the beamline itself. Note that, since we only compute the
tunes, we can pass an empty array for the `at` keyword argument to `twiss` to save
computation time.

```{code-cell} julia
function res_tunes(u, beamline)
  beamline.context.kqf = u[1]
  beamline.context.kqd = u[2]
  tw = twiss(beamline, at=[])
  return [tw.q1 - 0.25, tw.q2 - 0.25]
end

function jac_tunes(u, beamline)
  dnf = Descriptor([1, 1, 1, 1, 1, 1], 2, [1, 1], 1)
  dk = params(dnf)
  beamline.context.kqf = u[1] + dk[1]
  beamline.context.kqd = u[2] + dk[2]
  tw = twiss(beamline, at=[], GTPSA_descriptor=dnf)
  scalarize!(beamline)
  return jac([tw.q1, tw.q2])
end
```

Now we follow the machinery of `NonlinearSolve` - that is, construct a `NonlinearFunction`
(providing the Jacobian function with keyword argument `jac`), then a `NonlinearProblem`
passing our initial guess, and then `solve`:

```{code-cell} julia
f = NonlinearFunction(res_tunes, jac=jac_tunes)
u0 = [0.36, -0.36]
prob = NonlinearProblem(f, u0, fodo)
sol = solve(prob, NewtonRaphson())
```

Now when we call `twiss` again, we see that our tunes have been made `0.25`:

```{code-cell} julia
twiss(fodo)
```

Great! Now let's check the chromaticities:

```{code-cell} julia
twiss(fodo, chrom=2)
```

Both are not quite `+1`. Let's optimize the sextupoles now to make them both `+1`. Following
the same procedure,

```{code-cell} julia
function res_chrom(u, beamline)
  beamline.context.ksf = u[1]
  beamline.context.ksd = u[2]
  tw = twiss(beamline, at=[], chrom=2)
  chromx = getterm(tw.q1, delta=1)
  chromy = getterm(tw.q2, delta=1)
  return [chromx - 1, chromy - 1]
end

function jac_chrom(u, beamline)
  dnf = Descriptor([1, 1, 1, 1, 1, 2], 3, [1, 1], 1)
  dk = params(dnf)
  beamline.context.ksf = u[1] + dk[1]
  beamline.context.ksd = u[2] + dk[2]
  tw = twiss(beamline, at=[], GTPSA_descriptor=dnf)
  scalarize!(beamline)
  chromx = getterm(tw.q1, delta=1, as_taylor_series=true)
  chromy = getterm(tw.q2, delta=1, as_taylor_series=true)
  return jac([chromx, chromy])
end

f2 = NonlinearFunction(res_chrom, jac=jac_chrom)
u0 = [1., -1.]
prob2 = NonlinearProblem(f2, u0, fodo)
sol2 = solve(prob2, NewtonRaphson())
```

And now:

```{code-cell} julia
twiss(fodo, chrom=2)
```

Perfect. The same procedure can be applied to any column outputted from `twiss` at any
element. For open-lattice matching, currently `a_initial` must be provided to `twiss`.
