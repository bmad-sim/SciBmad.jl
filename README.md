# SciBmad
<!--
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://bmad-sim.github.io/SciBmad.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://bmad-sim.github.io/SciBmad.jl/dev/)
[![Build Status](https://github.com/bmad-sim/SciBmad.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/bmad-sim/SciBmad.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/bmad-sim/SciBmad.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/bmad-sim/SciBmad.jl)
!-->

SciBmad is a new open source, high-performance, polymorphic, and differentiable accelerator physics simulation ecosystem, consisting of a set of modular packages:

- **[`BeamTracking.jl`](https://github.com/bmad-sim/BeamTracking.jl):** Universally polymorphic, differentiable, portable, and parallelized integrators for simulating charged particle beams on the CPU and various GPUs including NVIDIA CUDA, Apple Metal, Intel oneAPI, and AMD ROCm
- **[`GTPSA.jl`](https://github.com/bmad-sim/GTPSA.jl):** Fast high-order (Taylor mode) automatic differentiation using the Generalised Truncated Power Series Algebra (GTPSA) library
- **[`Beamlines.jl`](https://github.com/bmad-sim/Beamlines.jl):** Defines advanced accelerator lattices in a fast, flexible, fully-polymorphic, and differentiable way, providing both eagerly- and lazily-evaluated deferred expressions for interdependent parameters
- **[`NonlinearNormalForm.jl`](https://github.com/bmad-sim/NonlinearNormalForm.jl):** Map-based perturbation theory of differential-algebraic maps, which may include spin and large damping, using Lie algebraic methods
- **[`AtomicAndPhysicalConstants.jl`](https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl):** Library providing physical constants and properties for any atomic or subatomic particle, in units chosen by the user, for use in simulations


## Examples

Users are pointed to example Jupyter notebooks in both Julia and Python in the [examples directory](https://github.com/bmad-sim/SciBmad.jl/tree/main/examples).

