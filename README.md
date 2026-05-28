# SciBmad
<!--
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://bmad-sim.github.io/SciBmad.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://bmad-sim.github.io/SciBmad.jl/dev/)
[![Build Status](https://github.com/bmad-sim/SciBmad.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/bmad-sim/SciBmad.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/bmad-sim/SciBmad.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/bmad-sim/SciBmad.jl)
!-->

[Paper](https://github.com/bmad-sim/SciBmad.jl/blob/main/paper/THP5325.pdf), [Slides](https://github.com/user-attachments/files/25094046/scibmad-eic-02-02-2026.pdf),    [Examples](https://github.com/bmad-sim/SciBmad.jl/tree/main/examples)

## Overview

SciBmad is a new open source, high-performance, CPU/GPU compatible, polymorphic, and forwards-/backwards-/Taylor-differentiable accelerator physics simulation ecosystem, usable within either Python and Julia.

## Installation instructions

SciBmad is compatible with [Windows](https://github.com/bmad-sim/SciBmad.jl/blob/main/WINDOWS.md), [Mac](https://github.com/bmad-sim/SciBmad.jl/blob/main/MAC.md), or [Linux](https://github.com/bmad-sim/SciBmad.jl/blob/main/LINUX.md). Click on your corresponding system to be linked to detailed installation instructions.


## Examples

Users are pointed to example Jupyter notebooks in both Julia and Python in the [examples directory](https://github.com/bmad-sim/SciBmad.jl/tree/main/examples).


## Project Status

SciBmad development is progressing rapidly. Features included in the current state of the project (0.4.0), the next release (0.4.1), and currently being planned for later releases are:

### Current Release 0.4.0
- CPU/GPU parallelized 6D symplectic particle tracking including spin and radiation
- Fully forwards-/backwards-/Taylor differentiable to extract gradients w.r.t. anything
- Taylor series nonlinear normal form analysis (i.e. nonlinear periodic Twiss functions) including spin and radiation
- Arbitrary time-dependent accelerator parameters (e.g. magnet strengths, misalignments)
- CPU/GPU parallelized tracking and analysis over differing accelerator parameters (batch parameter evaluation)
- Arbitrarily-interdependent accelerator parameters with lazily-evaluated deferred expressions
- CPU/GPU parallelized dynamic aperture scans
- Arbitrary placements and orientations of accelerator elements
- CPU/GPU parallelized intrabeam scattering (IBS)
- CPU/GPU parallelized Numerical Analysis of Fundamental Frequencies (NAFF)
- CPU/GPU parallelized Newton root finder
- CPU/GPU parallelized and differentiable symplectic tracking through arbitrary electromagnetic fields (implicit integration)

### Next Release 0.4.1
- Twiss functions at every integration step (inside elements)
- Resonance driving terms (including parameter dependence) included in Twiss

### Release 0.4.2
- Open lattice Twiss function propagation

### Future Releases
- Generalized gradient field description
- Bindings to [WarpX](https://github.com/BLAST-WarpX/warpx) for collective effects (e.g. strogn-strong beam beam)
- Weak-strong symplectic beam-beam interaction
- PyTorch bindings
- Exact multipoles in curved coordinate systems
- Electric multipoles
- Space charge
- Wakefields
- Coherent synchrotron radiation

## SciBmad Family

SciBmad consists of a set of modular packages:

- **[`BeamTracking.jl`](https://github.com/bmad-sim/BeamTracking.jl):** Universally polymorphic, differentiable, portable, and parallelized integrators for simulating charged particle beams on the CPU and various GPUs including NVIDIA CUDA, Apple Metal, Intel oneAPI, and AMD ROCm
- **[`GTPSA.jl`](https://github.com/bmad-sim/GTPSA.jl):** Fast high-order (Taylor mode) automatic differentiation using the Generalised Truncated Power Series Algebra (GTPSA) library
- **[`Beamlines.jl`](https://github.com/bmad-sim/Beamlines.jl):** Defines accelerator lattices in a fast, flexible, fully-polymorphic, and differentiable way, providing both eagerly- and lazily-evaluated deferred expressions for interdependent parameters
- **[`NonlinearNormalForm.jl`](https://github.com/bmad-sim/NonlinearNormalForm.jl):** Map-based perturbation theory of differential-algebraic maps, which may include spin and large damping, using Lie algebraic methods
- **[`FundamentalFrequencies.jl`](https://github.com/bmad-sim/FundamentalFrequencies.jl):** GPU-batchable Numerical Analysis of Fundamental Frequencies (NAFF)
- **[`AtomicAndPhysicalConstants.jl`](https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl):** Library providing physical constants and properties for any atomic or subatomic particle for use in simulations

