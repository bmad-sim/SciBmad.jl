# SciBmad
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://bmad-sim.github.io/SciBmad.jl/stable/)
<!---
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://bmad-sim.github.io/SciBmad.jl/dev/)
[![Build Status](https://github.com/bmad-sim/SciBmad.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/bmad-sim/SciBmad.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/bmad-sim/SciBmad.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/bmad-sim/SciBmad.jl)!-->

<!--
, [Slides](https://github.com/user-attachments/files/25094046/scibmad-eic-02-02-2026.pdf),    [Examples](https://github.com/bmad-sim/SciBmad.jl/tree/main/examples)
!-->
## Overview

SciBmad is a new open source, high-performance, CPU/GPU compatible, polymorphic, and forwards-/backwards-/Taylor-differentiable accelerator physics simulation ecosystem.

## Project Status

SciBmad development is progressing rapidly. Features included in the current state of the project (0.5.0) and currently being planned for later releases are:

### Current Release 0.5.0
- CPU/GPU parallelized 6D symplectic particle tracking including spin and radiation
- Fully forwards-/backwards-/Taylor differentiable to extract gradients w.r.t. anything
- Taylor series nonlinear normal form analysis (i.e. nonlinear periodic Twiss functions) including spin and radiation
- Arbitrary time-dependent accelerator parameters (e.g. magnet strengths, reference energy, misalignments)
- CPU/GPU parallelized tracking and analysis over differing accelerator parameters (batch parameter evaluation)
- Arbitrarily-interdependent accelerator parameters with lazily-evaluated deferred expressions
- CPU/GPU parallelized dynamic aperture scans
- Arbitrary placements and orientations of accelerator elements
- CPU/GPU parallelized intrabeam scattering (IBS)
- CPU/GPU parallelized Numerical Analysis of Fundamental Frequencies (NAFF)
- CPU/GPU parallelized Newton root finder
- CPU/GPU parallelized and differentiable symplectic tracking through arbitrary electromagnetic fields (implicit integration)
- Twiss functions at every integration step (inside elements)
- Resonance driving terms (including parameter dependence) included in Twiss
- Context-switching for evalution of deferred expressions for accelerator parameters
- Open lattice Twiss functions given initial normalizing transformation

### Future Releases
- Easy optimization interface
- Generalized gradient field description
- Wakefields
- Bindings to [WarpX](https://github.com/BLAST-WarpX/warpx) for collective effects (e.g. strong-strong beam beam)
- Weak-strong symplectic beam-beam interaction
- PyTorch bindings
- Exact multipoles in curved coordinate systems
- Electric multipoles
- Space charge
- Coherent synchrotron radiation

## SciBmad Family

SciBmad consists of a set of modular packages:

- **[`BeamTracking`](https://github.com/bmad-sim/BeamTracking.jl):** Universally polymorphic, differentiable, portable, and parallelized integrators for simulating charged particle beams on the CPU and various GPUs including NVIDIA CUDA, Apple Metal, Intel oneAPI, and AMD ROCm
- **[`GTPSA`](https://github.com/bmad-sim/GTPSA.jl):** Fast high-order (Taylor mode) automatic differentiation using the Generalised Truncated Power Series Algebra (GTPSA) library
- **[`Beamlines`](https://github.com/bmad-sim/Beamlines.jl):** Defines accelerator lattices in a fast, flexible, fully-polymorphic, and differentiable way, providing both eagerly- and lazily-evaluated deferred expressions for interdependent parameters
- **[`NonlinearNormalForm`](https://github.com/bmad-sim/NonlinearNormalForm.jl):** Map-based perturbation theory of differential-algebraic maps, which may include spin and large damping, using Lie algebraic methods
- **[`FundamentalFrequencies`](https://github.com/bmad-sim/FundamentalFrequencies.jl):** GPU-batchable Numerical Analysis of Fundamental Frequencies (NAFF)
- **[`AtomicAndPhysicalConstants`](https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl):** Library providing physical constants and properties for any atomic or subatomic particle for use in simulations

## Paper and Citation

[Paper](https://github.com/bmad-sim/SciBmad.jl/blob/main/paper/THP5325.pdf)

If you find SciBmad useful in your work, please cite this paper:

```
@inproceedings{signorelli:ipac2026-thp5325,
    author = {M. G. Signorelli and J. Devlin and G. H. Hoffstaetter and D. Sagan},
    title = {SciBmad: A differentiable, GPU-parallelized software library for particle accelerator design, nonlinear analysis, and machine learning},
    booktitle = {Proc. IPAC'26},
    %  booktitle = {Proc. 17th International Particle Accelerator Conference},
    pages = {4737-4740},
    paper = {THP5325},
    venue = {Deauville, France},
    intype = {presented at IPAC'26},
    series = {IPAC'26 - 17th International Particle Accelerator Conference},
    number = {17},
    publisher = {JACoW Publishing, Geneva, Switzerland},
    month = {05},
    year = {2026},
    issn = {2673-5350},
    isbn = {978-3-95450-252-3},
    url = {https://jacow.org/ipac2026/pdf/THP5325.pdf},
    note = {presented at IPAC'26, Deauville, France, 2026, paper THP5325},
    language = {English},
    eventdate = {2026-05-17/2026-05-22},
}
```