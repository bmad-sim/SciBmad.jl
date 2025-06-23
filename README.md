# SciBmad

## Overview

`SciBmad` is a new, open-source project to develop, in
the Julia language, a set of modular packages providing the
fundamental tools and methods commonly needed for accelerator simulations. By avoiding the necessity of ”reinventing
the wheel”, simulation programs can be developed in less
time and with fewer bugs than developing from scratch. 

A modern simulation framework such as SciBmad is greatly
needed since the ever increasing demands placed upon machine performance require ever more comprehensive accelerator modeling.
In addition to the toolkit packages, the SciBmad project
will develop accelerator simulation programs for various
simulation tasks. Initial program development will focus on
Machine Learning / Artificial Intelligence (ML/AI) applications. However, SciBmad will be applicable to accelerator
simulations in general.

## SciBmad.jl Repo

This `SciBmad.jl` repository is just meant as a landing page for the SciBmad project. 
Code development is done in other repositories in the GitHub [Bmad Consortium](https://github.com/bmad-sim) area.

## Motivation

The disorganized state of accelerator physics simulation development has been 
recognized by the latest High-Energy Physics community Snowmass white paper~\SciBmade{biedron2022}:
(DOE-sponsored Snowmass reports outline the major funding priorities for the HEP community)
> The development of beam and accelerator physics codes has often been largely uncoordinated. This comes at a great cost, is not desirable and may not be tenable. 
>
> Due to developers retiring or moving on to other projects, numerous simulation programs have been completely abandoned or are seldom used.
> This has resulted in a collection of codes that are not interoperable, use different I/O formats and quite often duplicate some physics functionalities using the exact same underlying algorithms. 
> Frequently there is a huge impediment to maintaining these programs due to poorly-written code and lack of documentation. Additionally, many of the programs that are available tend to be ``rigid''. That is, it is generally difficult to modify a program to simulate something it is not designed to simulate *a priori*. Adding a new type of lattice element that a particle can be tracked through is one such example.
As a consequence, the Snowmass white paper makes a recommendation on how to ameliorate the situation:
> **Recommendation on community ecosystems \& data repositories**
> Organize the beam and accelerator modeling tools and community through the development of (a) ecosystems of codes, libraries and frameworks that are interoperable via open community data standards, (b) open access data repositories for reuse and community surrogate model training, (c) dedicated Centers and distributed consortia with open community governance models and dedicated personnel to engage in cross-organization and -industry development, standardization, application and evaluation of accelerator and beam modeling software and data.

The Bmad-Julia project is structured to be part of this ecosystem. 
The Bmad-Julia project was started last year (2023) and
aims to develop a set
of open source packages (code modules), written in the Julia programming language, that can
serve as the basis for future accelerator simulation programs. 
This will enable simulation programs to be developed
in less time and with fewer bugs (due to code reuse) than can be done for a program developed from scratch. 
Along with the packages, programs will be developed for various simulation
tasks. The first applications will involve Machine Learning / Artificial
Intelligence (ML/AI) but Bmad-Julia will not be limited to ML/AI applications.

## SciBmad Project Goals

The general goals of the SciBmad project are (a) to develop an ecosystem of modular and extensible packages that
allows for the easy construction
of accelerator simulation programs, and (b) to use these packages to develop simulation programs. Initially, there will be a strong ML/AI orientation.

The success and sustainability of SciBmad will depend heavily on the involvement of the entire
accelerator physics community. To this end, the SciBmad project seeks community engagement with weekly planning meetings, open to all,
and with a SLACK workspace where development can be discussed. Additionally, regular workshops and other informational meetings
will be scheduled as the project advances.

In the short-term, needed are packages for defining and manipulating lattices, 
tracking of particles, Truncated Power Series Algebra (TPSA), differential algebra maps, 
and normal form decomposition and analysis of differential algebra maps to extract such things as emittances, Twiss parameters, 
resonance strengths, etc.
In cases where packages external to SciBmad have the needed functionality, rather than reinvent the wheel, 
appropriate interfaces will be developed.

Experience with Bmad has shown that there is no ideal way to track through a lattice with
different tracking methods, all having their own strengths and weaknesses. SciBmad
will thus include multiple tracking packages. Considerations in developing
these packages include speed, accuracy, and the ability to simulate various models of machine components. Needed is 
software that has the ability (not necessarily at the same time) to track TPSA maps, track using GPUs, 
track with backwards differentiation, etc. 

## Packages

- **AcceleratorLattice.jl:** Package for accelerator lattice
construction and manipulation. Lattice elements are Julia
structs which have a Dict component that can store arbitrary
information which is important for flexibility.

- **AtomicAndPhysicalConstants.jl:** Library of atomic
and subatomic particle properties and other physical constants (speed of light, etc.).
The package has Particle structs
for defining particles such as Helium-3+ and positrons. The
units used are flexible and can be set differently in different
packages that use this package. EG: Units of mass may be
set to MeV/c2, AMU, or something else as desired.

- **GTPSA.jl:** Full-featured Julia interface to the Generalized Truncated Power Series Algebra (GTPSA)
library developed by Laurent Deniau. GTPSA performs automatic
differentiation (AD) of real and complex multi-variable functions to arbitrary orders.
GTPSA.jl is significantly faster than
other Julia AD packages for calculating derivatives above
first-order.

- **SimUtils.jl:** Simulation utility routines for needed
functionality that does not exist in any external packages.
Except for the AcceleratorLattice package, all the packages currently being developed are useful in fields outside
of accelerator physics from cosmology to chemistry.

- **NonlinearNormalForm.jl:** Nonlinear normal form
analysis using differential algebra maps including spin. Included are methods for calculating parameter-dependent nor-
mal forms using vector fields (Lie operators), operations including vector fields and maps (e.g. logarithms of maps, Lie
brackets, etc.), map inversion/partial inversion, and more.

- **BeamTracking:** This package provides universally polymorphic and fully portable, parallelizable routines for simulating charged particle beams both on the CPU and, using KernelAbstractions.jl, various GPU backends including NVIDIA CUDA, Apple Metal, Intel oneAPI, and AMD ROCm.

## Machine Learning

Machine Learning (ML) and Artificial Intelligence (AI) are having a large impact on accelerator operations and optimizations.
ML/AI can provide advanced optimization tools that are uniquely suited
for many accelerator tasks, e.g., physics-informed Bayesian Optimization
addresses tasks where measurement uncertainties are relevant
or where obtaining measurements is time-consuming or expensive.
Additionally, the construction of ML-based surrogate models
has shown powerful speedups when describing hard-to-model sections
of accelerators, e.g., for space charge-dominated beams.

Despite this, no comprehensive accelerator simulation toolkit has been designed with ML/AI in focus. The development of SciBmad will address this.
SciBmad will provide a maximally-differentiable simulation and lattice design environment, with the full-features of present Bmad and more. This enables machine learning techniques (e.g. backwards differentiation to train ML models) to be employed naturally, as well as any optimization techniques utilizing automatic differentiation for lattice design purposes; Julia's entire ecosystem of optimizers and machine learning packages are at one's fingertips in SciBmad. Alternatively, Julia provides bindings to existing Python ML/AI packages such as OpenAI Gym and PyTorch allowing for their
utilization as well.

Computation speed is critical for ML/AI applications as well as for
other accelerator-based simulations.
Hence, there will be a heavy focus on developing GPU compatible code along with multi-threading and multi-processing capabilities.

## Interoperability

Bmad-Julia is being developed with consideration for compatibility with the wider accelerator modeling community. This includes data standards and lattice description. 
This will facilitate cross-checking and benchmarking results, and enable integration between toolkits. This is important for start-to-end simulations, integration with external optimization tools, and seamless training of machine learning models.
For example, routines for reading and writing particle distributions using the openPMD standard will be developed and
Bmad-Julia will leverage prior investment in a scalable data stack that was developed for the US Exascale Computing Project (ECP).

Translation software will be developed to translate between the Bmad-Julia lattice format and the existing
Bmad format, as well as other commonly used formats (MAD, etc.). Since Bmad-Julia will encompass the major features found in Bmad, translation from Bmad to Bmad-Julia should be close
to 100%. Back translation to Bmad will be fairly good except in a few areas such as control element descriptions since Bmad-Julia has a more expressive syntax in this area.

For translation to non-Bmad lattice formats,
translation will never be perfect (far from it) since
different programs have different element types and differing constructs. Yet
experience has shown that just being able to translate basic information,
like type of element lengths and strengths, is very helpful. 

For portability for programs that do not use Julia,
or do not use the Bmad-Julia lattice parsing package, 
a standard lattice syntax will be developed using a standard data format (EG: YAML, TOML, JSON). Along
with this, standardized parameter names and meanings will be established.
The lattice syntax will be extensible
in the sense that the syntax will allow custom information to be stored along side the information
defined by the standard. It is envisaged that this portable lattice standard could serve as 
a lingua franca for lattice communication among non Bmad-Julia programs.

## Based on Bmad

SciBmad is inspired by the current Bmad ecosystem of toolkits and programs
for the simulation of relativistic charged particles and X-rays in accelerators and storage rings. Bmad has a wide range of capabilities including tracking of polarized beams and X-rays, while simulating coherent synchrotron radiation (CSR), wakefields, Touschek scattering,
higher order mode (HOM) resonances, space-charge dominated
beams, weak-strong beam-beam interactions, and much more. 

Although the SciBmad code will be completely separate from the existing
Bmad code, the development of SciBmad will
rely heavily on the many years of experience the developers have developing Bmad.
Concepts developed in Bmad will serve as a paradigm for the present effort.

It should be noted that, due to Bmad's wide adoption and breadth of capabilities,  
for the foreseeable future, SciBmad will not be a replacement for Bmad,
and both will co-exist side by side.

## Why Julia?

In modern AI/ML and modeling frameworks (e.g., PyTorch, BLAST, WarpX, ImpactX), the trend to combine a precompiled, high-performance language such as C++ with a scriptable interface language such as Python has been highly successful in ensuring performance, modularity, and productivity.
Revisiting the language choice, a new alternative emerged with the Julia programming language, which adopts just-in-time (JIT) compilation and multiple dispatch as central paradigms of the language. Such features significantly simplify the development process, enable the entire ecosystem to be fully differentiable, and provides performance on-par with that of C. At the same time, Julia can be rapidly scripted (dynamic dispatch) in simple syntax, effectively removing the ``two-language'' challenge for developers.
Although still relatively young, Julia provides capabilities that simplifies important aspects of this project. 

Julia comes with a full-featured interactive command-line REPL (Read-Evaluate-Print-Loop) built into the Julia executable similar to the REPL in Python. This means that the lattice description format, and the interaction between a user and simulation software in general, can be through the Julia language itself. Consequently, simulations will not be constrained by some program-defined language (like with MAD, Elegant, Bmad, etc.), and the user will automatically have access to such features as plotting, optimization packages, linear algebra packages, etc. This is a massive boost to the versatility and usability of any simulation program. Additionally, code
maintainability is greatly improved since the quantity of code that needs to be developed
is reduced.

That Julia is an excellent choice for the SciBmad project is exemplified by the conclusions
of the paper ``Potential of the Julia programming language for high-energy physics computing'' (Jonas Eschle et. al, Comput Softw Big Sci 7, 10 (2023)) 
written by a team of researchers from various laboratories around the world:
> Julia and its ecosystem are impressively fulfilling all these requirements [for use in HEP applications]... 
> The capaSciBmady to provide, at the same time, ease of programming and performance makes Julia the ideal programming language for HEP ... the HEP community will definitively benefit from a large scale adoption of the Julia programming language for its software development."
