(ring-design)=
# Ring Design Tutorial

This ring design tutorial serves as an introduction to SciBmad and to the design of a collider
ring. In this tutorial you will design your own collider ring based on the 18 GeV Electron
Storage Ring (ESR) of the Electron-Ion Collider (EIC), currently in design at the Brookhaven
National Laboratory on Long Island. The EIC will probe the properties of matter via deep
inelastic scattering, while colliding spin-polarized electrons with protons and nuclei. By the
end, you will have gained an understanding of many important concepts involved in building a
real collider ring, as well as experience using SciBmad.

The tutorial follows the
[Collider Ring Design Tutorial using Bmad](https://github.com/bmad-sim/bmad-ecosystem/tree/main/bmad-doc/tutorial_ring_design)
(J. Devlin, G. Hoffstaetter, D. Kuzovkova, D. Sagan, M. Signorelli, J. Unger, N. Wang), and
designs the same ring. It is meant to teach accelerator physics at the level of a graduate
course. It is assumed that you know something about beta functions, dispersion and closed orbits.
No prior experience with SciBmad or {jltext}`Julia`{pytext}`Julia (which runs SciBmad)` is assumed,
but it helps to have read the
[Quickstart Guide](../quickstart.md) first. Every example is shown in Julia or in Python, as
chosen with the switch at the top of the page.

## How the tutorial works

Each chapter starts with a short introduction to what is to be done, followed by an example. The
chapters end with exercises. Answers to some of them are in [Answers to Selected
Exercises](answers.md).

:::{only} julia
In SciBmad, a lattice is a Julia program: elements and beamlines are Julia objects, and
matching is a Julia function call. You will build the ring step by step in one file, for example
`ring.jl`, and add to it in each chapter. Everything else, such as looking at the optics or
plotting them, you do in the Julia REPL (or in a Jupyter notebook) after running the file with
`include("ring.jl")`.
:::
:::{only} python
In SciBmad, a lattice is a program: elements and beamlines are objects, and matching is a
function call. You will build the ring step by step in one Python file, for example `ring.py`,
and add to it in each chapter. Everything else, such as looking at the optics or plotting them,
you do in Python (or in a Jupyter notebook) after running the file, e.g. with
`from ring import *`.
:::

The matches are part of the lattice file. Each time the file is run, the quadrupole strengths are
matched again, which takes a few seconds. This has two advantages. The lattice file records not
only the strengths but also how they were found. And when you change something early in the
file, such as the phase advance of the arc cells, all the matches that depend on it follow.
Nothing has to be copied from one file to another.

For each chapter that adds to the ring, the state of the lattice file at the end of the chapter
is provided in the directory
[`docs/src/ring-design/lattices`](https://github.com/bmad-sim/SciBmad.jl/tree/main/docs/src/ring-design/lattices).
:::{only} julia
Each of these files starts by including the file of the previous chapter. To start with a later
chapter, or to compare with your own work, include the file of the chapter before it, e.g.
`include("lattices/05-ring.jl")`.
:::
:::{only} python
In Python, the directory `lattices` is a package, and each module of it starts by importing the
module of the previous chapter. To start with a later chapter, or to compare with your own work,
import the module of the chapter before it, from the directory that contains `lattices`, e.g.
`from lattices.ch05_ring import *`.
:::

| Chapter | Lattice file (Julia) | Lattice module (Python) |
|---|---|---|
| [1. FODO Cells](fodo.md) | `01-fodo.jl` | `ch01_fodo.py` |
| [2. Dispersion Suppressor](dispersion-suppressor.md) | `02-dispersion-suppressor.jl` | `ch02_dispersion_suppressor.py` |
| [3. Matching to the Straight Section](match-to-straight.md) | `03-match-to-straight.jl` | `ch03_match_to_straight.py` |
| [4. Machine Coordinates](coordinates.md) | | |
| [5. Constructing the Ring](ring.md) | `05-ring.jl` | `ch05_ring.py` |
| [6. Low-Beta Interaction Region](interaction-region.md) | `06-interaction-region.jl` | `ch06_interaction_region.py` |
| [7. Tune Cell](tune-cell.md) | `07-tune-cell.jl` | `ch07_tune_cell.py` |
| [8. Particle Phase Space Coordinates](phase-space.md) | | |
| [9. RF Cavities](rf.md) | `09-rf.jl` | `ch09_rf.py` |
| [10. Long Term Tracking](tracking.md) | | |
| [11. Deferred Expressions and Contexts](control.md) | | |
| [12. Dynamic Aperture](dynamic-aperture.md) | `12-sextupoles.jl` | `ch12_sextupoles.py` |
| [13. Comparing Lattices](comparing.md) | | |
| [14. Orbit Correction](orbit-correction.md) | `14-orbit-correction.jl` | `ch14_orbit_correction.py` |
| [15. Spin Tracking with Ramping](spin.md) | `15-spin.jl` (a separate ring) | `ch15_spin.py` |

The SciBmad documentation describes everything used here in more detail. Links to the relevant
pages are given along the way.

```{toctree}
:hidden:
:maxdepth: 1

fodo
dispersion-suppressor
match-to-straight
coordinates
ring
interaction-region
tune-cell
phase-space
rf
tracking
control
dynamic-aperture
comparing
orbit-correction
spin
python
answers
```
