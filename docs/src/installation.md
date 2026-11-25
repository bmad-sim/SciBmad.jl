# Installation
:::{note}
SciBmad runs on Windows, macOS, and Linux.
:::

SciBmad is written in the [Julia programming language](https://julialang.org/), and we recommend using Julia for the best experience. If you are new to Julia, these
resources might be helpful:

- [MATLAB-Python-Julia Cheat Sheet](https://cheatsheets.quantecon.org/)
- [Julia wikibook](https://en.wikibooks.org/wiki/Introducing_Julia)
- [ThinkJulia (for those new to programming)](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html)

A SciBmad distribution is a precompiled Julia executable that comes with SciBmad as well as a large number of other useful packages, including those for plotting, optimization, automatic differentiation, statistics, etc. Because it is precompiled, it is ready-to-go out of the box.

The latest SciBmad distribution can be installed with [Conda](https://conda-forge.org/download/), using the command

```
conda install -c bmad-sim scibmad 
```

Once installed, simply run `scibmad`, which will open the Julia executable:

```console
$ scibmad
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.12.7 (2026-08-15)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org release
|__/                   |

julia> 
```

:::{note}
If you already have Julia installed on your machine: the executable `scibmad` is its own Julia distribution, different from running your own `julia` executable. SciBmad can also be installed through the General registry in your own `julia` executable if you prefer, however in this case Julia will have to precompile SciBmad which takes some time.
:::

To see the full list of packages included in the SciBmad distribution, run `versions(all=true);` in the `scibmad` Julia.

To exit, run `exit()`, or enter Ctrl + D.

## Updating SciBmad

SciBmad can be updated using the usual machinery of `conda`:

```
conda update -c bmad-sim scibmad
```

## Jupyter Kernel Installation

[JupyterLab](https://jupyterlab.readthedocs.io/en/stable/index.html) is a widely used computational notebook authoring and editing environment, and may be a helpful companion when using SciBmad. Installation instructions for JupyterLab can be found [here](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html). 

To install a Jupyter kernel for the SciBmad distribution, open Julia using `scibmad` and run
```julia
using IJulia
installkernel("SciBmad Distribution", "--threads=auto", displayname="SciBmad Distribution", specname="scibmad-distribution",)
```

Then when Jupyter is opened, the SciBmad Distribution kernel will appear as an option.

## Adding Other Packages

Initially you will not need this, but at some point you may want to use a package that is not included in the SciBmad distribution. This can be done by adding packages within Julia. For example, to add the `Cxx.jl` package,

```julia
import Pkg; Pkg.add("Cxx")
```

You can suggest packages to be included in the SciBmad distribution by submitting an issue on Github [here](https://github.com/bmad-sim/SciBmad-Distribution/issues).

## Adding GPU Support

SciBmad supports GPU parallelization on the following hardware:

- [NVIDIA CUDA](https://github.com/JuliaGPU/CUDA.jl)
- [AMD ROCm](https://github.com/JuliaGPU/AMDGPU.jl)
- [Intel oneAPI](https://github.com/JuliaGPU/oneAPI.jl)
- [Apple Metal](https://github.com/JuliaGPU/Metal.jl)

For GPU parallelization in SciBmad, the corresponding GPU package must be added. For example, to use SciBmad with NVIDIA CUDA, add the `CUDA` package:

```julia
import Pkg; Pkg.add("CUDA")
```

## Plotting

The SciBmad distribution includes the CairoMakie and GLMakie packages from the [Makie](https://docs.makie.org/stable/) ecosystem, and [Plots](https://docs.juliaplots.org/stable/). Our personal preference is Makie.

## Using SciBmad from Python

:::{warning}
The SciBmad Python interface is still early in development, and may have poor performance and usability. For now, we encourage using SciBmad from Julia.
:::

SciBmad can also be used from Python through
[juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/), which runs Julia inside
Python. Install it with `pip install juliacall`, add SciBmad once with

```python
import juliapkg
juliapkg.add("SciBmad", "7bf6e793-a2ec-421f-8700-d1465381ee80")
```

and load it in your scripts or notebooks:

```python
from juliacall import Main as jl
jl.seval("using SciBmad")
```

The first time, juliacall installs Julia and SciBmad, which takes a while. The Julia code in this
documentation then translates to Python as follows:

```python
import numpy as np

q = jl.Quadrupole(L=0.5, Kn1=0.36)  # Julia functions through jl, with the same keyword arguments
jl.seval("x = [1.0 2.0; 3.0 4.0]")  # Julia code without a Python equivalent, as a string
a = np.asarray(jl.x)                # Julia arrays as NumPy arrays (indices count from 0)
b = jl.Array(a)                     # NumPy arrays as Julia arrays, for performance
```

Some results in Julia are read with keywords inside the square brackets, such as `x[turn=3]`.
Python has no such syntax: write `jl.getindex(x, turn=3)` instead (in general, `x[i, k=1]` is
`jl.getindex(x, i, k=1)`). The [examples](examples-index.md) include Python notebooks.

## Using SciBmad with Claude Code

[Claude Code](https://claude.com/claude-code) can write and run SciBmad code for you. SciBmad
provides a skill for it: instructions and reference material that teach Claude how SciBmad is
used, which mistakes to avoid, and how to approach tasks such as matching, orbit correction and
dynamic aperture optimization. Install it in Claude Code with

```
/plugin marketplace add bmad-sim/SciBmad.jl
/plugin install scibmad@bmad-sim
```

Claude then uses the skill whenever a request involves SciBmad, for example "match the tunes of
my ring to 0.22 and 0.18" or "correct the orbit of `ring.jl` with its corrector coils". The skill
is written for the current version of SciBmad. To update it, run
`/plugin marketplace update bmad-sim`.

## Next Steps

Once SciBmad is installed, head to the [Quickstart Guide](quickstart.md).