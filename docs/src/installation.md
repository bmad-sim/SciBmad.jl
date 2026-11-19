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

SciBmad supports GPU parallelization on the following hardwares:

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

The first time, juliacall installs Julia and SciBmad, which takes a while. Julia functions are
then called through `jl`, e.g. `tw = jl.twiss(ring)`, and Julia code without a Python equivalent,
such as `@elements`, is run with `jl.seval("...")`. Results are Julia objects:
`np.asarray(tw.beta1)` gives a column as a NumPy array, and `tw.beta1[jl.qd]` its value at the
beginning of the element `qd`. Python has no keywords in indexing, so tracking results are read
from their arrays instead:
`np.asarray(res.data.v)[particle, coordinate, location, saved turn]`
(counting from 0). Pass NumPy arrays to SciBmad as Julia arrays, e.g. `jl.Array(v0)`, for
performance. The [examples](examples-index.md) include Python notebooks.

## Next Steps

Once SciBmad is installed, head to the [Quickstart Guide](quickstart.md).