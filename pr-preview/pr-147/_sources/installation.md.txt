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

The latest SciBmad distribution can be installed with [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), using the command

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

To see the full list of packages included in the SciBmad distribution, run `versions(all=true);` in Julia.

To exit, run `exit()`, or enter Ctrl + D.

## Updating SciBmad

SciBmad can be updated using the usual machinery of `conda`:

```
conda update -c bmad-sim scibmad
```

## Jupyter Kernel Installation

To install a Jupyter kernel for the SciBmad distribution, in Julia run
```julia
using IJulia
installkernel("SciBmad Distribution", "--threads=auto", displayname="SciBmad Distribution", specname="scibmad-distribution",)
```

Then when Jupyter is opened, the SciBmad Distribution kernel will appear as an option.

## Adding Other Packages

You may want to use a package that is not included in the SciBmad distribution. This can be done by adding packages within Julia. For example, to add the `Cxx.jl` package,

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