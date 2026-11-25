# Installation
:::{note}
SciBmad runs on Windows, macOS, and Linux.
:::

:::{only} python
SciBmad is written in the [Julia programming language](https://julialang.org/). The Python
package `scibmad` runs it from Python: every function and type of SciBmad is a name of
`scibmad`, with the same arguments, and the results are Python objects (NumPy arrays where
the results are numbers). Julia itself is installed with it, and runs inside the Python process.

The recommended installation is with [Conda](https://conda-forge.org/download/), which also
installs the SciBmad distribution: a Julia with SciBmad and the packages it uses precompiled,
so that SciBmad is ready to go immediately:

```
conda install -c bmad-sim -c conda-forge pyscibmad
```

`pip` works too:

```
pip install pyscibmad
```

Then, in Python:

```python
import scibmad as sb

sb.info()  # which Julia and which SciBmad.jl run SciBmad
```

With `pip`, the first use of SciBmad installs Julia and SciBmad.jl for that Python environment
(with [juliapkg](https://github.com/JuliaPy/pyjuliapkg)) and compiles them, which takes several
minutes once. Afterwards, starting SciBmad takes a few seconds.

`scibmad` uses [juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/) to run
Julia. How the few things that Julia writes differently from Python are written in Python (for
example indexing with keywords) is described in [SciBmad from Python](python.md). Every code
example in this documentation is shown in Python or in Julia, as chosen at the top of the page.
:::

:::{only} julia
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
:::

## Updating SciBmad

:::{only} julia
SciBmad can be updated using the usual machinery of `conda`:

```
conda update -c bmad-sim scibmad
```
:::
:::{only} python
SciBmad is updated with the Python package, which says which version of SciBmad.jl it uses:

```
conda update -c bmad-sim -c conda-forge pyscibmad
```

or `pip install -U pyscibmad`.
:::

## Jupyter Kernel Installation

:::{only} python
In Python, SciBmad runs in any Python environment, also in Jupyter notebooks with the usual
Python kernel: there is nothing to install for Jupyter. Julia's printed output (for example the
progress of `dynamic_aperture`) appears in the notebook.
:::

:::{only} julia
[JupyterLab](https://jupyterlab.readthedocs.io/en/stable/index.html) is a widely used computational notebook authoring and editing environment, and may be a helpful companion when using SciBmad. Installation instructions for JupyterLab can be found [here](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html). 

To install a Jupyter kernel for the SciBmad distribution, open Julia using `scibmad` and run
```julia
using IJulia
installkernel("SciBmad Distribution", "--threads=auto", displayname="SciBmad Distribution", specname="scibmad-distribution",)
```

Then when Jupyter is opened, the SciBmad Distribution kernel will appear as an option.
:::

## Adding Other Packages

:::{only} julia
Initially you will not need this, but at some point you may want to use a package that is not included in the SciBmad distribution. This can be done by adding packages within Julia. For example, to add the `Cxx.jl` package,

```julia
import Pkg; Pkg.add("Cxx")
```

You can suggest packages to be included in the SciBmad distribution by submitting an issue on Github [here](https://github.com/bmad-sim/SciBmad-Distribution/issues).
:::
:::{only} python
Python packages are installed as usual. Julia packages, which you will rarely need, are
installed into the Julia environment that SciBmad runs in, and loaded, with

```python
optim = sb.julia.package("Optim", install=True)  # the Julia module, e.g. optim.LBFGS()
```
:::

## Adding GPU Support

SciBmad supports GPU parallelization on the following hardware:

- [NVIDIA CUDA](https://github.com/JuliaGPU/CUDA.jl)
- [AMD ROCm](https://github.com/JuliaGPU/AMDGPU.jl)
- [Intel oneAPI](https://github.com/JuliaGPU/oneAPI.jl)
- [Apple Metal](https://github.com/JuliaGPU/Metal.jl)

:::{only} julia
For GPU parallelization in SciBmad, the corresponding GPU package must be added. For example, to use SciBmad with NVIDIA CUDA, add the `CUDA` package:

```julia
import Pkg; Pkg.add("CUDA")
```
:::
:::{only} python
In Python, `sb.device("cuda")` (or `"amdgpu"`, `"oneapi"`, `"metal"`) gives the GPU, whose
`asarray` copies NumPy arrays to it (see [GPU Parallelization](#gpuparallel)). The first time,
it installs the Julia package of that kind of GPU, which takes a few minutes once. To install it
beforehand, for example in a container: `python -m scibmad gpu cuda`.
:::

## Plotting

:::{only} julia
The SciBmad distribution includes the CairoMakie and GLMakie packages from the [Makie](https://docs.makie.org/stable/) ecosystem, and [Plots](https://docs.juliaplots.org/stable/). Our personal preference is Makie.
:::
:::{only} python
The columns of the results of `twiss` and `track` work with
[Matplotlib](https://matplotlib.org/) and NumPy as arrays, e.g. `plt.plot(tw.s, tw.beta1)`.
:::

## Julia and Python

:::{only} julia
SciBmad can also be used from Python, with the Python package `scibmad` (`pip install
pyscibmad`), which runs Julia inside Python. Every example in this documentation is also shown
in Python: switch with the "Julia | Python" button at the top of the page.
:::
:::{only} python
By default, `scibmad` runs the Julia of the SciBmad distribution installed with Conda in the
current environment, or else the one it installed itself. Another Julia, for example with your
own (development) versions of the SciBmad packages, is chosen with its executable and a Julia
project in which SciBmad and PythonCall are installed:

```
python -m scibmad config --julia /path/to/julia --project /path/to/project
```

This is stored for your user (`python -m scibmad config --reset` goes back to the default). The
environment variables `SCIBMAD_JULIA` and `SCIBMAD_PROJECT` do the same for one session, and so
does `sb.init(julia=..., project=...)` before the first use of SciBmad in a script, which also
sets the number of Julia threads (`threads=`). `python -m scibmad info` shows what is used.
:::

## Next Steps

Once SciBmad is installed, head to the [Quickstart Guide](quickstart.md).