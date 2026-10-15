# Installation
:::{note}
SciBmad runs on Windows, macOS, and Linux.
:::

SciBmad is written in the [Julia programming language](https://julialang.org/), and we
generally recommend using Julia for the best experience. If you are new to Julia, these
resources might be helpful:

- [MATLAB-Python-Julia Cheat Sheet](https://cheatsheets.quantecon.org/)
- [Julia wikibook](https://en.wikibooks.org/wiki/Introducing_Julia)
- [ThinkJulia (for those new to programming)](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html)

First, Julia must be installed. To do so, follow the platform-dependent
[installation instructions here](https://github.com/JuliaLang/juliaup). `juliaup` is a
Julia version manager that makes it easy to install and use new stable Julia versions as
they become available. **We highly recommend using the long term support (LTS) channel of Julia with SciBmad. This can be set in the terminal after installing `juliaup` using the command:**

```
juliaup add lts
juliaup default lts
```

After the Julia installation, run `julia` and add the `SciBmad` package with the command:
```julia
import Pkg; Pkg.add("SciBmad")
```

This may take around 10-20 minutes to compile and install.

## SciBmad Distribution

As an alternative to installing SciBmad with `Pkg`, SciBmad is also available as a
**Distribution**: a single download that bundles its own Julia together with SciBmad and a
set of commonly used packages (plotting, optimization, differentiation, Jupyter support,
and others), all already compiled. Installing the Distribution does not require Julia to
be installed separately, and `using SciBmad` loads without any installation or
precompilation wait.

The Distribution, together with its installation and usage instructions, is found at the
[SciBmad-Distribution](https://github.com/bmad-sim/SciBmad-Distribution) repository. It
can be installed with conda, or with an app installer from the repository's releases page.

Reasons to use the Distribution:
- Getting started quickly, without the 10-20 minute install and compile time.
- Setting up SciBmad on many machines, or for a class or workshop, where everyone should
  get the same tested set of package versions.

Reasons not to use the Distribution:
- The bundled package versions are fixed for each Distribution release, and new releases
  of the Distribution may lag behind new releases of SciBmad. Users who want the newest
  SciBmad as soon as it is released should install with `Pkg`.
- Developers modifying SciBmad or its component packages should install with `Pkg`.
- The Distribution is a large download, since it includes many packages a given user may
  not need.

## Updating SciBmad

To update SciBmad, and any other Julia packages you have, in Julia run

```julia
import Pkg; Pkg.update()
```

## Plotting

Julia has various plotting packages you may add as well, including
[Makie](https://docs.makie.org/stable/) and [Plots](https://docs.juliaplots.org/stable/).
Our personal preference is Makie.

## Julia Jupyter Kernel

A Jupyter kernel for Julia can be installed using the [IJulia](https://ijulia.org/stable/) package. We recommend running this kernel installation command in Julia, which will enable multithreading and automatically use the global project environment:

```julia
import Pkg; Pkg.add("IJulia")
using IJulia
IJulia.installkernel("Julia Global", env=Dict("JULIA_NUM_THREADS"=>"auto"), "--project=$(Base.active_project())")
```
Then when Jupyter is opened, the Julia kernel will appear as an option.

## Using SciBmad from Python

Python users can use SciBmad via the [PySciBmad](https://github.com/bmad-sim/PySciBmad)
Python interface package, currently in development. Julia can also be called directly
from Python using the [`juliacall` package](https://juliapy.github.io/PythonCall.jl/stable/juliacall/).

The [Examples](examples-index.md) section includes notebooks written in both Julia and
Python using `juliacall`.

:::{warning}
The Python interface to SciBmad is early in development.
:::

## Next steps

Once SciBmad is installed, head to the [Quickstart](quickstart.md).
