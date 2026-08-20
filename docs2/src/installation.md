# Installation Instructions

SciBmad is written in the [Julia programming language](https://julialang.org/), and we generally recommend using Julia for a best experience. If you are new to Julia, then these resources might be helpful:

- [MATLAB-Python-Julia Cheat Sheet](https://cheatsheets.quantecon.org/)
- [Julia wikibook](https://en.wikibooks.org/wiki/Introducing_Julia)
- [ThinkJulia (for those new to programming)](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html)

To install Julia, follow the platform-dependent [installation instructions here](https://github.com/JuliaLang/juliaup). `juliaup` is a Julia version manager that will make it easy to install and use new stable Julia versions when available.

After installation, the `SciBmad` package can be added in Julia by running:

```julia
import Pkg; Pkg.add("SciBmad")
```

This may around 10-20 minutes to compile and install. Julia also has various plotting packages you may add as well, including [Makie](https://docs.makie.org/stable/) and [Plots](https://docs.juliaplots.org/stable/). Our personal preference is `Makie`. 

Python users can use SciBmad via the [PySciBmad](github.com/bmad-sim/PySciBmad) Python interface package currently in development. Julia can also be called directly from Python using the [`juliacall` package](https://juliapy.github.io/PythonCall.jl/stable/juliacall/).