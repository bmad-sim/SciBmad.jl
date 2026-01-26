# SciBmad Windows Installation

1. **Install JupyterLab**

We recommend using SciBmad through [Jupyter lab](https://jupyterlab.readthedocs.io/en/latest/), which is a widely-used computational notebook authoring and editing environment. If you don't have it set up already, installation instructions can be found [here](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html). 

After installing Jupyter lab, make sure you can open it in your browser.

2. **Install Julia**

While SciBmad can be used from either Python or Julia, Julia must be installed because SciBmad is written in Julia. To install and configure the long-term support (LTS) version of Julia on your computer, in a terminal run the following commands in Powershell:

```
winget install --name Julia --id 9NJNWW8PVKMN -e -s msstore
juliaup add lts
juliaup default lts
```

3. **Language-Specific Installation**

Depending on whether you plan to use SciBmad in Python or Julia, follow your language-specific setup instructions below.

## Python

In Python, SciBmad is currently called through the [`juliacall` package](https://juliapy.github.io/PythonCall.jl/stable/juliacall/). We'll first install this package, and set an environment variable to ensure that the long-term suppport (LTS) version of Julia is used. In a (Anaconda) Powershell:

```
pip install juliacall
setx PYTHON_JULIAPKG_EXE julia
set PYTHON_JULIAPKG_EXE=julia
```

Finally, we'll install SciBmad using the following command:

```
python -c "from juliacall import Main as jl; jl.seval('import Pkg;'); jl.Pkg.add('SciBmad')"
```

This may take several minutes to install. After it's complete, you're ready to go! Download the [python/nonlinear-twiss.ipynb](https://github.com/bmad-sim/SciBmad.jl/blob/main/examples/python/nonlinear-twiss.ipynb) SciBmad Jupyter notebook as a simple first example to run.

### Julia

In a Powershell, run the command to install the Julia Jupyter kernel:

```
julia -e 'import Pkg; Pkg.add("IJulia");'
```

To customize your Julia Jupyter kernel install, see the [IJulia documentation](https://julialang.github.io/IJulia.jl/stable/manual/installation/).

Finally, SciBmad can be installed with:

```
julia -e 'import Pkg; Pkg.add("SciBmad");'
```

This may take several minutes to install. After it's complete, you're ready to go! Download the [julia/nonlinear-twiss.ipynb](https://github.com/bmad-sim/SciBmad.jl/blob/main/examples/julia/nonlinear-twiss.ipynb) SciBmad Jupyter notebook as a simple first example to run.
