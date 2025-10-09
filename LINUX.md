# SciBmad Linux Installation

1. **Install JupyterLab**

We recommend using SciBmad through [Jupyter lab](https://jupyterlab.readthedocs.io/en/latest/), which is a widely-used computational notebook authoring and editing environment. If you don't have it set up already, installation instructions can be found [here](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html). 

After installing Jupyter lab, make sure you can open it in your browser. This may be done in the terminal with the command `jupyter lab`.

2. **Install Julia**

While SciBmad can be used from either Python or Julia, Julia must be installed because SciBmad is written in Julia. To install and configure the long-term support (LTS) version of Julia on your computer, in a terminal run the following command:

```
curl -fsSL https://install.julialang.org | sh -s -- --default-channel lts
```


3. **Language-Specific Installation**

Depending on whether you plan to use SciBmad in Python or Julia, follow your language-specific setup instructions below.

## Python

In Python, SciBmad is currently called through the [`juliacall` package](https://juliapy.github.io/PythonCall.jl/stable/juliacall/). We'll first install this package, and modify some of its configuration files to ensure the long-term support (LTS) version of Julia is used:

```
pip install juliacall
find "$HOME" -type f -path "*/juliacall/juliapkg.json" -exec sed -i -E 's/^\s*"julia":.*$/  "julia": "~1.10.3",/' {} +
```

Finally, we'll install SciBmad using the following command:

```
python -c 'from juliacall import Main as jl; jl.seval("import Pkg;"); jl.Pkg.add("SciBmad")'
```

This may take several minutes to install. After it's complete, you're ready to go! Download the [`python.ipynb'](https://github.com/bmad-sim/SciBmad.jl/blob/main/examples/python.ipynb) SciBmad Jupyter notebook as a simple first example to run.

### Julia

In a terminal, run the command to install the Julia Jupyter kernel:

```
julia -e 'import Pkg; Pkg.add("IJulia");'
```

To customize your Julia Jupyter kernel install, see the [IJulia documentation](https://julialang.github.io/IJulia.jl/stable/manual/installation/).

Finally, SciBmad can be installed with:

```
julia -e 'import Pkg; Pkg.add("SciBmad");'
```

This may take several minutes to install. After it's complete, you're ready to go! Download the [`julia.ipynb'](https://github.com/bmad-sim/SciBmad.jl/blob/main/examples/julia.ipynb) SciBmad Jupyter notebook as a simple first example to run.
