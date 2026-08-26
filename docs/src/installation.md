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

There are two install SciBmad: one is by installing a "distribution", which is precompiled binary of Julia including SciBmad and an assortment of other useful packages (plotting packages, optimizers, statistics, etc). This is useful for a fast installation, however is currently only limited to CPU tracking. The second way to install SciBmad is through the more standard approach of installing Julia, and then adding SciBmad. This approach is most flexible, however will require 10-20 minutes of "precompile" time in the installation process.

If you are not using the GPU, then for most users we recommend installing a SciBmad distribution.

## Installing a SciBmad Distribution
### macOS
First, download one of the `dmg` files linked below to your `~/Downloads/` directory, depending on your computer architecture. You can check what architecture you have in a terminal with the command `uname -m`

- [macOS `x86_64` SciBmad Distribution](https://github.com/bmad-sim/SciBmad-Distribution/releases/download/SciBmad-0.5.2/scibmaddistribution-26.8.26-x86_64.dmg)
- [macOS `arm64` SciBmad Distribution](https://github.com/bmad-sim/SciBmad-Distribution/releases/download/SciBmad-0.5.2/scibmaddistribution-26.8.26-aarch64.dmg)

The released bundles are signed with a self-signed certificate, so macOS does not trust them out of the box. To enable trust in SciBmad, run the following commands

```bash
hdiutil attach ~/Downloads/scibmaddistribution-26.8.23-aarch64.dmg
ditto "/Volumes/SciBmadDistribution Installer/SciBmadDistribution.app" /Applications/SciBmadDistribution.app
hdiutil detach "/Volumes/SciBmadDistribution Installer"
sudo xattr -dr com.apple.quarantine /Applications/SciBmadDistribution.app
sudo chmod -R a-w /Applications/SciBmadDistribution.app
```
This may take a minute or two. Note: The `sudo` commands will ask for a password.

After installation, run the `SciBmadDistribution` app, and a Julia window will open.

### Linux
First, download one of the `snap` files linked below, depending on your computer architecture. You can check what architecture you have in a terminal with the command `uname -m`

- [Linux `x86_64` SciBmad Distribution](https://github.com/bmad-sim/SciBmad-Distribution/releases/download/SciBmad-0.5.2/scibmaddistribution-26.8.26-x86_64.snap)
- [Linux `arm64` SciBmad Distribution](https://github.com/bmad-sim/SciBmad-Distribution/releases/download/SciBmad-0.5.2/scibmaddistribution-26.8.26-aarch64.snap)

In the directory with the `snap` file, in a terminal run the command
```
snap install --classic --dangerous SciBmadDistribution.snap
```

### Windows
First download the `msix` file here:
- [Windows `msix` SciBmad Distribution](https://github.com/bmad-sim/SciBmad-Distribution/releases/download/SciBmad-0.5.2/scibmaddistribution-26.8.26-x86_64.msix)

The released bundle is signed with a self-signed certificate, so Windows will not trust them out of the box. To enable trust in SciBmad, open the MSIX bundle properties and add its certificate to the trusted certificate authorities first (see https://www.advancedinstaller.com/install-test-certificate-from-msix.html). Then double-click the installer and install the app.

### Distribution Jupyter Kernel 
To use [Jupyter](https://jupyter.org/) with a SciBmad distribution, in the `SciBmadDistribution` Julia application run the command
```julia
using IJulia
IJulia.installkernel("SciBmad")
```
The kernel can then be used within a Jupyter notebook.

## Installing SciBmad the Standard Way
First, Julia must be installed. To install Julia, follow the platform-dependent
[installation instructions here](https://github.com/JuliaLang/juliaup). `juliaup` is a
Julia version manager that makes it easy to install and use new stable Julia versions as
they become available. **We highly recommend using the long term support (LTS) channel of Julia with SciBmad. This can be set in the terminal using the command:**

```
juliaup default lts
```

After the Julia installation, run `julia` and add the `SciBmad` package with the command:
```julia
import Pkg; Pkg.add("SciBmad")
```

This may take around 10-20 minutes to compile and install.

# Plotting Packages

Julia has various plotting packages available, including
[Makie](https://docs.makie.org/stable/) and [Plots](https://docs.juliaplots.org/stable/), both of which are included in a SciBmad distribution. Our personal preference is `Makie`. A SciBmad distribution ships with `Plots` and `Makie`.

# Using SciBmad from Python

Python users can use SciBmad via the [PySciBmad](https://github.com/bmad-sim/PySciBmad)
Python interface package, currently in development. Julia can also be called directly
from Python using the [`juliacall` package](https://juliapy.github.io/PythonCall.jl/stable/juliacall/).

The [Examples](examples-index.md) section includes notebooks written in both Julia and
Python using `juliacall`.

```{warning}
The Python interface to SciBmad is still early in development.
```

## Next steps

Once SciBmad is installed, head to the [Quickstart](quickstart.md).
