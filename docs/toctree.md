<!--
Site navigation. This file is appended to the repository README by
`docs/src/conf.py` to produce `docs/src/index.md`, the root document. The
toctrees are `:hidden:` so the landing page itself stays a verbatim copy of the
README while Sphinx still gets the navigation structure it requires here.
-->

```{toctree}
:hidden:
:maxdepth: 1

Overview <self>
Table of Contents <contents>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting Started

installation
Quickstart <quickstart>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Lattice

element
beamline
defexpr
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Particle Tracking

track
tracking-methods
collective
timedependent
batch
co
dynamic-aperture
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Analysis

twiss
parametric-nf
optimize
fma
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Examples

examples-index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Physics

coordinates
sagancavity-tracking
sagancavity-physics
miscellaneous
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: About

governance
```
