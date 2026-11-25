(rd.python)=
# Appendix: SciBmad from Python

SciBmad can also be used from Python, through
[juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/), which runs Julia inside
Python. How to install it is described in [Using SciBmad from Python](../installation.md).

```python
from juliacall import Main as jl
jl.seval("using SciBmad")
```

Julia functions are called through `jl`, with the same keyword arguments. Julia code that has no
Python equivalent, such as the macros `@elements` and `@knob` and the targets written with `==`, is
given as a string to `jl.seval`. A lattice file of this tutorial can be included in the same way,
after which its variables are available through `jl`:

```python
import numpy as np

jl.seval('include("lattices/02-dispersion-suppressor.jl")')
tw = jl.twiss(jl.cell_af)
beta1 = np.asarray(tw.beta1)    # The column as a NumPy array
tw.q1                           # A summary value
```

Matching works the same way:

```python
jl.seval("""
match!(vary=[@knob(qff1.Kn1), @knob(qdf1.Kn1)],
       targets=[tw_dsf.dx[end] == 0, tw_dsf.dpx[end] == 0])
""")
jl.qff1.Kn1
```

Results that are read with keywords inside the square brackets in Julia, such as `res.x[turn=3]`,
are read with `jl.getindex(res.x, turn=3)` in Python.
