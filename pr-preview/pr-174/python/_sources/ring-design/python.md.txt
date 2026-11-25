(rd.python)=
# Appendix: SciBmad from Python

Every example of this tutorial is also written in Python: switch to it with the "Julia | Python"
button at the top of the page. The Python examples use the Python package `scibmad`, which runs
SciBmad (see [Installation](../installation.md)), with the lattice of each chapter in the
package [`lattices`](https://github.com/bmad-sim/SciBmad.jl/tree/main/docs/src/ring-design/lattices)
(`from lattices.ch05_ring import *`), the Python version of the Julia lattice files.

```python
import numpy as np
import scibmad as sb
from lattices.ch02_dispersion_suppressor import *

tw = sb.twiss(cell_af)
beta1 = np.asarray(tw.beta1)  # The column as a NumPy array
tw.q1                         # A summary value

sb.match(vary=[sb.Knob(qff1, "Kn1"), sb.Knob(qdf1, "Kn1")],
         targets=[tw_dsf.dx[-1] == 0, tw_dsf.dpx[-1] == 0])
```

What Python writes differently from Julia (for example `res.x(turn=3)` for Julia's
`res.x[turn=3]`, and `sb.Knob(qf, "Kn1")` for `@knob(qf.Kn1)`) is described in
[SciBmad from Python](../python.md).
