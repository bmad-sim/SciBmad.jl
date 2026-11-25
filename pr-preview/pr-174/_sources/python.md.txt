(python)=
# SciBmad from Python

The Python package `scibmad` (installed as `pyscibmad`, see [Installation](installation.md))
runs SciBmad, which is written in Julia, from Python. Every name of SciBmad, and of the packages
it includes (Beamlines, BeamTracking, GTPSA, NonlinearNormalForm, ...), is a name of `scibmad`,
with the same arguments and keyword arguments:

```python
import numpy as np
import scibmad as sb

qf = sb.Quadrupole(Kn1=0.36, L=0.5)
fodo = sb.Beamline([qf, sb.Drift(L=1.0), sb.Quadrupole(Kn1=-0.36, L=0.5), sb.Drift(L=1.0)],
                   species_ref=sb.Species("electron"), pc_ref=18e9)
tw = sb.twiss(fodo)
tw.beta1[qf]  # a number
```

The results are Python objects whose attributes, indexing, operators and display are those of
SciBmad, so the rest of this documentation applies to Python as it is written. Every code
example is shown in Python or in Julia, as chosen with the switch at the top of the page. This
page describes what Python writes differently, and why.

## What Python Writes Differently

| Julia | Python | |
|---|---|---|
| `@elements begin qf = Quadrupole(...) end` | `with sb.elements(): qf = sb.Quadrupole(...)` | names the elements after their variables (or give `name="qf"`) |
| `@knob(qf.Kn1, lower=0)` | `sb.Knob(qf, "Kn1", lower=0)` | Python cannot capture `qf.Kn1` itself, only its value |
| `res.x[qd, turn=3]` | `res.x(qd, turn=3)` | Python has no keywords in indexing: a call of a result that cannot be called is indexing |
| `res.x[]` | `res.x()` | |
| `tw.beta1[qd, :]`, `at=:` | `tw.beta1[qd, :]`, `at=...` | `:` in brackets; elsewhere `...` (or `slice(None)`) |
| `tw.beta1[end]`, `bl[end-1]`, `tw.beta1[begin]` | `tw.beta1[-1]`, `bl[-2]`, `tw.beta1[sb.begin]` | also `sb.end`, `sb.end - 1` |
| `tw.beta1[(0, 1.23)]` | `tw.beta1[0:1.23]` | an s-range is a slice of floats (a tuple in brackets would be two indexes) |
| `tw.beta1[2:4]` | `tw.beta1[2:5]` | a Python slice does not include its stop |
| `:exit`, `rawget(qf, :Kn1)` | `"exit"`, `sb.rawget(qf, "Kn1")` | strings are Symbols where a function needs one (or `sb.Symbol("exit")`) |
| `match!`, `push!` | `sb.match`, `sb.push_` | `f!` is `f_`; `match`, `undo`, `setknobs`, `phase_trombone`, `scalarize` also have their name without `!` |
| `true`, `false`, `nothing` | `True`, `False`, `None` | |
| `[1e-3 0 0 0 0 0]` | `[[1e-3, 0, 0, 0, 0, 0]]` | a list of lists of numbers, or a NumPy array, is a matrix |
| `(beta1=10, alpha1=0)` | `dict(beta1=10, alpha1=0)` | |
| `x -> x.kind == "Quadrupole"` | `lambda x: x.kind == "Quadrupole"` | |
| `r"q.*"i` | `re.compile("q.*", re.I)` | |
| `DefExpr(c -> -c.kq)` | `sb.DefExpr(lambda c: -c.kq)` | see [Deferred expressions](#python.defexpr) |
| `withvalues(c; k=1.0) do ... end` | `with sb.withvalues(c, k=1.0): ...` | |
| `x === y`, `x isa Twiss` | `x is y`, `isinstance(x, sb.Twiss)` | |
| `sin.(x)`, `x .* y` | `np.sin(x)`, `x * y` | arrays are NumPy arrays |
| `MtlArray(Float32.(v0))` | `sb.device("metal").asarray(v0, dtype=np.float32)` | see [GPUs](#python.gpu) |
| `try ... catch err ... end` | `try: ... except sb.JuliaError as err: ...` | |

## Numbers, Arrays and Indexes

Python numbers, strings, lists, tuples and dictionaries are converted to their Julia
equivalents when they are passed to SciBmad: a list becomes a Julia vector of the type that the
Julia literal `[...]` would have (so `[qf, qd]` is a vector of elements), a list of lists of
numbers a matrix, as for NumPy. NumPy arrays become Julia arrays of the same element type
(`float32` stays 32-bit), with the same indexes: `v0[i, j]` in NumPy is `v0[i+1, j+1]` in Julia.

Julia arrays of numbers are returned as NumPy arrays. A CPU array shares its memory with Julia
(no copy), so changing it changes the Julia array: `ring.context.kicks[0] = 1e-5` sets a
corrector. Other Julia arrays (of elements, of expressions, GPU arrays) are returned as
sequences that are indexed as NumPy arrays are, from 0.

Two kinds of integers are not array indexes: the beamline index of an element (its position in
the beamline, as in `fodo[4]` and `tw.beta1[4]`), and the particle number of `track` results
(`res.x(turn=3, particle=1)`). They start at 1, in Python as in Julia and in the printed
tables of SciBmad, so that the same element has the same number everywhere (for example the
`beamline_index` of an element is its index in the beamline). The lists that SciBmad returns
start at 0 as always in Python: `fodo[[d]][0]` is the first placement of `d`.

## Values and Operators

A SciBmad value without a Python equivalent (an element, a beamline, a `twiss` result, a Taylor
series, an expression of `TwissExpr`, ...) is a proxy of the Julia value. Its attributes are the
Julia properties (`ele.Kn1`, also to set them), `repr` and Jupyter show it as Julia does,
`help(sb.twiss)` shows the Julia documentation, and `dir()` lists the properties.

The operators of SciBmad's own values are Julia's, so that Taylor series, deferred expressions,
time-dependent parameters, batch parameters and the expressions of `TwissExpr` combine as in
Julia: `1e-4*np.sin(1e6*sb.Time())` is a time-dependent parameter, and `tw.q1 == 0.25` a target.
NumPy's functions (`np.sin`, `np.sqrt`, ...) of these values are the Julia functions. A
comparison of expressions is a target, not a truth value: `0.1 <= tw.q1 <= 0.3` is an error
(write `sb.Target(tw.q1, lower=0.1, upper=0.3)`).

The columns of `twiss` and `track` results behave as NumPy arrays of their values in arithmetic
(`tw.dx / np.sqrt(tw.beta1)`), with NumPy (`np.max(tw.beta1)`), and with Matplotlib
(`plt.plot(tw.s, tw.beta1)`); `np.asarray(tw.beta1)` is the array. Julia's functions such as
`sb.maximum` and `sb.mean` act on expressions of `TwissExpr`, which NumPy cannot.

Errors in Julia are raised as `sb.JuliaError` with Julia's message, also as the matching Python
exception (`IndexError`, `KeyError`, `TypeError`, `ValueError`, `AttributeError`, ...).

(python.defexpr)=
## Deferred Expressions

`sb.DefExpr(f)` calls the Python function `f` once, with symbolic context variables, and keeps
the Julia expression it builds from them: evaluating it later calls no Python, so it is as fast
as a Julia function, and works on GPUs. This works for arithmetic, NumPy functions and entries of
vector variables (`c.dk[i]`, from 0 as for the NumPy array). Python values other than the
context variables (such as `i` in a loop) are read when `sb.DefExpr` is called. A function that
does more with the variables, for example `math.sin(c.kq)` or an `if` on them, is called from
Julia every time instead, which is much slower and works only on the CPU (with a
`PerformanceWarning`).

Python functions given to SciBmad elsewhere (`fodo[lambda e: ...]`, targets as a function of a
`TwissExpr`, `sb.lazy(f)`, function knobs) are called from Julia, with the arguments as proxies.

(python.gpu)=
## GPUs

```python
gpu = sb.device("cuda")  # or "metal", "amdgpu", "oneapi"; sb.device() picks the one present
v0 = gpu.asarray(np.random.rand(100_000, 6) * 1e-5)
res = sb.track(fodo, v0=v0, n_turns=100)
x = np.asarray(res.x(turn=100))  # copied to the host
```

The first `sb.device` of a kind installs its Julia package (CUDA.jl, Metal.jl, AMDGPU.jl or
oneAPI.jl) if needed, once. Apple's Metal GPUs have no 64-bit floats: give
`dtype=np.float32`. GPU arrays are computed with on the GPU (`v0 * 2` is a GPU array), and
`np.asarray` copies them to the host.

## The Julia Session

`sb.julia` is the Julia session: `sb.julia.Base`, `sb.julia.Main` and the modules of SciBmad
are its modules, `sb.julia.package("Optim", install=True)` loads another Julia package, and
`sb.julia.seval("...")` evaluates Julia code. SciBmad itself never needs Julia code, with one
exception (below). How to choose the Julia that runs SciBmad, and its number of threads, is
described in [Installation](installation.md).

## Limitations

These follow from Python and Julia themselves, and are not expected to change:

- **Callbacks of the tracking** (`callbacks` of a `Bunch`, see [Callbacks](#track.callbacks))
  run inside the compiled tracking kernels, for every particle and integration step, also on
  GPUs. They must be Julia functions, e.g. `sb.julia.seval("(i, coords, p) -> ...")`.
- **Python functions in Julia** run only on the CPU and are much slower than Julia code (see
  [Deferred expressions](#python.defexpr)); time-dependent parameters built from `sb.Time()`
  with operators and NumPy functions are Julia code.
- **Interrupting**: Ctrl-C stops a long Julia computation only when it returns to Python.
- **Mixing Taylor series of two GTPSA Descriptors** aborts Julia, and with it Python (see
  [Parametric Normal Form](parametric-nf.md)).
- **Julia values cannot be pickled** (e.g. for `multiprocessing`); save results as NumPy arrays.
- **Static tools** (type checkers, IDE completion before running) do not know SciBmad's names,
  which come from Julia when the program runs; `dir(sb)` and completion in IPython and Jupyter
  list them.
- **New Julia functionality** is available in Python as soon as SciBmad has it, with the rules
  of this page. Only two of them could change meaning if SciBmad changed: a call of a value that
  cannot be called is indexing (if a type of result became callable, `x(...)` would call it),
  and the names without `!` above (`sb.match` stays `match!`).
