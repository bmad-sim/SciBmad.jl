---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Julia
  language: julia
  name: julia
---

(defexpr)=
# Deferred Expressions and Contexts

```{code-cell} julia
:tags: [remove-cell]
using SciBmad
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

```{code-cell} python
:tags: [remove-cell]
import numpy as np
import scibmad as sb
sb.options.display_width = 100
sb.options.display_height = 30
```

Consider the simple FODO cell:

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  d = Drift(L=1.2)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
end

fodo = Beamline([qf, d, qd, d], E_ref=18e9,
        species_ref=Species("electron"))
```

```{code-cell} python
with sb.elements():
    qf = sb.Quadrupole(Kn1=0.36, L=0.5)
    d = sb.Drift(L=1.2)
    qd = sb.Quadrupole(Kn1=-0.36, L=0.5)

fodo = sb.Beamline([qf, d, qd, d], E_ref=18e9,
                   species_ref=sb.Species("electron"))
fodo
```

We set `qf.Kn1 = 0.36`, and `qd.Kn1 = -0.36`. But what if we want to ensure that
`qd.Kn1 == -qf.Kn1` always? We can bake-in such an interdependence, common in accelerator
parameters, using a "deferred expression" - an expression where evaluation is postponed
until its result is actually needed, rather than immediately when it is defined. To do this, let's first define a variable `kq = 0.36` in an object called a `Context`. For interactive use, there is a global context `DEFAULT_CONTEXT` available:

```{code-cell} julia
println("BEFORE:")
println(DEFAULT_CONTEXT)

DEFAULT_CONTEXT.kq = 0.36 # Set variable kq

println("AFTER:")
println(DEFAULT_CONTEXT)
```

```{code-cell} python
print("BEFORE:")
print(sb.DEFAULT_CONTEXT)

sb.DEFAULT_CONTEXT.kq = 0.36  # Set variable kq

print("AFTER:")
print(sb.DEFAULT_CONTEXT)
```

:::{only} julia
Then, let's write a [lambda/anonymous function](https://docs.julialang.org/en/v1/manual/functions/#man-anonymous-functions) that receives the context as an argument (below, `c`) and returns either `+kq` or `-kq`:
:::
:::{only} python
Then, let's write a [lambda
function](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions) that receives the
context as an argument (below, `c`) and returns either `+kq` or `-kq`:
:::


```{code-cell} julia
# Define lambda functions
kqf = c -> c.kq
kqd = c -> -c.kq

# Call with the context:
@show kqf(DEFAULT_CONTEXT)
@show kqd(DEFAULT_CONTEXT);
```

```{code-cell} python
# Define lambda functions
kqf = lambda c: c.kq
kqd = lambda c: -c.kq

# Call with the context:
print("kqf(DEFAULT_CONTEXT) =", kqf(sb.DEFAULT_CONTEXT))
print("kqd(DEFAULT_CONTEXT) =", kqd(sb.DEFAULT_CONTEXT))
```

Now, if we can just set the quadrupole strengths `Kn1` of `qf` and `qd` to always be the outputs of these functions, then we have achieved our goal! SciBmad's `DefExpr` type allows you to do this, by wrapping such lambda functions:

```{code-cell} julia
qf.Kn1 = DefExpr(kqf);
qd.Kn1 = DefExpr(kqd);

@show qf.Kn1;
@show qd.Kn1;
```

```{code-cell} python
qf.Kn1 = sb.DefExpr(kqf)
qd.Kn1 = sb.DefExpr(kqd)

print("qf.Kn1 =", qf.Kn1)
print("qd.Kn1 =", qd.Kn1)
```

:::{only} python
`sb.DefExpr` calls the Python function once, with a context whose variables are symbols, and
keeps the Julia expression that the function builds from them: when the expression is
evaluated later, no Python is called, and it is as fast as one written in Julia. This works
for arithmetic and NumPy functions of the variables (`np.sin(c.kq)`), and for entries of
vector variables (`c.dk[i]`, counted from 0 as in Python). A function that does more, such as
`math.sin(c.kq)` or an `if` on a variable, is called from Julia each time instead, which is
much slower (a `PerformanceWarning` says so). Python values that the function uses other than
the context variables are read when `sb.DefExpr` is called.
:::

When we change the value of `kq`, it is reflected now in the quadrupoles

```{code-cell} julia
DEFAULT_CONTEXT.kq = 0.37
@show qf.Kn1;
@show qd.Kn1;
```

```{code-cell} python
sb.DEFAULT_CONTEXT.kq = 0.37
print("qf.Kn1 =", qf.Kn1)
print("qd.Kn1 =", qd.Kn1)
```

To try a value for a while, {jl}`push!`{py}`sb.push_` it onto the context and
{jl}`pop!`{py}`sb.pop_` it off again: the previous
value is used again (and a variable that {jl}`push!`{py}`sb.push_` added is removed). For example:

```{code-cell} julia
DEFAULT_CONTEXT.vrf = 1e6
cav = RFCavity(L=0.1, voltage=DefExpr(c -> c.vrf), rf_frequency=5e8)
println("Before push! : ", cav.voltage)
push!(DEFAULT_CONTEXT; vrf=2e6)
println("After push!  : ", cav.voltage)
pop!(DEFAULT_CONTEXT)
println("After pop!   : ", cav.voltage);
```

```{code-cell} python
sb.DEFAULT_CONTEXT.vrf = 1e6
cav = sb.RFCavity(L=0.1, voltage=sb.DefExpr(lambda c: c.vrf), rf_frequency=5e8)
print("Before push_ :", cav.voltage)
sb.push_(sb.DEFAULT_CONTEXT, vrf=2e6)
print("After push_  :", cav.voltage)
sb.pop_(sb.DEFAULT_CONTEXT)
print("After pop_   :", cav.voltage)
```

Alternatively, `withvalues` does the same around a block of code{pytext}` (a with-block)`, and
restores the values even if the code throws an error:

```{code-cell} julia
withvalues(DEFAULT_CONTEXT; vrf=2.0e6) do
  println("In withvalues block     : ", cav.voltage)
end
println("Out of withvalues block : ", cav.voltage); 
```

```{code-cell} python
with sb.withvalues(sb.DEFAULT_CONTEXT, vrf=2.0e6):
    print("In withvalues block     :", cav.voltage)
print("Out of withvalues block :", cav.voltage)
```

Thus far, we have only been using the global `DEFAULT_CONTEXT`. However, all `Beamline`s have their
own `context` property to store beamline-specific parameters, which the deferred expressions of the
elements in the beamline are evaluated with. If a variable is not found in the beamline context, it
will fall back to `DEFAULT_CONTEXT` if the variable is defined there. All of the above machinery
({jl}`push!`{py}`push_`, {jl}`pop!`{py}`pop_`, `withvalues`) also works with individual beamline
contexts.

```{code-cell} julia
DEFAULT_CONTEXT.kq = 0.1

qf = Quadrupole(Kn1=DefExpr(c -> c.kq), L=0.5);
bl = Beamline([qf]);

child_qf = bl[qf]
println("Before setting Beamline context ---------------")
println("Parent `qf` (not in a beamline): ", qf.Kn1)
println("Child `qf` in the beamline:      ", child_qf.Kn1) 

# Set kq in the beamline context
push!(bl.context, kq = 0.2)

println("After setting Beamline context ----------------")
println("Parent `qf` (not in a beamline): ", qf.Kn1)
println("Child `qf` in the beamline:      ", child_qf.Kn1) 
```

```{code-cell} python
sb.DEFAULT_CONTEXT.kq = 0.1

qf = sb.Quadrupole(Kn1=sb.DefExpr(lambda c: c.kq), L=0.5)
bl = sb.Beamline([qf])

child_qf = bl[qf]
print("Before setting Beamline context ---------------")
print("Parent `qf` (not in a beamline):", qf.Kn1)
print("Child `qf` in the beamline:     ", child_qf.Kn1)

# Set kq in the beamline context
sb.push_(bl.context, kq=0.2)

print("After setting Beamline context ----------------")
print("Parent `qf` (not in a beamline):", qf.Kn1)
print("Child `qf` in the beamline:     ", child_qf.Kn1)
```

```{docstring} Context
```

```{docstring} withvalues
```

```{docstring} DefExpr
```

## Getting the Deferred Expression Itself

Getting a parameter of an element always gives its current value: any deferred expression is
evaluated first, as above. Sometimes you want the deferred expression itself instead, for
example to build a new one from it. `rawget` gets a parameter exactly as it was set, and
`deval` evaluates it in a context:

```{code-cell} julia
d = rawget(qf, :Kn1);
println(d isa DefExpr)
println(deval(d, bl.context)) # The same as bl[qf].Kn1
```

```{code-cell} python
d = sb.rawget(qf, "Kn1")
print(isinstance(d, sb.DefExpr))
print(sb.deval(d, bl.context))  # The same as bl[qf].Kn1
```

For example, to make `qf` 10% stronger while it keeps following `kq` in the context:

```{code-cell} julia
qf.Kn1 = 1.1*rawget(qf, :Kn1)
println(bl[qf].Kn1)
bl.context.kq = 0.4
println(bl[qf].Kn1)
```

```{code-cell} python
qf.Kn1 = 1.1*sb.rawget(qf, "Kn1")
print(bl[qf].Kn1)
bl.context.kq = 0.4
print(bl[qf].Kn1)
```

Only the parameters stored as independent variables (set explicitly by the user) can be gotten this way. Others are computed from them: here `Kn1` was set, so e.g. the integrated strength `Kn1L` and the field gradient `Bn1` are computed from `Kn1` (and the length or the reference energy). `rawget` gives an error for these, saying which parameter to use instead:

```{code-cell} julia
:tags: [raises-exception]
rawget(qf, :Kn1L)
```

```{code-cell} python
:tags: [raises-exception]
sb.rawget(qf, "Kn1L")
```

```{docstring} rawget
```

```{docstring} deval
```
