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

(rd.control)=
# 11. Deferred Expressions and Contexts

```{code-cell} julia
:tags: [remove-cell]
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

## Overview

Often, parameters of several elements should be controlled together, for example to simulate
control room "knobs" such as a power supply that powers a chain of magnets. Bmad has control
elements for this (`overlay` and `group` elements). In SciBmad, a parameter can be set to a
*deferred expression* (`DefExpr`): a function that computes the value of the parameter from
variables, whenever the value is needed. The variables are kept in a *context*: every `Beamline`
has its own context, `bl.context`. A deferred expression of an element in a beamline is evaluated
with the context of that beamline. See [Deferred Expressions and Contexts](../defexpr.md) for more.

## Example

The two elements of a beamline for muons:

```{code-cell} julia
using SciBmad

@elements begin
  q = Quadrupole(L=1.0)
  b = SBend(L=1.0)
end
lat = Beamline([q, b], species_ref=Species("muon"), pc_ref=1e9)
```

```{code-cell} python
import numpy as np
import scibmad as sb

with sb.elements():
    q = sb.Quadrupole(L=1.0)
    b = sb.SBend(L=1.0)
lat = sb.Beamline([q, b], species_ref=sb.Species("muon"), pc_ref=1e9)
```

The variables `a`, `bb` and `hh` are set in the context of the beamline:

```{code-cell} julia
lat.context.a = 0.02
lat.context.bb = 0.0
lat.context.hh = 0.01
```

```{code-cell} python
lat.context.a = 0.02
lat.context.bb = 0.0
lat.context.hh = 0.01
```

The strength of `q` is $a + bb^2 + 0.7\,hh$, its horizontal offset $0.1\,hh$, and the bend of `b`
(both the curvature of the reference trajectory `g_ref` and the field `Kn0` that follows it) is
$0.1\,a + \tan(bb)$. A deferred expression wraps a function of the context `c`:

```{code-cell} julia
q.Kn1 = DefExpr(c -> c.a + c.bb^2 + 0.7*c.hh)
q.x_offset = DefExpr(c -> 0.1*c.hh)
b.g_ref = DefExpr(c -> 0.1*c.a + tan(c.bb))
b.Kn0 = DefExpr(c -> 0.1*c.a + tan(c.bb))

lat[q].Kn1, lat[q].x_offset, lat[b].g_ref
```

```{code-cell} python
q.Kn1 = sb.DefExpr(lambda c: c.a + c.bb**2 + 0.7*c.hh)
q.x_offset = sb.DefExpr(lambda c: 0.1*c.hh)
b.g_ref = sb.DefExpr(lambda c: 0.1*c.a + np.tan(c.bb))
b.Kn0 = sb.DefExpr(lambda c: 0.1*c.a + np.tan(c.bb))

lat[q].Kn1, lat[q].x_offset, lat[b].g_ref
```

Getting a parameter of an element in the beamline evaluates its expression with the context of the
beamline: here `Kn1` is $0.02 + 0.7 \cdot 0.01 = 0.027$. When a variable changes, so do all the
parameters that depend on it:

```{code-cell} julia
lat.context.hh = 0.02
lat[q].Kn1, lat[q].x_offset
```

```{code-cell} python
lat.context.hh = 0.02
lat[q].Kn1, lat[q].x_offset
```

Notes:

- In Bmad, a parameter controlled by several overlays is the sum of their contributions, and
  cannot be set directly. In SciBmad, the expression of a parameter is one function, which can
  combine any variables. Setting the parameter to a number, e.g. `q.Kn1 = 0.02`, replaces the
  expression (the parameter then no longer depends on the variables).
:::{only} julia
- An expression can be any Julia function of the variables, e.g. an interpolation in a table
  instead of Bmad's "knot" points.
:::
:::{only} python
- An expression can be any Python function of the variables, e.g. an interpolation in a table
  instead of Bmad's "knot" points (`np.interp` of a variable is called from Julia each time,
  which is slower than arithmetic and NumPy's functions of a variable, which become Julia code:
  see [SciBmad from Python](#python.defexpr)).
:::
- The variables of a context can be knobs of a match:
  {jl}`@knob(lat.context.a)`{py}`sb.Knob(lat.context, "a")` (see
  [Matching and Optimization](../optimize.md)). Varying one variable varies all parameters that
  depend on it. This is how families of magnets are matched together in the next chapter.
- To try a value for a while, {jl}`push!(lat.context; hh=0.03)`{py}`sb.push_(lat.context, hh=0.03)`
  sets it and {jl}`pop!(lat.context)`{py}`sb.pop_(lat.context)` sets
  the previous value back.

## Relative Control

Bmad's `group` elements control parameters "relatively": only changes of their variables change the
controlled parameters, which may also be set directly. A deferred expression always gives the
value of a parameter itself. To change a parameter relative to its present value, build a new
expression from it: `rawget` gets the parameter as it was set (a number or a deferred expression),
for example

```{code-cell} julia
b.Kn1 = 0.0
lat.context.z = 0.0
b.Kn1 = rawget(b, :Kn1) + DefExpr(c -> 0.4*sqrt(c.z))
lat.context.z = 0.01
lat[b].Kn1
```

```{code-cell} python
b.Kn1 = 0.0
lat.context.z = 0.0
b.Kn1 = sb.rawget(b, "Kn1") + sb.DefExpr(lambda c: 0.4*np.sqrt(c.z))
lat.context.z = 0.01
lat[b].Kn1
```

Consider the situation where you want to control the chromaticity of a ring (the change of the tune
with the particle energy) by varying sextupole strengths. To change the chromaticity by 1 unit, you
want to change the strengths of many sextupoles by amounts that you compute. You don't care about
the strengths of the sextupoles per se, only about their changes. With a variable `dchrom`, the
strength of each sextupole is then its design strength plus its share of the change:

```julia
s1.Kn2 = rawget(s1, :Kn2) + DefExpr(c -> -6.415e-4*c.dchrom)
```

```python
s1.Kn2 = sb.rawget(s1, "Kn2") + sb.DefExpr(lambda c: -6.415e-4*c.dchrom)
```

## Exercises

1. **Interpolation:** Write an expression for `q.Kn1` that interpolates linearly between the values
   0 and 0.7 for `hh` between 0 and 1, and is constant outside this range.

2. **Moving an element:** Expressions are good for varying the longitudinal positions of elements.
   Make a beamline of a drift `dd` (length 0.5 m), a bend `bb` (length 0.5 m) and a drift `de`
   (length 0.5 m), and write the lengths of the two drifts as expressions of a variable `ds`, so
   that `ds` moves the bend downstream without changing the length of the beamline. Check that the
   total length of the beamline stays the same. (Bmad has the `start_edge` attribute of `group`
   elements for this.)

3. **Girders:** Bmad's `girder` elements misalign several elements together, e.g. magnets that sit
   on a common table. SciBmad has no girders yet. How would you misalign `q` and `b` together by
   the same horizontal offset with an expression?
