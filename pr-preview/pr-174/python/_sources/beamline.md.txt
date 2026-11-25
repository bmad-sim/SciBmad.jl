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

(defining.beamline)=
# Defining a Beamline

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

To construct a `Beamline`, provide an array of `LatElement`s in the proper order, and
optionally specify the reference species as `species_ref` and the reference energy as one of
`E_ref`, `pc_ref`, or `p_over_q_ref`:

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-1.2, L=0.2)
end

fodo = Beamline([qf, sf, b, d, qd, sd, b, d],
        species_ref=Species("electron"), pc_ref=18e9)
```

```{code-cell} python
with sb.elements():
    qf = sb.Quadrupole(Kn1=0.36, L=0.5)
    sf = sb.Sextupole(Kn2=1.2, L=0.2)
    d = sb.Drift(L=0.1)
    b = sb.SBend(L=1.2, angle=np.pi/132)
    qd = sb.Quadrupole(Kn1=-0.36, L=0.5)
    sd = sb.Sextupole(Kn2=-1.2, L=0.2)

fodo = sb.Beamline([qf, sf, b, d, qd, sd, b, d],
                   species_ref=sb.Species("electron"), pc_ref=18e9)
fodo
```

Alternatively, the reference species and energy may be provided in the first
`LatElement`. So, we could have equivalently written this as:

```{code-cell} julia
@elements begin
  beg = Marker(species_ref=Species("electron"), pc_ref=18e9)
  qf = Quadrupole(Kn1=0.36, L=0.5)
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-1.2, L=0.2)
end

fodo = Beamline([beg, qf, sf, b, d, qd, sd, b, d])
```

```{code-cell} python
with sb.elements():
    beg = sb.Marker(species_ref=sb.Species("electron"), pc_ref=18e9)
    qf = sb.Quadrupole(Kn1=0.36, L=0.5)
    sf = sb.Sextupole(Kn2=1.2, L=0.2)
    d = sb.Drift(L=0.1)
    b = sb.SBend(L=1.2, angle=np.pi/132)
    qd = sb.Quadrupole(Kn1=-0.36, L=0.5)
    sd = sb.Sextupole(Kn2=-1.2, L=0.2)

fodo = sb.Beamline([beg, qf, sf, b, d, qd, sd, b, d])
fodo
```

The reference species and energy describe the particle the beamline is designed for, and
normalized strengths such as `Kn1` are relative to them. However, you can still track any other
species through the beamline, for example muons through an electron ring. The behavior is equivalent to what happens in real life: nothing about the machine changes (the magnets have the same fields, and the RF cavities the same frequency and phase), and each particle responds to it according to its own charge, mass, and energy (see [Track](track.md)).

```{docstring} Beamline
```

## Finding LatElements in a Beamline

Continuing with the above example, there are several ways to find `LatElement`s in a `Beamline`, by indexing the beamline:

```{code-cell} julia
# Find the *single* instance of `qf` in fodo:
qf_in_fodo = fodo[qf];

# Find *all* the drifts `d` in `fodo` (note the extra [])
ds = fodo[[d]];

# Get the element at index 4
fodo[4];

# Provide an anonymous function to search for elements
quads = fodo[x -> x.kind == "Quadrupole"];
focusing_sextupoles = fodo[x -> x.Kn2 > 0];
```

```{code-cell} python
# Find the *single* instance of `qf` in fodo:
qf_in_fodo = fodo[qf]

# Find *all* the drifts `d` in `fodo` (note the extra [])
ds = fodo[[d]]

# Get the element at index 4
fodo[4]

# Provide an anonymous function to search for elements
quads = fodo[lambda x: x.kind == "Quadrupole"]
focusing_sextupoles = fodo[lambda x: x.Kn2 > 0]
```

`fodo[d]` would be an error, because `d` is placed twice: index with `[[d]]` for all of its
placements, or with a beamline index for one of them. A vector can also mix elements and
beamline indexes, e.g. `fodo[[qf, 4]]`. These are the same rules as for reading `twiss` and
`track` results at elements.

The anonymous function is applied to all elements in the beamline, and if {jl}`true`{py}`True` is
returned, then that element is included in the search output.

:::{only} python
Beamline indexes (the positions of the elements in the beamline, as in `fodo[4]`) start at 1,
as in Julia and as the tables of SciBmad print them. Negative indexes count from the end
(`fodo[-1]` is the last element), and `fodo[2:5]` are the elements 2, 3 and 4 (a Python
slice). The vectors that SciBmad returns, such as `ds`, are Python sequences, which start at
0 as always in Python (`ds[0]` is the first drift `d`).
:::

:::{only} julia
A `Beamline` can also be looped over like a vector of its elements, for example
`for ele in fodo ... end` or `foreach(x -> x.n_steps = 4, fodo)`.
:::
:::{only} python
A `Beamline` can also be looped over like a list of its elements, for example
`for ele in fodo: ele.n_steps = 4`.
:::

(search:name)=
### Searching by Name

Elements can also be found by their names, using a string pattern with the same syntax as in
Bmad. `*` matches any number of characters, and `%` or `?` any single character. Putting an
element kind and `::` in front matches only elements of that kind, `#N` at the end picks the
`N`th match, and `name1:name2` gives all elements from `name1` through `name2`:

```{code-cell} julia
fodo["q*"];           # Names starting with "q": qf, qd
fodo["?d"];           # Two character names ending with "d": qd, sd
fodo["Sextupole::*"]; # All sextupoles: sf, sd
fodo["d#2"];          # The second d (element 9)
fodo["qf:qd"];        # Everything from qf through qd: qf, sf, b, d, qd
fodo["5:9"];          # Elements 5 through 9: d, qd, sd, b, d
```

```{code-cell} python
fodo["q*"]            # Names starting with "q": qf, qd
fodo["?d"]            # Two character names ending with "d": qd, sd
fodo["Sextupole::*"]  # All sextupoles: sf, sd
fodo["d#2"]           # The second d (element 9)
fodo["qf:qd"]         # Everything from qf through qd: qf, sf, b, d, qd
fodo["5:9"];          # Elements 5 through 9: d, qd, sd, b, d
```

Patterns can be combined with `,` (either one matches) and `&` (both match):

```{code-cell} julia
fodo["qf, qd"];               # qf or qd: qf, qd
fodo["Sextupole::* & qf:qd"]; # Sextupoles from qf through qd: sf
```

```{code-cell} python
fodo["qf, qd"]                # qf or qd: qf, qd
fodo["Sextupole::* & qf:qd"];  # Sextupoles from qf through qd: sf
```

:::{only} julia
For anything more complicated, a Julia [regular expression](https://docs.julialang.org/en/v1/manual/strings/#man-regex-literals) `r"..."` can be used instead. It must match the whole name, and the `i` flag makes it case insensitive:
:::
:::{only} python
For anything more complicated, a [regular expression](https://docs.python.org/3/library/re.html)
`re.compile("...")` can be used instead (it is passed to Julia, whose regular expressions have
nearly the same syntax as Python's). It must match the whole name, and the `re.I` flag makes it case
insensitive:
:::

```{code-cell} julia
fodo[r"(q|s)d"]; # qd or sd: qd, sd
fodo[r"Q.*"i];   # Names starting with "q" or "Q": qf, qd
```

```{code-cell} python
import re

fodo[re.compile("(q|s)d")]        # qd or sd: qd, sd
fodo[re.compile("Q.*", re.I)];    # Names starting with "q" or "Q": qf, qd
```

The elements found are always in their order in the beamline, each at most once. If the end
of a range comes before its start, as in `fodo["sd:sf"]`, the range wraps around the end of the
beamline, which is convenient for rings. Each end of a range must be exactly one element, so
`fodo["qf:b"]` is an error because `fodo` has two `b`s.

`findelements(fodo, "q*")` is the same as `fodo["q*"]`. It can also search a whole `Lattice`
with several branches, where `branch>>name` matches only in the branches named `branch`:

```{docstring} findelements
```

## Multiple LatElements in (multiple) Beamlines

Again continuing with the FODO cell example above, note that `fodo` contains two instances
of the element `d` and two instances of `b`. If the length of `d` is changed, then both
instances of `d` will see this new, changed length:

```{code-cell} julia
drifts = fodo[[d]];
d.L = 2.0
println(drifts[1].L)
println(drifts[2].L)
```

```{code-cell} python
drifts = fodo[[d]]
d.L = 2.0
print(drifts[0].L)
print(drifts[1].L)
```

However, both drifts in `fodo` are unique elements. We can check this using the
{jl}`===`{py}`is` operator:

```{code-cell} julia
println(drifts[1] === drifts[2])
println(d === drifts[1])
println(d === drifts[2])
```

```{code-cell} python
print(drifts[0] is drifts[1])
print(d is drifts[0])
print(d is drifts[1])
```

Under the hood, when an element is placed in a Beamline, a **shallow copy** of that element
is created that points to the "parent" element, from which it inherits its parameters. So,
in this above example when the "get" {jl}`drifts[1].L`{py}`drifts[0].L` is executed, the code goes
to the parent
element `d` and returns `d.L`. "Sets", such as {jl}`drifts[1].L = 10`{py}`drifts[0].L = 10`, will
also pass through
from the child to the parent:

```{code-cell} julia
drifts[1].L = 3.0
println(d.L)
println(drifts[2].L)
```

```{code-cell} python
drifts[0].L = 3.0
print(d.L)
print(drifts[1].L)
```

The only case where a child element can have parameters different from its parent is when a
given [parameter group](#pgs) is contained within the child. For example,
{jl}`drifts[1]`{py}`drifts[0]` and {jl}`drifts[2]`{py}`drifts[1]` both have their own instance of
`BeamlineParams`, from which we
can extract things like `beamline_index`, `s`, and `s_downstream`. On the other hand, the
parent element `d` does *not* have a `BeamlineParams`.

```{code-cell} julia
println("beamline_index:")
println(drifts[1].beamline_index)
println(drifts[2].beamline_index)

println("s_downstream:")
println(drifts[1].s_downstream)
println(drifts[2].s_downstream)

# This will error:
try
  d.beamline_index
catch err
  println(err)
end
```

```{code-cell} python
print("beamline_index:")
print(drifts[0].beamline_index)
print(drifts[1].beamline_index)

print("s_downstream:")
print(drifts[0].s_downstream)
print(drifts[1].s_downstream)

# This will error:
try:
    d.beamline_index
except AttributeError as err:
    print(err)
```

The parent element can be retrieved using `parent`:

```{code-cell} julia
drifts[1].parent
```

```{code-cell} python
drifts[0].parent
```

## Beamline-Dependent LatElement Parameters

Finally, elements in a beamline allow one to "get" parameters that may only be defined when
said element is in a beamline. We showed the `s` and `s_downstream`, but another example
would be the unnormalized magnetic field, if the normalized magnetic field is stored as an
independent variable:

```{code-cell} julia
ele = Quadrupole(Kn1=2, L=2)
bl = Beamline([ele], p_over_q_ref=3)
println(bl[1].Bn1) # Returns Kn1 * p_over_q_ref = 2 * 3
```

```{code-cell} python
ele = sb.Quadrupole(Kn1=2, L=2)
bl = sb.Beamline([ele], p_over_q_ref=3)
print(bl[1].Bn1)  # Returns Kn1 * p_over_q_ref = 2 * 3
```

The last parameter "set" will always define what the independent variable is. So if we then
set the unnormalized quadrupole strength `Bn1`, that will be the independent variable:

```{code-cell} julia
ele.Bn1 = 10
println(bl[1].Kn1) # Returns Bn1 / p_over_q_ref = 10 / 3
```

```{code-cell} python
ele.Bn1 = 10
print(bl[1].Kn1)  # Returns Bn1 / p_over_q_ref = 10 / 3
```

Now, if we then change the reference energy of the beamline, `Bn1` will remain constant but
`Kn1` will change:

```{code-cell} julia
bl.p_over_q_ref = 4
println(bl[1].Bn1) # == 10
println(bl[1].Kn1) # Now equals 10 / 4
```

```{code-cell} python
bl.p_over_q_ref = 4
print(bl[1].Bn1)  # == 10
print(bl[1].Kn1)  # Now equals 10 / 4
```
