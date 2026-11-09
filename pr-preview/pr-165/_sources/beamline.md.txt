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

Alternatively, the reference species and energy may be provided in the beginning
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

The reference species and energy describe the particle the beamline is designed for, and
normalized strengths such as `Kn1` are relative to them. However, you can still track any other
species through the beamline, for example muons through an electron ring. This is just
like sending a different particle through the same real accelerator: nothing about the
machine changes (the magnets have the same fields, and the RF cavities the same frequency and
phase), and each particle responds to it according to its own charge, mass, and energy (see
[Track](track.md)).

```{docstring} Beamline
```

## Finding LatElements in a Beamline

Continuing with the above example, there are several ways to find `LatElement`s in a
`Beamline`, by indexing the beamline:

```{code-cell} julia
# Find all instances of the drift `d`
drifts = fodo[d];

# Get the element at index 4
fodo[4];

# Provide an anonymous function to search for elements
quads = fodo[x -> x.kind == "Quadrupole"];
focusing_sextupoles = fodo[x -> x.Kn2 > 0];
```

The anonymous function is applied to all elements in the beamline, and if `true` is
returned, then that element is included in the search output.

A `Beamline` can also be looped over like a vector of its elements, for example
`for ele in fodo ... end` or `foreach(x -> x.n_steps = 4, fodo)`.

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

Patterns can be combined with `,` (either one matches) and `&` (both match):

```{code-cell} julia
fodo["qf, qd"];               # qf or qd: qf, qd
fodo["Sextupole::* & qf:qd"]; # Sextupoles from qf through qd: sf
```

For anything more complicated, a Julia [regular expression](https://docs.julialang.org/en/v1/manual/strings/#man-regex-literals)
`r"..."` can be used instead. It must match the whole name, and the `i` flag makes it case
insensitive:

```{code-cell} julia
fodo[r"(q|s)d"]; # qd or sd: qd, sd
fodo[r"Q.*"i];   # Names starting with "q" or "Q": qf, qd
```

The elements found are always in their order in the beamline, each at most once. If the end
of a range comes before its start, as in `fodo["sd:sf"]`, the range wraps around the end of the
beamline, which is convenient for rings. Each end of a range must be exactly one element, so
`fodo["qf:b"]` is an error because `fodo` has two `b`s.

`findelements(fodo, "q*")` is the same as `fodo["q*"]`. It can also search a whole `Lattice`
with several branches, where `branch>>name` matches only in the branches named `branch`:

```{docstring} findelements
```

## Multiple LatElements in (multiple) Beamline

Again continuing with the FODO cell example above, note that `fodo` contains two instances
of the line element `d` and two instances of `b`. If the length of `d` is changed, then both
instances of `d` will see this new, changed length:

```{code-cell} julia
drifts = fodo[d];
d.L = 2.0
println(drifts[1].L)
println(drifts[2].L)
```

However, both drifts in `fodo` are unique elements. We can check this using the
[`===`](https://docs.julialang.org/en/v1/base/base/#Core.:(===)) operator:

```{code-cell} julia
println(drifts[1] === drifts[2])
println(d === drifts[1])
println(d === drifts[2])
```

Under the hood, when an element is placed in a Beamline, a **shallow copy** of that element
is created that points to the "parent" element, from which it inherits its parameters. So,
in this above example when the "get" `drifts[1].L` is executed, the code goes to the parent
element `d` and returns `d.L`. "Sets", such as `drifts[1].L = 10`, will also pass through
from the child to the parent:

```{code-cell} julia
drifts[1].L = 3.0
println(d.L)
println(drifts[2].L)
```

The only case where a child element can have parameters different from its parent is when a
given [parameter group](#pgs) is contained within the child. For example,
`drifts[1]` and `drifts[2]` both have their own instance of `BeamlineParams`, from which we
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

The parent element can be retrieved using `parent`:

```{code-cell} julia
drifts[1].parent
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

The last parameter "set" will always define what the independent variable is. So if we then
set the unnormalized quadrupole strength `Bn1`, that will be the independent variable:

```{code-cell} julia
ele.Bn1 = 10
println(bl[1].Kn1) # Returns Bn1 / p_over_q_ref = 10 / 3
```

Now, if we then change the reference energy of the beamline, `Bn1` will remain constant but
`Kn1` will change:

```{code-cell} julia
bl.p_over_q_ref = 4
println(bl[1].Bn1) # == 10
println(bl[1].Kn1) # Now equals 10 / 4
```
