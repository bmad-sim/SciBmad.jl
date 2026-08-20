# Defining a Beamline

To construct a `Beamline`, provide an array of `LineElement`s in the proper order, and optionally specify the reference species as `species_ref` and the reference energy as one of `E_ref`, `pc_ref`, or `p_over_q_ref`:

```@example bl
using SciBmad # hide
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

Alternatively, the reference species and energy may be provided in the beginning `LineElement`. So, we could have equivalently written this as:

```@example bl
using SciBmad # hide
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

```@docs; canonical=false
Beamline
```

## Finding LineElements in a Beamline

Continuing with the above example, there are three ways to find `LineElement`s in a `Beamline`, by indexing the beamline:

```@example bl
# Find all instances of the drift `d`
drifts = fodo[d];

# Get the element at index 4
fodo[4];

# Provide an anonymous function to search for elements
quads = fodo[x -> x.kind == "Quadrupole"];
focusing_sextupoles = fodo[x -> x.Kn2 > 0];
```

The anonymous function is applied to all elements in the beamline, and if `true` is returned, then that element is included in the search output.

## Multiple LineElements in (multiple) Beamline

Again continuing with the FODO cell example above, note that `fodo` contains two instances of the line element `d` and two instances of `b` . If the length of `d` is changed, then both instances of `d` will see this new, changed length:

```@example bl
drifts = fodo[d];
d.L = 2.0
println(drifts[1].L)
println(drifts[2].L)
```

However, both drifts in `fodo` are unique elements. We can check this using the [`===`](https://docs.julialang.org/en/v1/base/base/#Core.:(===)) operator:

```@example bl
println(drifts[1] === drifts[2])
println(d === drifts[1])
println(d === drifts[2])
```

Under the hood, when an element is placed in a Beamline, a **shallow copy** of that element is created that points to the "parent" element, from which it inherits its parameters. So, in this above example when the "get" `drifts[1].L` is executed, the code goes to the parent element `d` and returns `d.L`. "Sets", such as `drifts[1].L = 10`, will also pass through from the child to the parent:

```@example bl
drifts[1].L = 3.0
println(d.L)
println(drifts[2].L)
```

The only case where a child element can have parameters different from its parent is when a given [parameter group](@ref pgs) is contained within the child. For example, `drifts[1]` and `drifts[2]` both have their own instance of `BeamlineParams`, from which we can extract things like `beamline_index`, `s`, and `s_downstream`. On the other hand, the parent element `d` does *not* have a `BeamlineParams`.

```@example bl
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

```@example bl
drifts[1].parent
```

## Beamline-Dependent LineElement Parameters

Finally, elements in a beamline allow one to "get" parameters that may only be defined when said element is in a beamline. We showed the `s` and `s_downstream`, but another example would be the unnormalized magnetic field, if the normalized magnetic field is stored as an independent variable:

```@example bl
ele = Quadrupole(Kn1=2, L=2)
bl = Beamline([ele], p_over_q_ref=3)
println(bl[1].Bn1) # Returns Kn1 * p_over_q_ref = 2 * 3
```

The last parameter "set" will always define what the independent variable is. So if we then set the unnormalized quadrupole strength `Bn1`, that will be the independent variable:

```@example bl
ele.Bn1 = 10
println(bl[1].Kn1) # Returns Bn1 / p_over_q_ref = 10 / 3
```

Now, if we then change the reference energy of the beamline, `Bn1` will remain constant but `Kn1` will change:

```@example bl
bl.p_over_q_ref = 4
println(bl[1].Bn1) # == 10
println(bl[1].Kn1) # Now equals 10 / 4
```