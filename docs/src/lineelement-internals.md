(developer.lineelement.internals)=
# LineElement Internals

This page describes how a `LineElement` stores its parameters. It is intended for developers
working on SciBmad and its constituent packages. For how to use a `LineElement`, see
[Defining a LineElement](#defining.lineelement).

## The `pdict` field

A `LineElement` struct has a single field, `pdict`, which is a `ParamDict` (an alias for
`Dict{Type{<:AbstractParams}, AbstractParams}`). It stores the element's parameter groups,
each keyed by its own type, e.g. `pdict[BendParams]` is a `BendParams`. Setting an entry
whose value is not of the key type throws an error. An element only holds the parameter
groups that have been set, so e.g. a `Drift` has no `BMultipoleParams`. Every element
constructed with the default `LineElement` constructor starts with a `UniversalParams`.

All properties are stored in, or computed from, the parameter groups in `pdict`:

- Getting a parameter group that is not in `pdict` (e.g. `ele.BendParams`) returns
  `nothing`, and setting a parameter group to `nothing` removes it from `pdict`.
- Getting a property whose parameter group is not in `pdict` returns the default value
  of that property, and setting the property adds the parameter group to `pdict`.
- The elements in a `Beamline` are separate `LineElement`s whose `pdict` holds only an
  `InheritParams`, pointing to the original element, and a `BeamlineParams`. Any other
  parameter group is read from and written to the `pdict` of the original element.

`pdict` should not be accessed directly: `ele.pdict` throws an error. Use
`ele.<parameter group name>` to get or set a whole parameter group instead. Internal code
that needs the dictionary itself uses `getfield(ele, :pdict)`, which bypasses `ProtectParams`
and `InheritParams`.

`fieldnames(LineElement)` is therefore just `(:pdict,)`, while `propertynames(ele)` lists
every name that can be used as `ele.<name>`, i.e. the parameter groups and all of their
properties, including virtual properties such as `Kn1L`.
