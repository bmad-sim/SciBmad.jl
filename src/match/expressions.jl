# Expressions of lattice quantities. Nothing here computes: the nodes are evaluated by
# value/value_and_jacobian (engine.jl)

"""
    LatticeExpr

An expression of lattice quantities: a quantity of a [`TwissExpr`](@ref) (`tw.beta1[ele]`), a
[`Knob`](@ref), or arithmetic and functions of them (`tw.beta1[ele]^2 - 5*mean(tw.alpha1[ele, :])`).
It is computed by [`value`](@ref), [`value_and_jacobian`](@ref) or [`match!`](@ref).
"""
abstract type LatticeExpr end

Base.broadcastable(e::LatticeExpr) = Ref(e)
# == makes a Target (see match.jl), so that collections keep working isequal is identity
Base.isequal(a::LatticeExpr, b::LatticeExpr) = a === b

"""
    Knob(obj, property; lower=-Inf, upper=Inf, step=Inf, name)
    Knob(obj, property, index; lower=-Inf, upper=Inf, step=Inf, name)
    Knob(get, set; lower=-Inf, upper=Inf, step=Inf, name="knob")
    @knob obj.property lower=... upper=...
    @knob obj.property[index] lower=... upper=...

Something that can be varied: the `property` of `obj` (a `LatElement`, a `Context`, a
`Controller`, or anything else with `obj.property` to get and set a number), or a pair of
functions `get()` and `set(value)`. `lower` and `upper` are the bounds used by `match!` and
`step` the largest change per iteration.

If the property is a vector of numbers (e.g. a variable of a `Context` holding the strengths of
several correctors), the knob stands for one knob per entry, named `property[1]`,
`property[2]`, ..., with the same bounds: `match!`, `value_and_jacobian` and `setknobs!` vary
each entry. `index` (or `@knob obj.property[index]`) is one entry alone.

A `Knob` is also a [`LatticeExpr`](@ref), so it can be used in expressions and targets.

## Examples
```julia
k1 = Knob(qf, :Kn1; lower=0, upper=1)
k2 = @knob ring.context.kqd
k3 = @knob ring.context.coilstrengths # one knob per entry of the vector
k4 = @knob ring.context.coilstrengths[2]
```
"""
struct Knob <: LatticeExpr
  obj::Any
  prop::Union{Symbol,Nothing}
  index::Int # an entry of a vector property, or 0: the property itself
  get::Any
  set::Any
  name::String
  lower::Float64
  upper::Float64
  step::Float64
end

function Knob(obj, prop::Union{Symbol,AbstractString}, index::Integer=0; lower=-Inf, upper=Inf,
              step=Inf, name=_knob_name(obj, Symbol(prop), index))
  lower <= upper || error("Knob $name: lower must be <= upper")
  return Knob(obj, Symbol(prop), Int(index), nothing, nothing, name, lower, upper, step)
end
function Knob(get::Function, set::Function; lower=-Inf, upper=Inf, step=Inf, name="knob")
  lower <= upper || error("Knob $name: lower must be <= upper")
  return Knob(nothing, nothing, 0, get, set, name, lower, upper, step)
end
Knob(p::Pair; kwargs...) = Knob(p.first, p.second; kwargs...)
Knob(k::Knob) = k

_knob_name(obj, prop) = String(prop)
function _knob_name(obj::LatElement, prop)
  name = (obj.UniversalParams::UniversalParams).name
  return isempty(name) ? String(prop) : name * "." * String(prop)
end
_knob_name(obj, prop, index) = index == 0 ? _knob_name(obj, prop) :
                                            string(_knob_name(obj, prop), "[", index, "]")

function _getknob(k::Knob)
  isnothing(k.prop) && return k.get()
  x = getproperty(k.obj, k.prop)
  return k.index == 0 ? x : x[k.index]
end
function _setknob!(k::Knob, v)
  if isnothing(k.prop)
    k.set(v)
  elseif k.index == 0
    setproperty!(k.obj, k.prop, v)
  else
    # An entry of a vector: in place if the vector can hold v, else in a copy that can
    a = getproperty(k.obj, k.prop)
    T = promote_type(eltype(a), typeof(v))
    if T === eltype(a)
      a[k.index] = v
    else
      b = copyto!(similar(a, T), a)
      b[k.index] = v
      setproperty!(k.obj, k.prop, b)
    end
  end
  return nothing
end
_sameknob(a::Knob, b::Knob) = isnothing(a.prop) ? a.set === b.set :
                              (a.obj === b.obj && a.prop === b.prop && a.index == b.index)

"""
    @knob obj.property [lower=...] [upper=...] [step=...] [name=...]
    @knob obj.property[index] [lower=...] [upper=...] [step=...] [name=...]

The same as `Knob(obj, :property; lower=..., upper=..., step=..., name=...)`, or with `index`,
`Knob(obj, :property, index; ...)`.
"""
macro knob(ex, kws...)
  index = 0
  if ex isa Expr && ex.head == :ref && length(ex.args) == 2
    index = esc(ex.args[2])
    ex = ex.args[1]
  end
  (ex isa Expr && ex.head == :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode) ||
    error("@knob expects `object.property` or `object.property[index]`")
  kwargs = map(kws) do kw
    (kw isa Expr && kw.head == :(=)) || error("@knob expects keyword arguments `name=value`")
    Expr(:kw, kw.args[1], esc(kw.args[2]))
  end
  return :(Knob($(esc(ex.args[1])), $(ex.args[2]), $index; $(kwargs...)))
end

# A function of expressions (and numbers)
struct _Call <: LatticeExpr
  f::Any
  args::Vector{Any}
end

# Derivative of an expression in a knob
struct _Deriv <: LatticeExpr
  arg::LatticeExpr
  knob::Knob
end

# A function of the lattice itself, evaluated while the knobs carry their derivatives
struct _Getter <: LatticeExpr
  f::Any
end

"""
    TwissExpr(bl::Beamline; kwargs...)

The quantities of `twiss(bl; kwargs...)` as expressions ([`LatticeExpr`](@ref)), computed only
when they are needed (by [`value`](@ref), [`value_and_jacobian`](@ref) or [`match!`](@ref)), and
then only at the locations used. `kwargs` are those of `twiss` (`rf_on`, `a_initial`, `spin`,
`delta0`, ...) except `at`, `cols`, `base_cols`, `as_taylor_series` and `GTPSA_descriptor`, which
are set from the expressions. The truncation orders are chosen from the quantities used; `order`
can be given to use a higher one.

`a_initial` may be the `a` of another `TwissExpr`, e.g. `TwissExpr(line; a_initial=tw.a[end])`,
to continue the Twiss functions of another beamline. `initial` may be a `Context`, whose variables
can then be knobs, e.g. to match the Twiss parameters at the start of a line:
`ini = Context(beta1=10.0, beta2=10.0)`, `TwissExpr(line; initial=ini)`, `@knob(ini.beta1)`.

## Quantities
- Summary values: `tw.q1`, `tw.q2`, `tw.q3`, `tw.etac`, `tw.alphac`, `tw.qspin`, `tw.damp1`, ...
  A derivative of an amplitude-dependent value is selected as for the result of `twiss`, e.g.
  `tw.q1[delta=2]` (∂²q1/∂δ²) or `tw.q1[J1=1]`; `tw.dq1` and `tw.dq2` are short for the
  chromaticities `tw.q1[delta=1]` and `tw.q2[delta=1]`, `tw.dq1_n` for `tw.q1[delta=n]`.
- Columns: `tw.<column>[location]` for every scalar column of `twiss`, including chromatic
  derivatives (`tw.dbeta1`, `tw.dx_2`) and `tw.h<ijkl>`.

## Locations
The same as for the columns of the result of `twiss` (see [`TwissColumn`](@ref)):
- `tw.beta1[7]`, `tw.beta1[ele]`: the beginning of the element with that beamline index (one
  expression), or of every placement of the element (a vector, also if it is placed once:
  `only(tw.beta1[ele])` is then the expression itself).
- `tw.beta1[ele, :exit]`: the end of the element.
- `tw.beta1[ele, :]`: every integration step of the element, both ends included (a vector).
  `tw.beta1[ele, k]`: after `k` integration steps.
- `tw.beta1[end]`, `tw.beta1[begin]`: the end and the beginning of the beamline (or of the part
  given with `start` and `stop`, which vectors are restricted to). `end` is a beamline index:
  `tw.beta1[end-1]` is the beginning of the last element.
- `tw.beta1["q*"]`, `tw.beta1[r"..."]`: elements by name, as for `findelements` (a vector).
- `tw.beta1[(s1, s2)]`, `tw.beta1[:]`: every integration step in an s-range, or of the whole
  beamline (a vector).

Vectors are elementwise: `tw.beta1["q*"]^2`, `tw.beta1["q*"] <= 30` (one target per element),
with `sum`, `mean`, `maximum`, `minimum` and `norm`.
"""
struct TwissExpr
  bl::Beamline
  kwargs::Dict{Symbol,Any}
  order::Int
end

function TwissExpr(bl::Beamline; order::Integer=0, chrom::Integer=0, kwargs...)
  for k in (:at, :cols, :base_cols, :as_taylor_series, :GTPSA_descriptor)
    haskey(kwargs, k) && error("TwissExpr sets `$k` itself from the expressions used")
  end
  return TwissExpr(bl, Dict{Symbol,Any}(kwargs), max(order, chrom))
end

const _TWISS_SUMMARY = (:q1, :q2, :q3, :etac, :alphac, :qspin, :damp1, :damp2, :damp3)

# A quantity of a TwissExpr: a summary value (index 0), or a column at the beginning of the
# element `index` (-1: the end of the beamline) plus `step` integration steps
struct _TwissLeaf <: LatticeExpr
  tw::TwissExpr
  col::String
  index::Int
  step::Int
  mono::NTuple{4,Int} # (delta, J1, J2, J3) of a summary value
end

struct _TwissExprColumn
  tw::TwissExpr
  col::String
end

Base.propertynames(tw::TwissExpr) = [_TWISS_SUMMARY..., :dq1, :dq2,
  (Symbol(k) for k in keys(_TWISS_FCN_MAP) if !(k in ("index", "name", "kind")))...]

function Base.getproperty(tw::TwissExpr, key::Symbol)
  if key in _TWISS_SUMMARY
    return _TwissLeaf(tw, String(key), 0, 0, (0, 0, 0, 0))
  elseif (m = match(r"^dq([12])(?:_([1-9]))?$", String(key))) !== nothing
    # The chromaticities dq1, dq2 and their derivatives dq1_n = ∂ⁿq1/∂δⁿ
    n = isnothing(m.captures[2]) ? 1 : parse(Int, m.captures[2])
    return _TwissLeaf(tw, "q" * m.captures[1], 0, 0, (n, 0, 0, 0))
  end
  col = String(key)
  _col_order(col) # errors for an unknown column
  return _TwissExprColumn(tw, col)
end

Base.show(io::IO, tw::TwissExpr) = print(io, "TwissExpr of a Beamline with ",
  length(getfield(tw, :bl).line), " elements", isempty(getfield(tw, :kwargs)) ? "" :
  " (" * join(("$k=$(v isa LatticeExpr || v isa Number || v isa Bool ? v : typeof(v))"
               for (k, v) in getfield(tw, :kwargs)), ", ") * ")")
Base.show(io::IO, c::_TwissExprColumn) = print(io, "TwissExpr column ", c.col,
                                          " (index it with a location, e.g. [ele] or [end])")

const _MATRIX_COLS = ("N", "Vi", "H1", "H2", "H3", "B1", "B2", "B3", "E1", "E2", "E3")
const _MAP_COLS = ("a", "a0", "a1", "a2", "as")

# Phase space order a column needs without parameters (as checked in _twiss_map_fcn)
function _col_order(col)
  if haskey(_TWISS_FCN_MAP, col)
    col in ("index", "name", "kind") && error("`$col` is not a number")
    col in _MATRIX_COLS && error("Matrix column `$col` is not supported in expressions yet")
    return col in ("w1", "w2", "w1a", "w1b", "w2a", "w2b") ? 2 : 1
  end
  m = match(r"^h([0-9]{4,6})$", col)
  if !isnothing(m)
    return max(1, sum(parse(Int, c) for c in m.captures[1]) - 1)
  end
  m = match(r"^d(.+?)(?:_([1-9]))?$", col)
  if !isnothing(m)
    ccol = m.captures[1]
    n = isnothing(m.captures[2]) ? 1 : parse(Int, m.captures[2])
    mh = match(r"^h([0-9]{4,6})$", ccol)
    if !isnothing(mh)
      return sum(parse(Int, c) for c in mh.captures[1]) - 1 + n
    elseif haskey(_TWISS_FCN_MAP, ccol) && !(ccol in ("index", "name", "kind", "s")) &&
           !(ccol in _MATRIX_COLS) && !(ccol in _MAP_COLS)
      if ccol in ("x", "px", "y", "py", "n0x", "n0y", "n0z", "nx", "ny", "nz")
        return n
      elseif ccol in ("w1", "w2", "w1a", "w1b", "w2a", "w2b")
        return n + 2
      else
        return n + 1
      end
    end
  end
  error("TwissExpr has no quantity `$col`. See the `twiss` docstring for the columns.")
end

function _leaf_order(l::_TwissLeaf)
  if l.index == 0
    # (the degree of the map terms that give the term, as in _adv_get: the spin tune comes from the
    # constant part of the spin map, one order less than the orbital tunes)
    delta, J1, J2, J3 = l.mono
    degree0 = l.col == "qspin" ? 0 : 1
    return l.mono == (0, 0, 0, 0) ? 1 : max(1, degree0 + delta + 2*(J1 + J2 + J3))
  end
  return _col_order(l.col)
end

# Term selection of a summary value, as getterm
function Base.getindex(l::_TwissLeaf; delta::Integer=0, J1::Integer=0, J2::Integer=0,
                       J3::Integer=0)
  l.index == 0 || error("Only summary values (q1, q2, ...) have terms in delta, J1, J2, J3")
  l.mono == (0, 0, 0, 0) || error("$(l) is already a derivative: give all of them at once, e.g. " *
                                  "tw.q1[delta=1, J1=1]")
  return _TwissLeaf(l.tw, l.col, 0, 0, (delta, J1, J2, J3))
end

# The beamline indexes of the beginning and the end of the part of the beamline (`start`, `stop`;
# n+1: the end of the beamline), `begin` and `end` of the column
# (with more than one index, e.g. beta1[end, :exit], `end` is lastindex(c, 1))
function _part(c::_TwissExprColumn)
  bl = getfield(c.tw, :bl)
  start = get(getfield(c.tw, :kwargs), :start, nothing)
  stop = get(getfield(c.tw, :kwargs), :stop, nothing)
  return (isnothing(start) ? 1 : _element_index(bl, start, "start"),
          isnothing(stop) ? length(bl.line) + 1 : _element_index(bl, stop, "stop"))
end
Base.firstindex(c::_TwissExprColumn, d...) = _part(c)[1]
Base.lastindex(c::_TwissExprColumn, d...) = _part(c)[2]

"""
    ExprVector

A vector of [`LatticeExpr`](@ref)s with elementwise arithmetic and comparisons, as returned by a
[`TwissExpr`](@ref) column indexed with several locations.
"""
struct ExprVector <: AbstractVector{LatticeExpr}
  v::Vector{LatticeExpr}
end
Base.size(a::ExprVector) = size(a.v)
Base.getindex(a::ExprVector, i::Int) = a.v[i]
Base.getindex(a::ExprVector, i::AbstractVector{<:Integer}) = ExprVector(a.v[i])
Base.getindex(a::ExprVector, ::Colon) = a
Base.IndexStyle(::Type{ExprVector}) = IndexLinear()

function Base.getindex(c::_TwissExprColumn, args...)
  isempty(args) && error("Index the column with a location, e.g. [qd], [5] or [end]")
  locs, one = _column_locations(getfield(c.tw, :bl), _part(c), args...)
  leaves = LatticeExpr[_TwissLeaf(c.tw, c.col, l[1], l[2], (0, 0, 0, 0)) for l in locs]
  return one ? only(leaves) : ExprVector(leaves)
end

"""
    lazy(f)
    lazy(f, args...)

An expression ([`LatticeExpr`](@ref)) from a Julia function.

`lazy(f)` is `f()`, a number computed from the lattice itself, e.g. `lazy(() -> qf.Bn1)` for the
field gradient of `qf`. It is evaluated with the knobs in place, so its derivatives in the knobs
are exact.

`lazy(f, args...)` is `f` of the values of the expressions (or numbers) `args`. `f` must compute
its result from these arguments only.
"""
lazy(f) = _Getter(f)
lazy(f, args...) = _Call(f, Any[args...])

"""
    deriv(expr::LatticeExpr, knob)

The derivative of the expression `expr` in the `Knob` `knob` (which does not have to be varied),
itself an expression. E.g. `deriv(tw.x[bpm], Knob(kicker, :Kn0L))` is an element of the orbit
response matrix. It can be nested for higher derivatives.
"""
GTPSA.deriv(e::LatticeExpr, knob) = _Deriv(e, Knob(knob))
GTPSA.deriv(e::AbstractArray{<:LatticeExpr}, knob) = (k = Knob(knob); map(x -> _Deriv(x, k), e))

# Functions that also work for the Taylor series the expressions are evaluated with
_eabs(x) = scalar(x) isa Complex ? sqrt(real(x)^2 + imag(x)^2) : (scalar(x) >= 0 ? x : -x)
_eabs2(x) = scalar(x) isa Complex ? real(x)^2 + imag(x)^2 : x^2
_emax(x, y) = scalar(x) >= scalar(y) ? x : y
_emin(x, y) = scalar(x) <= scalar(y) ? x : y
_emax(x, y, z...) = _emax(_emax(x, y), z...)
_emin(x, y, z...) = _emin(_emin(x, y), z...)
_esum(x...) = +(x...)

const _ExprOrReal = Union{LatticeExpr,Real}

for op in (:+, :-, :*, :/, :^)
  @eval begin
    Base.$op(a::LatticeExpr, b::LatticeExpr) = _Call($op, Any[a, b])
    Base.$op(a::LatticeExpr, b::Real) = _Call($op, Any[a, b])
    Base.$op(a::Real, b::LatticeExpr) = _Call($op, Any[a, b])
    Base.$op(a::ExprVector, b::LatticeExpr) = ExprVector(LatticeExpr[$op(x, b) for x in a.v])
    Base.$op(a::ExprVector, b::Real) = ExprVector(LatticeExpr[$op(x, b) for x in a.v])
    Base.$op(a::LatticeExpr, b::ExprVector) = ExprVector(LatticeExpr[$op(a, x) for x in b.v])
    Base.$op(a::Real, b::ExprVector) = ExprVector(LatticeExpr[$op(a, x) for x in b.v])
    Base.$op(a::ExprVector, b::ExprVector) = _elementwise($op, a, b)
    Base.$op(a::ExprVector, b::AbstractVector{<:Real}) = _elementwise($op, a, b)
    Base.$op(a::AbstractVector{<:Real}, b::ExprVector) = _elementwise($op, a, b)
  end
end
Base.:^(a::LatticeExpr, b::Integer) = _Call(^, Any[a, b])
Base.:^(a::ExprVector, b::Integer) = ExprVector(LatticeExpr[x^b for x in a.v])
Base.:-(a::LatticeExpr) = _Call(-, Any[a])
Base.:+(a::LatticeExpr) = a
Base.:-(a::ExprVector) = ExprVector(LatticeExpr[-x for x in a.v])
Base.:+(a::ExprVector) = a

function _elementwise(op, a, b)
  length(a) == length(b) || error("Vectors of expressions must have the same length " *
                                  "($(length(a)) and $(length(b)))")
  return ExprVector(LatticeExpr[op(x, y) for (x, y) in zip(a, b)])
end

for f in (:sqrt, :exp, :log, :sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh,
          :real, :imag)
  @eval begin
    Base.$f(a::LatticeExpr) = _Call($f, Any[a])
    Base.$f(a::ExprVector) = ExprVector(LatticeExpr[$f(x) for x in a.v])
  end
end
Base.abs(a::LatticeExpr) = _Call(_eabs, Any[a])
Base.abs(a::ExprVector) = ExprVector(LatticeExpr[abs(x) for x in a.v])
Base.abs2(a::LatticeExpr) = _Call(_eabs2, Any[a])
Base.abs2(a::ExprVector) = ExprVector(LatticeExpr[abs2(x) for x in a.v])
Base.atan(a::LatticeExpr, b::_ExprOrReal) = _Call(atan, Any[a, b])
Base.atan(a::Real, b::LatticeExpr) = _Call(atan, Any[a, b])
Base.max(a::LatticeExpr, b::_ExprOrReal) = _Call(_emax, Any[a, b])
Base.max(a::Real, b::LatticeExpr) = _Call(_emax, Any[a, b])
Base.min(a::LatticeExpr, b::_ExprOrReal) = _Call(_emin, Any[a, b])
Base.min(a::Real, b::LatticeExpr) = _Call(_emin, Any[a, b])

function _nonempty(a)
  isempty(a) && error("Empty vector of expressions")
  return Any[a...]
end
Base.sum(a::AbstractArray{<:LatticeExpr}) = _Call(_esum, _nonempty(a))
Base.maximum(a::AbstractArray{<:LatticeExpr}) = _Call(_emax, _nonempty(a))
Base.minimum(a::AbstractArray{<:LatticeExpr}) = _Call(_emin, _nonempty(a))
Statistics.mean(a::AbstractArray{<:LatticeExpr}) = sum(a)/length(a)
LinearAlgebra.norm(a::AbstractArray{<:LatticeExpr}) = sqrt(sum(map(abs2, a)))

# Printing
function _leaf_name(l::_TwissLeaf)
  if l.index == 0
    terms = ["$n=$p" for (n, p) in zip(("delta", "J1", "J2", "J3"), l.mono) if p != 0]
    return isempty(terms) ? l.col : l.col * "[" * join(terms, ", ") * "]"
  elseif l.index == -1
    return l.col * "[end]"
  end
  name = (getfield(l.tw, :bl).line[l.index].UniversalParams::UniversalParams).name
  loc = isempty(name) ? string(l.index) : name * "#" * string(l.index)
  return l.col * "[" * loc * (l.step == 0 ? "" : ", $(l.step)") * "]"
end
Base.show(io::IO, l::_TwissLeaf) = print(io, _leaf_name(l))
Base.show(io::IO, k::Knob) = print(io, k.name)
Base.show(io::IO, d::_Deriv) = print(io, "deriv(", d.arg, ", ", d.knob, ")")
Base.show(io::IO, ::_Getter) = print(io, "lazy(...)")
_show_arg(io, x) = x isa _Call && x.f in (+, -, *, /) ? print(io, "(", x, ")") : print(io, x)
function Base.show(io::IO, c::_Call)
  if c.f in (+, -, *, /, ^) && length(c.args) == 2
    _show_arg(io, c.args[1])
    print(io, c.f === (^) ? "^" : " $(c.f) ")
    _show_arg(io, c.args[2])
  elseif length(c.args) > 4
    print(io, _fname(c.f), "(", c.args[1], ", …, ", c.args[end], ")")
  else
    print(io, _fname(c.f), "(")
    join(io, c.args, ", ")
    print(io, ")")
  end
end
_fname(f) = f === _eabs ? "abs" : f === _eabs2 ? "abs2" : f === _emax ? "max" :
            f === _emin ? "min" : f === _esum ? "sum" : string(nameof(f))
