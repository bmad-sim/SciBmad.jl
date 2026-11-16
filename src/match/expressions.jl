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
    Knob(get, set; lower=-Inf, upper=Inf, step=Inf, name="knob")
    @knob obj.property lower=... upper=...

Something that can be varied: the `property` of `obj` (a `LatElement`, a `Context`, a
`Controller`, or anything else with `obj.property` to get and set a number), or a pair of
functions `get()` and `set(value)`. `lower` and `upper` are the bounds used by `match!` and
`step` the largest change per iteration.

A `Knob` is also a [`LatticeExpr`](@ref), so it can be used in expressions and targets.

## Examples
```julia
k1 = Knob(qf, :Kn1; lower=0, upper=1)
k2 = @knob ring.context.kqd
```
"""
struct Knob <: LatticeExpr
  obj::Any
  prop::Union{Symbol,Nothing}
  get::Any
  set::Any
  name::String
  lower::Float64
  upper::Float64
  step::Float64
end

function Knob(obj, prop::Union{Symbol,AbstractString}; lower=-Inf, upper=Inf, step=Inf,
              name=_knob_name(obj, Symbol(prop)))
  lower <= upper || error("Knob $name: lower must be <= upper")
  return Knob(obj, Symbol(prop), nothing, nothing, name, lower, upper, step)
end
function Knob(get::Function, set::Function; lower=-Inf, upper=Inf, step=Inf, name="knob")
  lower <= upper || error("Knob $name: lower must be <= upper")
  return Knob(nothing, nothing, get, set, name, lower, upper, step)
end
Knob(p::Pair; kwargs...) = Knob(p.first, p.second; kwargs...)
Knob(k::Knob) = k

_knob_name(obj, prop) = String(prop)
function _knob_name(obj::LatElement, prop)
  name = (obj.UniversalParams::UniversalParams).name
  return isempty(name) ? String(prop) : name * "." * String(prop)
end

_getknob(k::Knob) = isnothing(k.prop) ? k.get() : getproperty(k.obj, k.prop)
_setknob!(k::Knob, v) = isnothing(k.prop) ? k.set(v) : setproperty!(k.obj, k.prop, v)
_sameknob(a::Knob, b::Knob) = isnothing(a.prop) ? a.set === b.set :
                              (a.obj === b.obj && a.prop === b.prop)

"""
    @knob obj.property [lower=...] [upper=...] [step=...] [name=...]

The same as `Knob(obj, :property; lower=..., upper=..., step=..., name=...)`.
"""
macro knob(ex, kws...)
  (ex isa Expr && ex.head == :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode) ||
    error("@knob expects `object.property`")
  kwargs = map(kws) do kw
    (kw isa Expr && kw.head == :(=)) || error("@knob expects keyword arguments `name=value`")
    Expr(:kw, kw.args[1], esc(kw.args[2]))
  end
  return :(Knob($(esc(ex.args[1])), $(ex.args[2]); $(kwargs...)))
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
to continue the Twiss functions of another beamline.

## Quantities
- Summary values: `tw.q1`, `tw.q2`, `tw.q3`, `tw.etac`, `tw.alphac`, `tw.qspin`, `tw.damp1`, ...
  A term of an amplitude-dependent value is selected as for `getterm`, e.g. `tw.q1[delta=1]`;
  `tw.dq1` and `tw.dq2` are short for the chromaticities `tw.q1[delta=1]` and `tw.q2[delta=1]`.
- Columns: `tw.<column>[location]` for every scalar column of `twiss`, including chromatic
  derivatives (`tw.dbeta1`, `tw.dx_2`) and `tw.h<ijkl>`.

## Locations
- `tw.beta1[ele]`, `tw.beta1[7]`: the beginning of the element (or of the element with that
  beamline index). An element placed several times gives a vector.
- `tw.beta1[ele, :exit]`: the end of the element.
- `tw.beta1[ele, :]`: every integration step of the element, both ends included (a vector).
  `tw.beta1[ele, k]`: after `k` integration steps.
- `tw.beta1[end]`, `tw.beta1[begin]`: the end and the beginning of the beamline.
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
  # Elements whose rows are added when tunes or phases are used, so that consecutive rows are
  # less than half a turn of phase apart (twiss unwraps the phases row by row), and the number
  # of elements of the Beamline when they were chosen (-1: not yet)
  phase_rows::Vector{Int}
  phase_rows_n::Base.RefValue{Int}
  phase_gap::Base.RefValue{Float64} # largest phase advance between them when they were chosen
end

function TwissExpr(bl::Beamline; order::Integer=0, chrom::Integer=0, kwargs...)
  for k in (:at, :cols, :base_cols, :as_taylor_series, :GTPSA_descriptor)
    haskey(kwargs, k) && error("TwissExpr sets `$k` itself from the expressions used")
  end
  return TwissExpr(bl, Dict{Symbol,Any}(kwargs), max(order, chrom), Int[], Ref(-1), Ref(0.0))
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

struct TwissColumn
  tw::TwissExpr
  col::String
end

Base.propertynames(tw::TwissExpr) = [_TWISS_SUMMARY..., :dq1, :dq2,
  (Symbol(k) for k in keys(_TWISS_FCN_MAP) if !(k in ("index", "name", "kind")))...]

function Base.getproperty(tw::TwissExpr, key::Symbol)
  if key in _TWISS_SUMMARY
    return _TwissLeaf(tw, String(key), 0, 0, (0, 0, 0, 0))
  elseif key == :dq1
    return _TwissLeaf(tw, "q1", 0, 0, (1, 0, 0, 0))
  elseif key == :dq2
    return _TwissLeaf(tw, "q2", 0, 0, (1, 0, 0, 0))
  end
  col = String(key)
  _col_order(col) # errors for an unknown column
  return TwissColumn(tw, col)
end

Base.show(io::IO, tw::TwissExpr) = print(io, "TwissExpr of a Beamline with ",
  length(getfield(tw, :bl).line), " elements", isempty(getfield(tw, :kwargs)) ? "" :
  " (" * join(("$k=$(v isa LatticeExpr || v isa Number || v isa Bool ? v : typeof(v))"
               for (k, v) in getfield(tw, :kwargs)), ", ") * ")")
Base.show(io::IO, c::TwissColumn) = print(io, "TwissExpr column ", c.col,
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
    delta, J1, J2, J3 = l.mono
    return l.mono == (0, 0, 0, 0) ? 1 : delta + 2*(J1 + J2 + J3) + 1
  end
  return _col_order(l.col)
end

# Term selection of a summary value, as getterm
function Base.getindex(l::_TwissLeaf; delta::Integer=0, J1::Integer=0, J2::Integer=0,
                       J3::Integer=0)
  l.index == 0 || error("Only summary values (q1, q2, ...) have terms in delta, J1, J2, J3")
  return _TwissLeaf(l.tw, l.col, 0, 0, (delta, J1, J2, J3))
end

struct _End end
Base.lastindex(::TwissColumn) = _End()
Base.firstindex(::TwissColumn) = 1

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

_nsteps(ele) = (up = ele.UniversalParams::UniversalParams; up.n_slices*up.n_steps)

# Beamline indexes of the elements selected by `sel`
function _element_indexes(bl::Beamline, sel)
  line = bl.line
  if sel isa Integer
    1 <= sel <= length(line) || error("Beamline index $sel is not in 1:$(length(line))")
    return Int[sel]
  elseif sel isa LatElement
    idxs = findall(line) do ele
      ele === sel || (haskey(getfield(ele, :pdict), InheritParams) &&
                      getfield(ele, :pdict)[InheritParams].parent === sel)
    end
    isempty(idxs) && error("Element $(sel.name) is not in the Beamline of this TwissExpr")
    return idxs
  elseif sel isa Union{AbstractString,Regex}
    eles = findelements(bl, sel)
    isempty(eles) && error("No element of the Beamline matches $(repr(sel))")
    return Int[ele.beamline_index for ele in eles]
  elseif sel isa AbstractVector
    return reduce(vcat, (_element_indexes(bl, x) for x in sel); init=Int[])
  end
  error("Unsupported location $(repr(sel))")
end

# The location `step` integration steps after the beginning of element i; its end is the
# beginning of the next element (index -1: the end of the beamline)
function _location(bl, i, step)
  n = _nsteps(bl.line[i])
  0 <= step <= n || error("Element $i has $n integration steps: step $step does not exist")
  if step == n
    return i == length(bl.line) ? (-1, 0) : (i + 1, 0)
  end
  return (i, step)
end

_leaf(c::TwissColumn, loc) = _TwissLeaf(c.tw, c.col, loc[1], loc[2], (0, 0, 0, 0))
_one_or_vector(leaves) = length(leaves) == 1 ? only(leaves) : ExprVector(leaves)

Base.getindex(c::TwissColumn, ::_End) = _leaf(c, (-1, 0))
Base.getindex(c::TwissColumn, ::_End, pos) = _leaf(c, (-1, 0))
function Base.getindex(c::TwissColumn, sel::Union{Integer,LatElement})
  bl = getfield(c.tw, :bl)
  return _one_or_vector(LatticeExpr[_leaf(c, (i, 0)) for i in _element_indexes(bl, sel)])
end
function Base.getindex(c::TwissColumn, sel::Union{AbstractString,Regex,AbstractVector})
  bl = getfield(c.tw, :bl)
  return ExprVector(LatticeExpr[_leaf(c, (i, 0)) for i in _element_indexes(bl, sel)])
end
function Base.getindex(c::TwissColumn, sel, pos::Union{Symbol,AbstractString,Integer,Colon})
  bl = getfield(c.tw, :bl)
  idxs = _element_indexes(bl, sel)
  pos isa AbstractString && (pos = Symbol(pos))
  if pos isa Colon || pos == :all
    return ExprVector(LatticeExpr[_leaf(c, _location(bl, i, k)) for i in idxs
                                  for k in 0:_nsteps(bl.line[i])])
  elseif pos isa Integer
    leaves = LatticeExpr[_leaf(c, _location(bl, i, pos)) for i in idxs]
  elseif pos == :exit
    leaves = LatticeExpr[_leaf(c, _location(bl, i, _nsteps(bl.line[i]))) for i in idxs]
  elseif pos == :entrance
    leaves = LatticeExpr[_leaf(c, (i, 0)) for i in idxs]
  else
    error("Unknown position $(repr(pos)): use :entrance, :exit, : or a step number")
  end
  return sel isa Union{Integer,LatElement} ? _one_or_vector(leaves) : ExprVector(leaves)
end
Base.getindex(c::TwissColumn, ::Colon) = c[(0.0, Inf)]
function Base.getindex(c::TwissColumn, range::Tuple{Real,Real})
  bl = getfield(c.tw, :bl)
  range[1] <= range[2] || error("Invalid s range $range")
  s, L, n = _line_steps(bl)
  leaves = LatticeExpr[]
  for i in eachindex(s), k in 0:n[i]-1
    sk = s[i] + k*L[i]/n[i]
    range[1] <= sk < range[2] && push!(leaves, _leaf(c, (i, k)))
  end
  range[1] <= s[end] + L[end] <= range[2] && push!(leaves, _leaf(c, (-1, 0)))
  return ExprVector(leaves)
end

# s at the beginning, length and number of integration steps of every element
function _line_steps(bl::Beamline)
  line = bl.line
  s = Vector{Float64}(undef, length(line))
  L = Vector{Float64}(undef, length(line))
  n = Vector{Int}(undef, length(line))
  scur = 0.0
  for (i, ele) in enumerate(line)
    up = (ele.UniversalParams)::UniversalParams
    if up.L isa DefExpr
      up = Beamlines.deval(up, bl.context)
    end
    s[i] = scur
    L[i] = scalar(up.L)
    n[i] = up.n_slices*up.n_steps
    scur += L[i]
  end
  return s, L, n
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
