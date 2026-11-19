# Evaluation of LatticeExprs and of their derivatives in Knobs.
#
# The knobs are set to value + GTPSA parameter, every TwissExpr used is computed with one twiss
# call (only the columns and rows used), each quantity is reduced to a Taylor series in the knobs
# alone (a parameter-only Descriptor), and the expression trees are evaluated with these. The
# lattice is always restored.
#
# Truncation orders: for a quantity that needs phase space order v and derivatives of total order
# n in the knobs, all six variables get order v+n. The one-turn map then has every term that the
# parameter dependent closed orbit turns into a term of lower order in x (e.g. x^2), and twiss
# tracks only once (see `_check_cachable`).

struct _Group
  tw::TwissExpr
  leaves::Vector{_TwissLeaf}
end

struct _Collection
  groups::Vector{_Group} # in dependency order
  groupof::IdDict{TwissExpr,Int}
  getters::Vector{_Getter}
  seen::IdDict{Any,Nothing}
end
_Collection() = _Collection(_Group[], IdDict{TwissExpr,Int}(), _Getter[], IdDict{Any,Nothing}())

_collect!(c::_Collection, e) = nothing # numbers, knobs
function _collect!(c::_Collection, e::_Call)
  haskey(c.seen, e) && return
  c.seen[e] = nothing
  for a in e.args
    _collect!(c, a)
  end
end
function _collect!(c::_Collection, e::_Deriv)
  haskey(c.seen, e) && return
  c.seen[e] = nothing
  _collect!(c, e.arg)
end
function _collect!(c::_Collection, e::_Getter)
  haskey(c.seen, e) && return
  c.seen[e] = nothing
  push!(c.getters, e)
  return
end
function _collect!(c::_Collection, e::_TwissLeaf)
  haskey(c.seen, e) && return
  c.seen[e] = nothing
  if !haskey(c.groupof, e.tw)
    # The TwissExprs this one takes inputs from (e.g. a_initial) come first
    for v in values(getfield(e.tw, :kwargs))
      v isa LatticeExpr && _collect!(c, v)
    end
    push!(c.groups, _Group(e.tw, _TwissLeaf[]))
    c.groupof[e.tw] = length(c.groups)
  end
  push!(c.groups[c.groupof[e.tw]].leaves, e)
  return
end

# Orders of derivative (deriv) in each knob that an expression takes, and their largest total
function _deriv_orders(e, memo)
  if e isa _Deriv
    haskey(memo, e) && return memo[e]
    orders, total = _deriv_orders(e.arg, memo)
    orders = copy(orders)
    k = findfirst(p -> _sameknob(p.first, e.knob), orders)
    if isnothing(k)
      push!(orders, e.knob => 1)
    else
      orders[k] = orders[k].first => orders[k].second + 1
    end
    return memo[e] = (orders, total + 1)
  elseif e isa _Call
    haskey(memo, e) && return memo[e]
    orders = Pair{Knob,Int}[]
    total = 0
    for a in e.args
      oa, ta = _deriv_orders(a, memo)
      total = max(total, ta)
      for p in oa
        k = findfirst(q -> _sameknob(q.first, p.first), orders)
        if isnothing(k)
          push!(orders, p)
        else
          orders[k] = p.first => max(orders[k].second, p.second)
        end
      end
    end
    return memo[e] = (orders, total)
  elseif e isa _TwissLeaf
    # Derivatives taken in the inputs of its TwissExpr (e.g. a_initial)
    haskey(memo, e.tw) && return memo[e.tw]
    orders = Pair{Knob,Int}[]
    total = 0
    for v in values(getfield(e.tw, :kwargs))
      v isa LatticeExpr || continue
      oa, ta = _deriv_orders(v, memo)
      total = max(total, ta)
      append!(orders, oa)
    end
    return memo[e.tw] = (orders, total)
  end
  return (Pair{Knob,Int}[], 0)
end

mutable struct _EvalState
  seeds::Vector{Knob}
  x0::Vector{Float64}
  pvars::Any # the knobs' variables in the parameter-only Descriptor (nothing: no derivatives)
  vals::IdDict{Any,Any}  # values of leaves and (memoized) nodes as Taylor series in the knobs
  maps::IdDict{Any,Any}  # map-valued leaves (e.g. a), in the twiss Descriptor
end

function _knob_value(k::Knob)
  x = _getknob(k)
  if x isa DefExpr
    error("Knob $(k.name) is a deferred expression. Vary what it depends on instead " *
          "(e.g. a variable of the Beamline's context).")
  elseif x isa AbstractVector
    error("Knob $(k.name) is a vector: give one entry (e.g. `@knob obj.property[1]`), or vary " *
          "it in match!/value_and_jacobian, where it is one knob per entry.")
  elseif !(x isa Union{AbstractFloat,Integer,Rational})
    error("Knob $(k.name) holds a $(typeof(x)), not a plain number. If it is left over from " *
          "a parametric calculation, use `scalarize!` first.")
  end
  return Float64(x)
end

_seed_index(st::_EvalState, k::Knob) = findfirst(s -> _sameknob(s, k), st.seeds)

_scalarize_knob!(obj) = nothing
_scalarize_knob!(obj::LatElement) = (scalarize!(obj); nothing)
function _scalarize_knob!(obj::Controller)
  for (target, _) in keys(obj.fcns)
    _scalarize_knob!(target)
  end
  return
end

# The part of the Taylor series t with the phase space monomial varmono, as a Taylor series in the
# parameters alone (in the Descriptor of pzero), or a number if there are no parameters
function _to_p(t::TPS, varmono, pzero)
  nn = TI.ndiffs(t)
  ords = zeros(UInt8, nn)
  T = TI.numtype(typeof(t))
  v = Ref{T}()
  out = isnothing(pzero) ? zero(T) : (T <: Complex ? complex(pzero) : zero(pzero))
  idx = TI.cycle!(t, -1, mono=ords, val=v) # -1 to include the constant term
  while idx > -1
    if all(k -> ords[k] == varmono[k], 1:6)
      if isnothing(pzero)
        out = v[]
      else
        TI.setm!(out, v[], ords[7:nn])
      end
    end
    idx = TI.cycle!(t, idx, mono=ords, val=v)
  end
  return out
end
function _to_p(x::Number, varmono, pzero)
  all(==(0), varmono) || error("The truncation order is too low for this term")
  return x isa Complex ? ComplexF64(x) : Float64(x)
end
function _to_p(a::AmplitudeDependentValue, mono4, pzero)
  delta, J1, J2, J3 = mono4
  if a.coast && J3 != 0
    error("`J3` is not an amplitude of this value (the beam is coasting), use `delta`")
  elseif !a.coast && delta != 0
    error("`delta` is not an amplitude of this value (the beam is not coasting; turn the RF " *
          "off with `rf_on=false`), or use `J3`")
  end
  varmono = a.coast ? (J1, J1, J2, J2, 0, delta) : (J1, J1, J2, J2, J3, J3)
  # (a derivative: the coefficient times the factorials)
  return factorial(delta)*factorial(J1)*factorial(J2)*factorial(J3) *
         _to_p(a.taylor_series, varmono, pzero)
end

# Copy of the Taylor series t in the Descriptor of zer (terms it does not have are dropped)
function _redescribe(t::TPS, zer)
  out = zero(zer)
  nn_old = TI.ndiffs(t)
  nn = TI.ndiffs(out)
  ords = zeros(UInt8, nn_old)
  new = zeros(UInt8, nn)
  v = Ref{TI.numtype(typeof(t))}()
  desc = GTPSA.getdesc(out).desc
  idx = TI.cycle!(t, -1, mono=ords, val=v)
  while idx > -1
    new .= 0
    n = min(nn_old, nn)
    new[1:n] .= view(ords, 1:n)
    if all(==(0), view(ords, n+1:nn_old)) && GTPSA.mad_desc_isvalidm(desc, Cint(nn), new)
      TI.setm!(out, v[], new)
    end
    idx = TI.cycle!(t, idx, mono=ords, val=v)
  end
  return out
end
function _redescribe(m::DAMap, zer)
  GTPSA.getdesc(first(m.v)) == GTPSA.getdesc(zer) && return m
  nv = NNF.nvars(m)
  v = [_redescribe(t, zer) for t in m.v]
  q = isnothing(m.q) ? nothing : [_redescribe(m.q.q0, zer), _redescribe(m.q.q1, zer),
                                  _redescribe(m.q.q2, zer), _redescribe(m.q.q3, zer)]
  return DAMap(v0=m.v0, v=v, q=q, s=m.s, nv=nv, np=TI.ndiffs(zer) - nv)
end

function _kwarg_value(v, st, dzero)
  if v isa _TwissLeaf && v.col in _MAP_COLS
    return st.maps[v]
  elseif v isa _Getter
    return v.f()
  elseif v isa LatticeExpr
    error("As a keyword argument of a TwissExpr, an expression must be a map of another " *
          "TwissExpr (e.g. `tw.a[end]`) or `lazy(f)`")
  elseif v isa DAMap
    return _redescribe(v, dzero)
  end
  return v
end

# GTPSA ends the Julia process (it does not throw) when a function such as 1/sqrt gets a NaN. A
# periodic twiss throws before that (no closed orbit, or an unstable one-turn map), but an open
# line has no such check: knobs that make it extremely defocusing let the map overflow. So the
# initial orbit and the transfer matrix around it (by differences) are tracked as numbers first
function _check_open_line(bl, a::DAMap, kw)
  v0 = zeros(1, 6)
  if haskey(kw, :v0)
    v0 .= kw[:v0]
  else
    v0[1, 1:length(a.v0)] .= a.v0
    length(a.v0) == 5 && (v0[1, 6] = get(kw, :delta0, 0.0))
  end
  _check_open_line(bl, v0, kw)
end
function _check_open_line(bl, v0::Matrix, kw)
  v = repeat(v0, 13)
  for j in 1:6
    v[1+j, j] += 1e-6
    v[7+j, j] -= 1e-6
  end
  res = track(bl; v0=v, scalar_params=true, rf_on=get(kw, :rf_on, true))
  out = view(res.data.v, :, :, 1, size(res.data.v, 4))
  alive = view(res.data.state, :, 1, size(res.data.state, 3))
  if !all(isfinite, out) || any(!=(BeamTracking.STATE_ALIVE), alive)
    error("The initial orbit of the open line is lost or not finite at its end")
  end
  maximum(abs, view(out, 2:7, :) .- view(out, 8:13, :))/2e-6 < 1e8 ||
    error("The transfer matrix of the open line has elements above 1e8 (extremely defocusing)")
  return
end

# One twiss call for all quantities used of a TwissExpr
function _run_group!(st::_EvalState, g::_Group, D, dzero, pzero)
  tw = g.tw
  bl = getfield(tw, :bl)
  cols = unique!(String[l.col for l in g.leaves if l.index != 0 && l.col != "s"])
  needmaps = any(l -> l.index != 0 && l.col in _MAP_COLS, g.leaves)
  for l in g.leaves
    if l.step > 0
      l.step < _nsteps(bl, l.index) || error("$(l): the number of integration steps of the " *
                                             "element changed since the expression was written")
    end
  end
  # (the rows: exactly the locations of the expressions)
  at = _Locations([(l.index, l.step) for l in g.leaves if l.index != 0])
  kw = Dict{Symbol,Any}(k => _kwarg_value(v, st, dzero) for (k, v) in getfield(tw, :kwargs))
  get(kw, :a_initial, nothing) isa DAMap && _check_open_line(bl, kw[:a_initial], kw)
  if !isnothing(get(kw, :initial, nothing)) # (the orbit of initial, its values as numbers)
    _check_open_line(bl, first(_initial_v0(kw[:initial], get(kw, :delta0, nothing))), kw)
  end
  t = twiss(bl; at=at, cols=cols, base_cols=["index", "s"], GTPSA_descriptor=D,
            as_taylor_series=(!isnothing(pzero) || needmaps), kw...)
  df = t.df
  for l in g.leaves
    if l.index == 0
      haskey(t.summ.sdict, Symbol(l.col)) || error("twiss did not compute `$(l.col)` here " *
        "(e.g. `q3` needs RF, `qspin` needs `spin=true`)")
      st.vals[l] = _to_p(t.summ.sdict[Symbol(l.col)], l.mono, pzero)
      continue
    end
    x = df[_twiss_row(t, (l.index, l.step)), l.col]
    if l.col in _MAP_COLS
      st.maps[l] = x
    else
      st.vals[l] = _to_p(x, (0, 0, 0, 0, 0, 0), pzero)
    end
  end
  return
end

_peval(e::Real, st) = e
function _peval(k::Knob, st)
  i = _seed_index(st, k)
  return isnothing(i) || isnothing(st.pvars) ? _knob_value(k) : st.x0[i] + st.pvars[i]
end
_peval(e::Union{_TwissLeaf,_Getter}, st) = haskey(st.vals, e) ? st.vals[e] :
  error("$(e) is a map: it can only be used as an input of another TwissExpr")
function _peval(e::_Call, st)
  haskey(st.vals, e) && return st.vals[e]
  if e.f === _esum || e.f === _emax || e.f === _emin # (no splatting of long argument lists)
    op = e.f === _esum ? (+) : e.f
    r = _peval(e.args[1], st)
    for k in 2:length(e.args)
      r = op(r, _peval(e.args[k], st))
    end
  else
    r = e.f((_peval(a, st) for a in e.args)...)
  end
  return st.vals[e] = r
end
function _peval(e::_Deriv, st)
  haskey(st.vals, e) && return st.vals[e]
  x = _peval(e.arg, st)
  return st.vals[e] = x isa TPS ? GTPSA.deriv(x, _seed_index(st, e.knob)) : 0.0
end

_flatten_exprs(e::Union{LatticeExpr,Real}) = Any[e]
_flatten_exprs(e::AbstractArray) = Any[x for x in vec(e)] # (a matrix too, e.g. of derivatives)
_knobs(k::Union{Knob,Pair}) = _knobs((k,))
function _knobs(ks)
  knobs = Knob[]
  for k in Iterators.map(Knob, ks)
    # (a knob on a vector of numbers is one knob per entry)
    x = isnothing(k.prop) || k.index != 0 ? nothing : _getknob(k)
    if x isa AbstractVector
      for i in eachindex(x)
        push!(knobs, Knob(k.obj, k.prop, i, nothing, nothing, string(k.name, "[", i, "]"),
                          k.lower, k.upper, k.step))
      end
    else
      push!(knobs, k)
    end
  end
  for i in eachindex(knobs), j in 1:i-1
    _sameknob(knobs[i], knobs[j]) && error("Knob $(knobs[i].name) is given twice")
  end
  return knobs
end

# Values of the expressions and, if jac, their derivatives in the knobs vary
function _evaluate(exprs::Vector{Any}, vary::Vector{Knob}, jac::Bool; complex_ok::Bool=false)
  for e in exprs
    e isa Union{LatticeExpr,Real} || error("Expected expressions (LatticeExpr), got a " *
      "$(typeof(e)). (A vector of expressions compared with `==` to a plain vector of " *
      "expressions is a Bool: build targets from an ExprVector, or with `.==`.)")
  end
  c = _Collection()
  memo = IdDict{Any,Any}()
  dorders = Pair{Knob,Int}[]
  nd = 0
  for e in exprs
    _collect!(c, e)
    oe, te = _deriv_orders(e, memo)
    nd = max(nd, te)
    for p in oe
      k = findfirst(q -> _sameknob(q.first, p.first), dorders)
      if isnothing(k)
        push!(dorders, p)
      else
        dorders[k] = p.first => max(dorders[k].second, p.second)
      end
    end
  end
  # The knobs that carry derivatives: those varied, then those of deriv
  seeds = jac ? copy(vary) : Knob[]
  orders = fill(1, length(seeds))
  for p in dorders
    k = findfirst(s -> _sameknob(s, p.first), seeds)
    if isnothing(k)
      push!(seeds, p.first)
      push!(orders, p.second)
    else
      orders[k] += p.second
    end
  end
  n = isempty(seeds) ? 0 : (jac && !isempty(vary) ? 1 : 0) + nd
  V = n + maximum(max(getfield(g.tw, :order), maximum(_leaf_order, g.leaves)) for g in c.groups;
                  init=1)
  x0 = Float64[_knob_value(k) for k in seeds]
  st = _EvalState(seeds, x0, nothing, IdDict{Any,Any}(), IdDict{Any,Any}())
  desc_old = GTPSA.desc_current
  pzero = nothing
  try
    if n > 0
      st.pvars = vars(Descriptor(orders, n))
      pzero = zero(st.pvars[1])
      D = Descriptor(fill(V, 6), V, orders, n)
    else
      D = Descriptor(6, V)
    end
    GTPSA.desc_current = D
    dk = params(D)
    dzero = zero(first(vars(D)))
    # Vectors with varied entries hold the Taylor series in copies, and get the original back
    saved = Tuple{Any,Symbol,Any}[]
    try
      for (i, k) in enumerate(seeds)
        if k.index != 0 && !any(s -> s[1] === k.obj && s[2] === k.prop, saved)
          a = getproperty(k.obj, k.prop)
          push!(saved, (k.obj, k.prop, a))
          setproperty!(k.obj, k.prop, copyto!(similar(a, typeof(x0[i] + dk[i])), a))
        end
      end
      for (i, k) in enumerate(seeds)
        _setknob!(k, x0[i] + dk[i])
      end
      for g in c.groups
        _run_group!(st, g, D, dzero, pzero)
      end
      for g in c.getters
        st.vals[g] = _to_p(g.f(), (0, 0, 0, 0, 0, 0), pzero)
      end
    finally
      # (with D still current: a number set next to a TPS is converted with the current one)
      for (i, k) in enumerate(seeds)
        k.index == 0 && _setknob!(k, x0[i])
        _scalarize_knob!(k.obj)
      end
      for (obj, prop, a) in saved
        setproperty!(obj, prop, a)
      end
      # A function knob may have set parameters of any element (which then hold Taylor series)
      if any(k -> isnothing(k.prop), seeds)
        for g in c.groups, ele in getfield(g.tw, :bl).line
          scalarize!(ele)
        end
      end
    end
  finally
    GTPSA.desc_current = desc_old
  end
  # Complex values (e.g. h terms) are values (complex_ok), but not residuals of a match
  rs = Any[_peval(e, st) for e in exprs]
  i = findfirst(r -> !(scalar(r) isa Real), rs)
  isnothing(i) || complex_ok || error("The expression $(exprs[i]) is complex (e.g. an h term): " *
                                      "a target or `minimize` needs `abs`, `real` or `imag` of it")
  T = isnothing(i) ? Float64 : ComplexF64
  vals = Vector{T}(undef, length(exprs))
  J = jac ? zeros(T, length(exprs), length(vary)) : nothing
  for (i, r) in enumerate(rs)
    vals[i] = scalar(r)
    if jac && r isa TPS
      J[i, :] .= view(GTPSA.gradient(r), 1:length(vary))
    end
  end
  return vals, J
end

"""
    value(expr)
    value(exprs::AbstractArray)

The current value of the expression ([`LatticeExpr`](@ref)) `expr`, or of each expression of an
array (complex for an `h` term). Each [`TwissExpr`](@ref) used is computed with one `twiss`
call, only at the locations and with the columns the expressions use.
"""
value(e::Union{LatticeExpr,Real}) = _evaluate(Any[e], Knob[], false; complex_ok=true)[1][1]
value(e::AbstractArray) =
  reshape(_evaluate(_flatten_exprs(e), Knob[], false; complex_ok=true)[1], size(e))

"""
    value_and_jacobian(exprs, knobs) -> (values, jacobian)
    value_and_jacobian(exprs, knobs, x) -> (values, jacobian)

The current values of the expressions ([`LatticeExpr`](@ref)s) `exprs` and their exact derivatives
in the `knobs` ([`Knob`](@ref)s, or `obj => :property` pairs): `jacobian[i, j]` is the derivative of
`exprs[i]` (in the order of `vec(exprs)`) in `knobs[j]`. For a single expression, the value and its
gradient are returned. With `x`, the knobs are first set to the values `x` (and stay there).

The derivatives are computed by automatic differentiation: the knobs are made parameters of the
truncated power series (GTPSA) of one `twiss` call per [`TwissExpr`](@ref), with the truncation
orders the expressions need, and the lattice is restored afterwards.
"""
function value_and_jacobian(e::Union{LatticeExpr,Real}, knobs)
  vals, J = _evaluate(Any[e], _knobs(knobs), true; complex_ok=true)
  return vals[1], J[1, :]
end
function value_and_jacobian(e::AbstractArray, knobs)
  vals, J = _evaluate(_flatten_exprs(e), _knobs(knobs), true; complex_ok=true)
  return reshape(vals, size(e)), J
end
function value_and_jacobian(e, knobs, x)
  setknobs!(knobs, x)
  return value_and_jacobian(e, knobs)
end

"""
    setknobs!(knobs, x)

Sets the `knobs` to the values `x`.
"""
function setknobs!(knobs, x)
  ks = _knobs(knobs)
  length(ks) == length(x) || error("$(length(ks)) knobs but $(length(x)) values")
  for (k, xk) in zip(ks, x)
    _setknob!(k, xk)
  end
  return knobs
end
