"""
    Target(expr, value; tol=1e-8, weight=1, name)
    Target(expr; lower=-Inf, upper=Inf, tol=1e-8, weight=1, name)
    expr == value,  expr <= value,  expr >= value

A goal for [`match!`](@ref): the expression ([`LatticeExpr`](@ref)) `expr` should equal `value`,
or lie between `lower` and `upper`. The target is met when it is within `tol`. `weight` multiplies
its residual, to make it count more or less than the other targets while they are not all met.
"""
struct Target
  expr::LatticeExpr
  lower::Float64
  upper::Float64
  tol::Float64
  weight::Float64
  name::String
end
function Target(expr::LatticeExpr, value::Real; tol=1e-8, weight=1, name=repr(expr))
  return Target(expr, value, value, tol, weight, name)
end
function Target(expr::LatticeExpr; lower=-Inf, upper=Inf, tol=1e-8, weight=1, name=repr(expr))
  lower <= upper || error("Target $name: lower must be <= upper")
  return Target(expr, lower, upper, tol, weight, name)
end

Base.broadcastable(t::Target) = Ref(t)
Base.:!(::Target) = error("A Target is not a Bool. To compare a plain vector of expressions " *
                          "elementwise, use `.==`, `.<=` or `.>=`.")
Base.:(==)(a::LatticeExpr, b::Real) = Target(a, b)
Base.:(==)(a::Real, b::LatticeExpr) = Target(b, a)
Base.:(==)(a::LatticeExpr, b::LatticeExpr) = Target(a - b, 0.0)
Base.:(<=)(a::LatticeExpr, b::Real) = Target(a; upper=b)
Base.:(>=)(a::LatticeExpr, b::Real) = Target(a; lower=b)
Base.:(<=)(a::Real, b::LatticeExpr) = Target(b; lower=a)
Base.:(>=)(a::Real, b::LatticeExpr) = Target(b; upper=a)
Base.:(<=)(a::LatticeExpr, b::LatticeExpr) = Target(a - b; upper=0.0)
Base.:(>=)(a::LatticeExpr, b::LatticeExpr) = Target(a - b; lower=0.0)
for op in (:(==), :(<=), :(>=))
  @eval begin
    Base.$op(a::ExprVector, b::Real) = Target[$op(x, b) for x in a.v]
    Base.$op(a::ExprVector, b::LatticeExpr) = Target[$op(x, b) for x in a.v]
    Base.$op(a::Real, b::ExprVector) = Target[$op(a, x) for x in b.v]
    Base.$op(a::LatticeExpr, b::ExprVector) = Target[$op(a, x) for x in b.v]
    Base.$op(a::ExprVector, b::ExprVector) = _elementwise_targets($op, a, b)
    Base.$op(a::ExprVector, b::AbstractVector{<:Real}) = _elementwise_targets($op, a, b)
    Base.$op(a::AbstractVector{<:Real}, b::ExprVector) = _elementwise_targets($op, a, b)
  end
end
function _elementwise_targets(op, a, b)
  length(a) == length(b) || error("Vectors must have the same length " *
                                  "($(length(a)) and $(length(b)))")
  return Target[op(x, y) for (x, y) in zip(a, b)]
end

function _goal(t::Target)
  t.lower == t.upper && return "== " * repr(t.lower)
  isinf(t.lower) && return "<= " * repr(t.upper)
  isinf(t.upper) && return ">= " * repr(t.lower)
  return "in [" * repr(t.lower) * ", " * repr(t.upper) * "]"
end
Base.show(io::IO, t::Target) = print(io, "Target: ", t.name, " ", _goal(t))

value(t::Target) = value(t.expr)

_flatten_targets!(out, t::Target) = push!(out, t)
function _flatten_targets!(out, t)
  t isa Union{AbstractArray,Tuple} || error("Expected targets (e.g. `tw.q1 == 0.25`), got a " *
    "$(typeof(t)). (Comparing a plain vector of expressions with `==` gives a Bool: use `.==`.)")
  for x in t
    _flatten_targets!(out, x)
  end
  return out
end

"""
    MatchProblem(; vary, targets=[], minimize=[], minimize_weight=1)

A matching/optimization problem: the [`Knob`](@ref)s `vary` (or `obj => :property` pairs), the
[`Target`](@ref)s to meet, and expressions `minimize` whose sum of squares should be as small as
possible. [`match!`](@ref) solves it with its built-in solver. To use any other optimizer:

- `prob.x`, `prob.lower`, `prob.upper`: the current values and the bounds of the knobs
- `value(prob, x)`: the residual vector at the knob values `x`: for each target its weight times
  its distance from its goal (zero where it is met), then `minimize_weight` times each
  expression to minimize
- `SciBmad.value_and_jacobian(prob, x)`: the residuals and their exact Jacobian in the knobs
- `penalty(prob, x)`: the sum of the squared residuals, `Inf` where the lattice has no solution
  (for optimizers that take a scalar and no derivatives, e.g. genetic algorithms)
- `setknobs!(prob, x)`: sets the knobs

These functions leave the knobs at `x`.

The sum of the squared residuals is what is minimized, so targets and expressions to minimize
are balanced against each other when they cannot all vanish. Larger target weights (or a smaller
`minimize_weight`) hold the targets more tightly.
"""
struct MatchProblem
  knobs::Vector{Knob}
  targets::Vector{Target}
  minimize::Vector{Any}
  minimize_weight::Float64
  exprs::Vector{Any} # of the targets, then minimize
end
function MatchProblem(; vary, targets=Target[], minimize=LatticeExpr[], minimize_weight=1)
  ts = _flatten_targets!(Target[], targets)
  mins = _flatten_exprs(minimize)
  isempty(ts) && isempty(mins) && error("Nothing to match: give `targets` and/or `minimize`")
  return MatchProblem(_knobs(vary), ts, mins, minimize_weight,
                      Any[(t.expr for t in ts)..., mins...])
end

function Base.getproperty(prob::MatchProblem, key::Symbol)
  if key == :x
    return Float64[_knob_value(k) for k in getfield(prob, :knobs)]
  elseif key == :lower
    return Float64[k.lower for k in getfield(prob, :knobs)]
  elseif key == :upper
    return Float64[k.upper for k in getfield(prob, :knobs)]
  end
  return getfield(prob, key)
end
Base.propertynames(::MatchProblem) = (:x, :lower, :upper, :knobs, :targets, :minimize,
                                      :minimize_weight)

Base.show(io::IO, prob::MatchProblem) = print(io, "MatchProblem with ", length(prob.knobs),
  " knobs, ", length(prob.targets), " targets", isempty(prob.minimize) ? "" :
  " and $(length(prob.minimize)) expressions to minimize")

setknobs!(prob::MatchProblem, x) = (setknobs!(prob.knobs, x); prob)

# Residuals from the values of the expressions: the weighted distance of each target from its
# goal (zero inside an interval), then the expressions to minimize. A row of the Jacobian of a
# target that is met inside its interval is zero
function _residuals!(prob::MatchProblem, vals, J)
  for (i, t) in enumerate(prob.targets)
    d = vals[i] - clamp(vals[i], t.lower, t.upper)
    vals[i] = t.weight*d
    if !isnothing(J)
      J[i, :] .*= (d == 0 && t.lower != t.upper) ? 0.0 : t.weight
    end
  end
  for i in length(prob.targets)+1:length(vals)
    vals[i] *= prob.minimize_weight
    isnothing(J) || (J[i, :] .*= prob.minimize_weight)
  end
  return vals, J
end

function value(prob::MatchProblem, x)
  setknobs!(prob, x)
  return _residuals!(prob, _evaluate(prob.exprs, prob.knobs, false)[1], nothing)[1]
end
function value_and_jacobian(prob::MatchProblem, x)
  setknobs!(prob, x)
  return _residuals!(prob, _evaluate(prob.exprs, prob.knobs, true)...)
end

"""
    penalty(prob::MatchProblem, x)

The sum of the squared residuals of `prob` at the knob values `x`, or `Inf` if the lattice has no
solution there (e.g. it is unstable).
"""
function penalty(prob::MatchProblem, x)
  try
    return sum(abs2, value(prob, x))
  catch err
    err isa InterruptException && rethrow()
    return Inf
  end
end

"""
    MatchResult

The result of [`match!`](@ref): `converged` (all targets met), `status`, `iterations`, the knob
values before and after (`x0`, `x`), the values of the target expressions (`values`), the sum of
squared residuals after each iteration (`history`), and the `problem`. `undo!(res)` sets the knobs
back to `x0`; `match!(res; kwargs...)` continues.
"""
struct MatchResult
  problem::MatchProblem
  converged::Bool
  status::String
  iterations::Int
  evaluations::Int
  elapsed::Float64
  x0::Vector{Float64}
  x::Vector{Float64}
  values::Vector{Float64}
  history::Vector{Float64}
end

"""
    undo!(res::MatchResult)

Sets the knobs back to their values before the `match!` that returned `res`.
"""
undo!(res::MatchResult) = (setknobs!(res.problem, res.x0); res)

_num(x) = string(round(x; sigdigits=10))
# print shows every row; the display in the REPL (limited) fits the tables into the terminal
Base.show(io::IO, res::MatchResult) = _show_result(io, res, typemax(Int))
function Base.show(io::IO, ::MIME"text/plain", res::MatchResult)
  get(io, :limit, false) || return _show_result(io, res, typemax(Int))
  prob = res.problem
  ntables = (isempty(prob.targets) ? 0 : 1) + (isempty(prob.minimize) ? 0 : 1) + 1
  # (the terminal less the first line, a header and a "more" line per table, and the prompt)
  maxrows = max(3, (displaysize(io)[1] - 4 - 2*ntables) ÷ ntables)
  return _show_result(io, res, maxrows)
end
function _show_result(io::IO, res::MatchResult, maxrows)
  prob = res.problem
  println(io, res.converged ? "Match converged" : "Match NOT converged", " (", res.status, "): ",
          res.iterations, " iterations, ", res.evaluations, " evaluations, ",
          round(res.elapsed; sigdigits=3), " s")
  nt = length(prob.targets)
  if nt > 0
    rows = [(t.name, _num(res.values[i]), _goal(t),
             abs(res.values[i] - clamp(res.values[i], t.lower, t.upper)) <= t.tol ? "yes" : "NO")
            for (i, t) in enumerate(prob.targets)]
    _print_table(io, ("Target", "value", "goal", "met"), rows, maxrows)
  end
  if !isempty(prob.minimize)
    rows = [(repr(e), _num(res.values[nt+i])) for (i, e) in enumerate(prob.minimize)]
    _print_table(io, ("Minimized", "value"), rows, maxrows)
  end
  rows = [(k.name, _num(res.x0[i]), _num(res.x[i]), _num(k.lower), _num(k.upper))
          for (i, k) in enumerate(prob.knobs)]
  _print_table(io, ("Knob", "before", "after", "lower", "upper"), rows, maxrows)
  return
end
function _print_table(io, header, rows, maxrows)
  shown = length(rows) > maxrows ? rows[1:maxrows] : rows
  w = [maximum(length, (header[j], (r[j] for r in shown)...)) for j in eachindex(header)]
  println(io, " ", rstrip(join((rpad(header[j], w[j]) for j in eachindex(header)), "  ")))
  for r in shown
    println(io, " ", rstrip(join((rpad(r[j], w[j]) for j in eachindex(header)), "  ")))
  end
  length(rows) > maxrows && println(io, " … ", length(rows) - maxrows, " more")
  return
end

"""
    match!(; vary, targets=[], minimize=[], kwargs...) -> MatchResult
    match!(bl::Beamline; vary, targets, twiss=(;), kwargs...)
    match!(prob::MatchProblem; kwargs...)
    match!(res::MatchResult; kwargs...)

Varies the [`Knob`](@ref)s `vary` (also `obj => :property` pairs) until the [`Target`](@ref)s
`targets` are met, and leaves the lattice there. `minimize` are expressions whose sum of squares is
made as small as possible together with the distances from the targets (each times its `weight`;
the expressions times `minimize_weight`, default `1`). Targets are written with expressions
([`LatticeExpr`](@ref)) of the quantities of a [`TwissExpr`](@ref):

```julia
tw = TwissExpr(cell)
match!(vary=[@knob(qf.Kn1), @knob(qd.Kn1)], targets=[tw.q1 == 0.25, tw.q2 == 0.25])
```

Given a `Beamline`, `targets` (and `minimize`) may be functions of its `TwissExpr` (made with the
`twiss` keyword arguments in `twiss`), which can also be written as a `do` block:

```julia
match!(cell; vary=[@knob(qf.Kn1), @knob(qd.Kn1)], targets = tw -> [tw.q1 == 0.25, tw.q2 == 0.25])
```

The derivatives are exact (see [`SciBmad.value_and_jacobian`](@ref)). Each step is a
Gauss-Newton step from the singular value decomposition of the Jacobian: the smallest change of
the knobs that meets the targets to first order. Steps that do not reduce the residuals, or where
the lattice has no solution (e.g. it is unstable), are rejected and shorter or damped ones taken,
so there may be fewer, as many, or more targets than knobs.
Knobs stay within their bounds and step limits, and `<=` and `>=` targets only count while they
are violated. Like every such method it finds a nearby solution: targets of very different size
(a tune and a beta function) should get `weight`s that make their residuals comparable. To use
another optimizer, see [`MatchProblem`](@ref).

## Keyword arguments
- `maxiter`: Maximum number of iterations, default is `50`
- `verbose`: If `true`, prints the progress of the solver at each iteration. Default is `false`
- `solver`: `:gauss_newton` (default): Gauss-Newton steps from the singular value
    decomposition of the Jacobian (their halves, quarters and eighths, then damped steps, where a
    step does not reduce the residuals), or `:levenberg_marquardt`: the Levenberg-Marquardt
    method of `BatchSolve` (`BatchSolve.levenberg_marquardt!`)
- `rcond`: For `solver=:gauss_newton`, singular values of the Jacobian below `rcond` times the
    largest are ignored (knob combinations the targets do not respond to). Default is `1e-10`

Returns a [`MatchResult`](@ref). `match!(res)` continues the match from the current lattice.
"""
function match!(prob::MatchProblem; maxiter::Integer=50, verbose::Bool=false,
                solver::Symbol=:gauss_newton, rcond::Real=1e-10)
  if solver == :levenberg_marquardt
    return _match_levenberg_marquardt!(prob, maxiter, verbose)
  elseif solver != :gauss_newton
    error("Unknown solver $(repr(solver)): use :levenberg_marquardt or :gauss_newton")
  end
  t_start = time()
  knobs = prob.knobs
  lower, upper = prob.lower, prob.upper
  maxstep = Float64[k.step for k in knobs]
  nt = length(prob.targets)
  x0 = prob.x
  all(lower .<= x0 .<= upper) || error("A knob starts outside of its bounds")
  x = copy(x0)
  vals, J = try
    _evaluate(prob.exprs, knobs, true)
  catch err
    err isa InterruptException && rethrow()
    error("match!: the targets cannot be computed at the starting values of the knobs (",
          sprint(showerror, err), "). Start from a lattice that has a solution.")
  end
  evaluations = 1
  met(v) = all(abs(v[i] - clamp(v[i], t.lower, t.upper)) <= t.tol
               for (i, t) in enumerate(prob.targets))
  raw = copy(vals)
  r, J = _residuals!(prob, vals, J)
  f = sum(abs2, r)
  history = Float64[f]
  status = "maximum number of iterations"
  iterations = 0
  lambda = -1.0 # Levenberg-Marquardt damping (set from the first Jacobian) and its growth factor
  nu = 2.0
  gauss_newton = true
  verbose && println("match!: iteration 0, sum of squared residuals ", f)
  while true
    if nt > 0 && met(raw) && isempty(prob.minimize)
      status = "all targets met"
      break
    elseif iterations == maxiter
      break
    end
    # Knobs at a bound that the step would push outside stay there
    g = J'*r
    free = [k for k in eachindex(x) if !((x[k] <= lower[k] && g[k] > 0) ||
                                         (x[k] >= upper[k] && g[k] < 0))]
    stalled = nt > 0 && met(raw) ? "all targets met" : "no further improvement possible"
    if isempty(free) || norm(view(g, free)) <= 1e-13*(norm(J)*norm(r) + floatmin())
      status = stalled
      break
    end
    F = svd(J[:, free])
    c = F.U'*r
    lambda < 0 && (lambda = 1e-6*F.S[1]^2)
    accepted = false
    xt, rt, Jt, rawt, ft = x, r, J, raw, f
    # Trials (kind 0: the Gauss-Newton step; < 0: that fraction of it; NaN: a Levenberg-Marquardt
    # step, shorter and turned towards steepest descent as the damping grows). The Gauss-Newton
    # step and its first fractions come first while they keep being accepted, else last
    kinds = gauss_newton ? [0.0; -0.5; -0.25; -0.125; fill(NaN, 40)] :
                           [fill(NaN, 40); -(0.5 .^ (0:12))]
    for (trial, kind) in enumerate(kinds)
      isinf(kind) && continue
      damping = kind == 0 || kind < 0 ? 0.0 : lambda
      dx = zeros(length(x))
      for k in eachindex(F.S)
        if F.S[k] > rcond*F.S[1] && F.S[k] > 0
          dx[free] .-= (c[k]*F.S[k]/(F.S[k]^2 + damping)) .* view(F.Vt, k, :)
        end
      end
      scale = min(1.0, minimum(maxstep[k]/abs(dx[k]) for k in eachindex(dx) if dx[k] != 0;
                               init=1.0))
      kind < 0 && (scale *= -kind)
      xt = clamp.(x .+ scale .* dx, lower, upper)
      all(abs.(xt .- x) .<= 1e-15 .* (1 .+ abs.(x))) && (isnan(kind) ? continue : break)
      predicted = f - sum(abs2, r .+ J*(xt .- x)) # reduction if the residuals were linear
      try
        setknobs!(prob, xt)
        # (the first trial is usually accepted: compute its Jacobian with it)
        vt, Jt = _evaluate(prob.exprs, knobs, trial == 1)
        evaluations += 1
        rawt = copy(vt)
        rt, Jt = _residuals!(prob, vt, Jt)
        ft = sum(abs2, rt)
      catch err
        err isa InterruptException && (setknobs!(prob, x); rethrow())
        ft = Inf
      end
      if ft < f
        if isnan(kind)
          # (Nielsen's update: less damping the better the linear prediction was)
          rho = predicted > 0 ? (f - ft)/predicted : 0.0
          lambda *= max(1/3, 1 - (2*rho - 1)^3)
          nu = 2.0
          gauss_newton = rho > 0.75
        else
          gauss_newton = true
        end
        accepted = true
        break
      end
      if kind == -0.125
        gauss_newton = false
      elseif isnan(kind)
        lambda *= nu
        nu *= 2
        if lambda > 1e12*F.S[1]^2 # (no damped step helps: go on with the trials after them)
          lambda = 1e-6*F.S[1]^2
          nu = 2.0
          for k in trial+1:length(kinds)
            isnan(kinds[k]) && (kinds[k] = Inf)
          end
        end
      end
    end
    if !accepted
      setknobs!(prob, x)
      status = stalled
      break
    end
    if isnothing(Jt)
      vt, Jt = _evaluate(prob.exprs, knobs, true)
      evaluations += 1
      rawt = copy(vt)
      rt, Jt = _residuals!(prob, vt, Jt)
    end
    small = f - ft <= 1e-13*f
    x, r, J, raw, f = xt, rt, Jt, rawt, ft
    iterations += 1
    push!(history, f)
    verbose && println("match!: iteration ", iterations, ", sum of squared residuals ", f,
                       ", damping ", lambda)
    if small && !(nt > 0 && met(raw) && isempty(prob.minimize))
      status = nt > 0 && met(raw) ? "all targets met" : "no further improvement possible"
      break
    end
  end
  converged = nt > 0 ? met(raw) : status == "no further improvement possible"
  return MatchResult(prob, converged, status, iterations, evaluations, time() - t_start, x0, x,
                     raw, history)
end

# match! with the Levenberg-Marquardt method of BatchSolve. The residuals and their Jacobian come
# from one evaluation, so the Jacobian is taken at the trial points (trial_jac); an evaluation
# that fails (e.g. an unstable lattice) is a non-finite residual, which rejects the step
function _match_levenberg_marquardt!(prob::MatchProblem, maxiter, verbose)
  t_start = time()
  knobs = prob.knobs
  x0 = prob.x
  lower, upper = prob.lower, prob.upper
  all(lower .<= x0 .<= upper) || error("A knob starts outside of its bounds")
  maxstep = Float64[k.step for k in knobs]
  evaluations = Ref(0)
  history = Float64[]
  seen_x = Vector{Float64}[] # the last points evaluated and the values of the expressions there
  seen_vals = Vector{Float64}[]
  function evaluate!(y, J, x)
    evaluations[] += 1
    try
      setknobs!(prob, x)
      vals, Jx = _evaluate(prob.exprs, knobs, !isnothing(J))
      push!(seen_x, copy(x))
      push!(seen_vals, copy(vals))
      length(seen_x) > 4 && (popfirst!(seen_x); popfirst!(seen_vals))
      _residuals!(prob, vals, Jx)
      y .= vals
      isnothing(J) || (J .= Jx)
      cost = sum(abs2, vals)
      (isempty(history) || cost < history[end]) && push!(history, cost)
    catch err
      err isa InterruptException && (setknobs!(prob, x0); rethrow())
      if evaluations[] == 1
        setknobs!(prob, x0)
        error("match!: the targets cannot be computed at the starting values of the knobs (",
              sprint(showerror, err), "). Start from a lattice that has a solution.")
      end
      y .= NaN
      isnothing(J) || (J .= NaN)
    end
    return nothing
  end
  y = zeros(length(prob.exprs))
  jac = zeros(length(prob.exprs), length(knobs))
  # Converged when the residuals are below the tightest tolerance (then every target is met),
  # else at a point that small steps do not improve
  abstol = minimum((t.tol*t.weight for t in prob.targets); init=Inf)
  abstol = isempty(prob.minimize) && isfinite(abstol) ? abstol : 0.0
  sol = BatchSolve.levenberg_marquardt!((y, x) -> evaluate!(y, nothing, x), evaluate!, y, jac,
          copy(x0); maxiter, verbose, abstol, reltol=1e-12, gradtol=1e-9, trial_jac=true,
          lower=any(isfinite, lower) ? lower : nothing, upper=any(isfinite, upper) ? upper : nothing,
          maxstep=any(isfinite, maxstep) ? maxstep : nothing)
  x = sol.u
  setknobs!(prob, x)
  k = findlast(==(x), seen_x)
  raw = isnothing(k) ? _evaluate(prob.exprs, knobs, false)[1] : seen_vals[k]
  met = all(abs(raw[i] - clamp(raw[i], t.lower, t.upper)) <= t.tol
            for (i, t) in enumerate(prob.targets))
  nt = length(prob.targets)
  if sol.retcode == RETCODE_MAXITER
    status = "maximum number of iterations"
  elseif nt > 0 && met
    status = "all targets met"
  else
    status = "no further improvement possible"
  end
  converged = nt > 0 ? met : sol.retcode == RETCODE_SUCCESS
  return MatchResult(prob, converged, status, sol.iters, evaluations[], time() - t_start, x0, x,
                     raw, history)
end

function match!(; vary, targets=Target[], minimize=LatticeExpr[], minimize_weight=1,
                kwargs...)
  return match!(MatchProblem(; vary, targets, minimize, minimize_weight); kwargs...)
end
function match!(bl::Beamline; vary, targets=Target[], minimize=LatticeExpr[],
                minimize_weight=1, twiss=(;), kwargs...)
  tw = TwissExpr(bl; twiss...)
  targets isa Function && (targets = targets(tw))
  minimize isa Function && (minimize = minimize(tw))
  return match!(MatchProblem(; vary, targets, minimize, minimize_weight); kwargs...)
end
match!(targets::Function, bl::Beamline; kwargs...) = match!(bl; targets, kwargs...)
match!(res::MatchResult; kwargs...) = match!(res.problem; kwargs...)
