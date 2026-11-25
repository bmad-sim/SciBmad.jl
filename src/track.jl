"""
    struct TrackConfig

Defines configuration settings for a particles tracking run.

## Properties
- `n_turns::Int`: Number of turns, defaults to `1`
- `save_every_n_turns::Int`: Multiple of turn number which specifies the turns tracking 
    data is saved, defaults to `1`
- `in_body_coordinates::Bool`: If `true`, particles saved inside misaligned elements are not
    transformed back to the reference curve coordinates. Default is `false`
- `save_reference::Bool`: If `true`, the reference time and energy at each location and saved
    turn are saved too (`t_ref` and `beta_gamma_ref` of the result). This compiles the tracking
    separately and makes it slower (the reference time is computed at every integration step).
    Default is `false`
- `scalar_params::Bool`: If `true`, only uses the primal values of any `LatElement` 
    properties set as `GTPSA.TPS`, `ForwardDiff.Dual`, or `ReverseDiff.TrackedReal` 
    (ignores derivatives part, useful for closed orbit finding prior to parametric 
    normal form). Default is `false`
- `rf_on::Bool`: If `false`, then any `RFParams` in any `LatElement`s are ignored for the 
    entirety of the tracking. Default is `true`
- `ramp_particle_energy_without_rf::Bool`: If `true` when ramping (`Beamline`'s reference 
    energy is a `TimeFunction`), then particle energies will be artificially ramped with 
    the reference energy. Default is `false`
- `ramp_update_each_particle::Bool`: If `true` when ramping, then each particle sees the
    reference energy at its own time (from its z-offset) at the element entrances, and the
    reference energy also changes through each element. If `false`, all particles see the
    reference energy at the reference time at each element entrance. Usually only needed if
    ramping is fast compared to the time scale of bunch passage through elements. (Time-dependent
    element parameters are always evaluated at each particle's own time.) Default is `false`.
- `batch_start::Int`: Specifies the starting index for `BatchParam`s that particles in a 
    bunch will see. E.g., if `batch_start=3`, then particle 1 sees index 3 in all 
    `BatchParam`s, particle 2 sees index 4, etc. Indexing of `BatchParam`s will loop 
    around to the start if the particle index exceeds the length of the `BatchParams`. 
    Default is `1`. WARNING: setting this argument will caused unaligned memory access on 
    the GPU, which may reduce performance. In this case, consider `circshift`-ing all 
    `BatchParam`s instead.
- `verbose::Bool`: If `true`, print the the status of tracking. Default is `false`
- `groupsize::Int`: Designates a group of threads acting in parallel, preferably in 
    lockstep. If tracking on a CUDA GPU, this is equal to the "block" size. On the CPU, 
    would be a multiple of the vector register width. Default is `0` which lets 
    `KernelAbstractions` decide.
- `use_cpu_multithreading::Bool `: If `true`, then CPU multithreading will be used. Only 
    beneficial for a large number of particles, as there is a cost to launching threads. 
    Default is `false`.
- `use_KA::Bool`: If `true`, uses `KernelAbstractions` to launch tracking kernels. Default 
    is `true`, and it must be `true` for the GPU. 
- `use_explicit_SIMD::Bool`: If `true`, uses `SciBmad`'s explicit CPU SIMD tracking, which
    often guarantees vectorization of tracking on the CPU. Default is `!use_KA`
"""
@kwdef struct TrackConfig
  n_turns::Int                          = 1
  save_every_n_turns::Int               = 1
  in_body_coordinates::Bool             = false
  save_reference::Bool                  = false
  scalar_params::Bool                   = false
  rf_on::Bool                           = true
  ramp_particle_energy_without_rf::Bool = false
  ramp_update_each_particle::Bool       = false
  batch_start::Int                      = 1
  verbose::Bool                         = false
  groupsize::Int                        = 0 # autoselect
  use_cpu_multithreading::Bool          = false
  use_KA::Bool                          = true
  use_explicit_SIMD::Bool               = !use_KA
end

function Base.show(io::IO, config::TrackConfig)
  fields = fieldnames(typeof(config))
  width = maximum(length, String.(fields))
  println(io, nameof(typeof(config)))
  for field in fields
    println(io, " ", rpad(String(field), width), " = ", repr(getproperty(config, field)))
  end
  return
end

"""
    struct TrackResult

Contains all information for a given particle tracking run. The particles are read at locations
of the beamline (given as for the columns of `twiss`; without a location, the only saved
location, by default the end of the beamline), for a saved turn and a particle:

```julia
res.x[turn=2]                     # x of all particles after turn 2 (at the only saved location)
res.x[turn=2, particle=1]         # of particle 1
res.x[]                           # [particle, saved turn]
res.px[qd, turn=2]                # at the beginning of the element qd (see `at` in `track`)
res.v[qd, :exit, particle=1]      # all six coordinates at its end: [coordinate, saved turn]
```

## Properties
- `x`, `px`, `y`, `py`, `z`, `pz`: the coordinates, `v`: all six (the coordinate is the second
  index), `q`: the spin quaternions if spin was tracked, `state`: the state of the particles
  (`STATE_ALIVE`, ...), `t_ref`, `beta_gamma_ref`: the reference time and energy (one value per
  location and turn) if `save_reference=true`. See [`TrackColumn`](@ref) for how they are read
- `turns`: the saved turns (turn 0: the initial coordinates; `NaN` at the locations before
  `start`)
- `start`: the beamline index of the element where the tracking started (see `start` in
  [`track`](@ref))
- `s`, `index`, `name`, `kind`: the s, the beamline index (`-1` for the end of the beamline), the
  name and the kind of the element of each saved location
- `data`: the saved arrays, indexed by location number (in the order of `s`) and saved turn (1:
  the initial coordinates): `state[particle, location, saved turn]`,
  `v[particle, coordinate, location, saved turn]`, `q[particle, quaternion, location, saved turn]`
  (or `nothing`), `t_ref[location, saved turn]` and `beta_gamma_ref[location, saved turn]` (or
  `nothing`; per particle with `BatchParam`s: a `BatchParam` for each location and turn)
- `config`: the `TrackConfig`, `elapsed_time`: the tracking time in seconds, `bunch`: the
  `Bunch` at the end of tracking
"""
struct TrackResult{L,D}
  config::TrackConfig
  elapsed_time::Float64 # in seconds
  start::Int
  s::L
  index::Vector{Int}
  name::Vector{String}
  kind::Vector{String}
  data::D        # (state, v, q, t_ref, beta_gamma_ref): the saved arrays
  bunch::Bunch
  bl::Beamline
  rows::Dict{Tuple{Int,Int},Int} # the location number of each (beamline index, step in element)
end

const _TRACK_COLUMNS = (:x, :px, :y, :py, :z, :pz, :v, :q, :state, :t_ref, :beta_gamma_ref)

function Base.getproperty(res::TrackResult, key::Symbol)
  key in _TRACK_COLUMNS && return TrackColumn(res, key)
  if key === :turns
    config = getfield(res, :config)
    return collect(0:config.save_every_n_turns:config.n_turns)
  end
  return getfield(res, key)
end

Base.propertynames(::TrackResult) = (_TRACK_COLUMNS..., :turns, :start, :s, :index, :name, :kind,
                                        :data, :config, :elapsed_time, :bunch)

function Base.show(io::IO, res::TrackResult)
  n_particles, n_locations, n_slots = size(getfield(res, :data).state)
  print(io, "TrackResult(", n_particles, n_particles == 1 ? " particle, " : " particles, ",
        n_locations, n_locations == 1 ? " location, " : " locations, ", n_slots,
        " saved turns, start = ", res.start, ")")
end

function Base.show(io::IO, ::MIME"text/plain", res::TrackResult)
  data = getfield(res, :data)
  config = getfield(res, :config)
  n_particles = size(data.state, 1)
  turns = res.turns
  single = length(getfield(res, :index)) == 1 # (then the location can be left out)
  loc = single ? "" : "location, "
  println(io, "TrackResult of ", n_particles, n_particles == 1 ? " particle:" : " particles:")
  println(io, "config:")
  fields = fieldnames(TrackConfig)
  width = maximum(length ∘ String, fields)
  for field in fields
    println(io, " ", rpad(String(field), width), " = ", repr(getfield(config, field)))
  end
  println(io)
  println(io, "elapsed_time = ", getfield(res, :elapsed_time), " seconds")
  println(io)
  println(io, "start = ", getfield(res, :start))
  println(io, "turns = ", length(turns) > 6 ?
              "$(first(turns)):$(config.save_every_n_turns):$(last(turns))" : string(turns))
  println(io)
  dims(x) = "[" * join(("1:$n" for n in size(x)), ", ") * "]"
  rows = Tuple{String,String}[
    ("s, index, name, kind", "the saved locations: " * _track_locations_string(res)),
    ("state", "particle state codes, e.g. " * _kwindex(io, "state", loc * "turn=n, particle=i")),
    ("x, px, y, py, z, pz", "the coordinates, e.g. " *
                            _kwindex(io, "x", loc * "turn=n, particle=i")),
    ("v", "all six coordinates, e.g. " * _kwindex(io, "v", loc * "turn=n, particle=i"))]
  isnothing(data.q) ||
    push!(rows, ("q", "spin quaternions (track_spin turns them into spin 3-vectors)"))
  isnothing(data.t_ref) ||
    push!(rows, ("t_ref, beta_gamma_ref", "reference time and energy, e.g. " *
                 _kwindex(io, "t_ref", loc * "turn=n")))
  push!(rows, ("data", "the internal, low-level data arrays:"))
  order = (state="[particle, location, saved turn]",
           v="[particle, coordinate, location, saved turn]",
           q="[particle, quaternion, location, saved turn]", t_ref="[location, saved turn]",
           beta_gamma_ref="[location, saved turn]")
  for k in keys(data)
    x = getfield(data, k)
    isnothing(x) && continue
    push!(rows, (" " * "$k$(dims(x))", ndims(x) == 3 && k !== :state ?
                                        "[particle, location, saved turn]" : getfield(order, k)))
  end
  push!(rows, ("bunch", "the Bunch at the end of tracking"))
  w = max(22, maximum(length(r[1]) for r in rows) + 1)
  for (k, (name, text)) in enumerate(rows)
    print(io, rpad(name, w), text)
    k < length(rows) && println(io)
  end
end

# The saved locations by name and s (the first and last of many)
function _track_locations_string(res)
  idxs, names, s = getfield(res, :index), getfield(res, :name), getfield(res, :s)
  loc(l) = (idxs[l] == -1 ? "the end of the beamline" : names[l]) *
           " (s = $(round(scalar(s[l]), sigdigits=6)))"
  n = length(idxs)
  n <= 6 && return join((loc(l) for l in 1:n), ", ")
  return "$n locations, from $(loc(1)) to $(loc(n))"
end

"""
    TrackColumn

A quantity of a [`TrackResult`](@ref) (e.g. `res.x`), read at locations of the beamline as
the columns of `twiss` (see [`TwissColumn`](@ref)), and for a saved turn and a particle given
with the keywords `turn` (the turn number, 0 for the initial coordinates) and `particle`,
written after the location (or alone):

- `res.x[]`: without a location, at the only saved location (by default the end of the
  beamline; with several saved locations, one must be given), `[particle, saved turn]`
- `res.x[turn=n]`: all particles after turn `n`, `res.x[turn=n, particle=i]`: one number.
  `turn` and `particle` can also be `:` (all, as without them), a range or a vector
- `res.x[qd, turn=n]`, `res.x[5, ...]`, `res.x[qd, :exit, ...]`, `res.x[end, ...]`: at the
  beginning of `qd` (placed once) or of the element with beamline index 5, at its end, at the end
  of the beamline (`end` is the beamline index of the end, one more than the number of elements)
- `res.x[qd, :]`, `res.x[[qd]]`, `res.x["q*"]`, `res.x[(2.0, 6.0)]`: at several locations (every
  step of `qd`, every placement of `qd`, ...; as for `twiss`): their index comes after the
  particle (`[particle, location, saved turn]`)

`res.v`, `res.q` and the spins from [`track_spin`](@ref) have the coordinate, quaternion or spin
component as the index after the particle, and `t_ref` and `beta_gamma_ref` no particle index
(unless they are per particle). The locations must have been saved (see `at` in [`track`](@ref)).
"""
struct TrackColumn{R<:TrackResult,A}
  res::R
  name::Symbol
  data::A # the array of a quantity computed from the result (the spins), else nothing
end

TrackColumn(res::TrackResult, name::Symbol) = TrackColumn(res, name, nothing)

# (with more than one index, e.g. x[end, turn=2], `end` is lastindex(c, 1); n+1: the end)
Base.firstindex(::TrackColumn, d...) = 1
Base.lastindex(c::TrackColumn, d...) = length(getfield(c.res, :bl).line) + 1

function Base.getindex(c::TrackColumn, args...;
                       turn::Union{Nothing,Colon,Integer,AbstractVector{<:Integer}}=nothing,
                       particle=nothing)
  res = c.res
  raw, k = isnothing(c.data) ? _track_raw(getfield(res, :data), c.name) : (c.data, nothing)
  if isempty(args) # the only saved location
    n = length(getfield(res, :index))
    n == 1 || error(n == 0 ? "No locations were saved (`at=[]`)" : "$n locations were saved: " *
                    "give one, e.g. $(c.name)[qd, turn=n] or $(c.name)[end, turn=n]")
    L = 1
  else
    bl = getfield(res, :bl)
    locs, one = _column_locations(bl, (1, length(bl.line) + 1), args...)
    ls = [_track_location(res, l) for l in locs]
    L = one ? only(ls) : ls
  end
  config = getfield(res, :config)
  S = isnothing(turn) || turn isa Colon ? (:) : turn isa Integer ? _track_slot(config, turn) :
      [_track_slot(config, n) for n in turn]
  P = isnothing(particle) ? (:) : particle
  I = ndims(raw) == 2 ? (L, S) : ndims(raw) == 3 ? (P, L, S) : (P, something(k, :), L, S)
  # (one value: through a copy, also from a GPU array)
  return all(i -> i isa Integer, I) ? Array(view(raw, I...))[] : raw[I...]
end

function Base.show(io::IO, c::TrackColumn)
  single = length(getfield(c.res, :index)) == 1
  loc = single ? "" : "location, "
  print(io, "TrackColumn `", c.name, "` (read it with `",
        _kwindex(io, String(c.name), loc * "turn=n, particle=i"), "`)")
end

# Indexing with keywords in the hints of the displays: `x[turn=n]`, or `x(turn=n)` for Python
# (Python has no keywords in indexing; its interface shows values with `:python => true`)
_kwindex(io, name, args) = get(io, :python, false)::Bool ? "$name($args)" : "$name[$args]"

# The saved array of the quantity name, and the coordinate (for x, ..., pz)
function _track_raw(data, name)
  k = findfirst(==(name), (:x, :px, :y, :py, :z, :pz))
  isnothing(k) || return data.v, k
  raw = getfield(data, name)
  isnothing(raw) && error(name === :q ? "The spin was not tracked (`spin=true`)" :
                          "The reference was not saved (`save_reference=true`)")
  return raw, nothing
end

# The location number of (i, k): k integration steps after the beginning of the element with
# beamline index i (-1: the end of the beamline)
function _track_location(res, (i, k))
  l = get(getfield(res, :rows), (i, k), 0)
  l == 0 && error("The particles were not saved ", i == -1 ? "at the end of the beamline" :
                  k == 0 ? "at the beginning of element $i" : "at step $k of element $i",
                  ": include it with `at`")
  return l
end

# The saved turn (slot) of turn n
function _track_slot(config, n)
  m = config.save_every_n_turns
  0 <= n <= config.n_turns && n % m == 0 ||
    error("Turn $n was not saved (the saved turns are 0:$m:$(config.n_turns))")
  return n ÷ m + 1
end

"""
    track(bl::Beamline; kwargs...) -> TrackResult

All-encompassing function to track particles through the beamline `bl` using the configured 
settings specified by the keyword arguments `kwargs`.

## Keyword Arguments
- `v0`: A matrix of size `(n_particles, 6)` storing the initial particle phase space coordinates
- `spin`: If `true`, spin tracking will be enabled with identity quaternions as initial quaternions
- `q0`: A matrix of size `(n_particles, 4)` storing the initial particle spin quaternions if spin tracking
- `weight`: Optional vector of length `(n_particles)` specifying macroparticle weights per particle, default 
    is `nothing` corresponding to uniform weights for all particles
- `species`: The species of the particles, default the reference species of the beamline
    (`bl.species_ref`)
- `bunch`: Instead of specifying `v0`, `spin`, `q0`, `weight`, and `species`, an initial `Bunch`
    may be explicitly provided
- `start::Union{Integer,LatElement}`: The element (or its beamline index) at whose beginning the
    tracking starts. Default is `1`
- `at::Union{Nothing,Colon,AbstractVector}`: Where the particles are saved, as in `twiss`: a
    vector of locations, each as one index of a column (see [`TrackColumn`](@ref)): elements
    (at the beginning of each placement), beamline indexes, names (as for `findelements`), and
    s-ranges (tuples: the integrator steps inside them, and the end of the beamline if they
    contain it). `:` is every integrator step and the end of the beamline. An entry that is not
    in the beamline is an error. Default is `nothing`, only the end of the beamline. The
    locations are ordered in s (from the beginning of the beamline).
    The locations before `start` are first reached in the second turn: before (turns 0 and 1)
    they are `NaN` (`state` `STATE_PREBORN`)
- `n_turns::Int`: Number of turns, each ending at the end of the beamline (the first turn
    begins at `start`), defaults to `1`
- `save_every_n_turns::Int`: Multiple of turn number which specifies the turns tracking 
    data is saved, defaults to `1`
- `in_body_coordinates::Bool`: If `true`, particles saved inside misaligned elements are not
    transformed back to the reference curve coordinates. Default is `false`
- `save_reference::Bool`: If `true`, the reference time and energy at each location and saved
    turn are saved too (`t_ref` and `beta_gamma_ref` of the result). This compiles the tracking
    separately and makes it slower (the reference time is computed at every integration step).
    Default is `false`
- `scalar_params::Bool`: If `true`, only uses the primal values of any `LatElement` 
    properties set as `GTPSA.TPS`, `ForwardDiff.Dual`, or `ReverseDiff.TrackedReal` 
    (ignores derivatives part, useful for closed orbit finding prior to parametric 
    normal form). Default is `false`
- `rf_on::Bool`: If `false`, then any `RFParams` in any `LatElement`s are ignored for the 
    entirety of the tracking. Default is `true`
- `ramp_particle_energy_without_rf::Bool`: If `true` when ramping (`Beamline`'s reference 
    energy is a `TimeFunction`), then particle energies will be artificially ramped with 
    the reference energy. Default is `false`
- `ramp_update_each_particle::Bool`: If `true` when ramping, then each particle sees the
    reference energy at its own time (from its z-offset) at the element entrances, and the
    reference energy also changes through each element. If `false`, all particles see the
    reference energy at the reference time at each element entrance. Usually only needed if
    ramping is fast compared to the time scale of bunch passage through elements. (Time-dependent
    element parameters are always evaluated at each particle's own time.) Default is `false`.
- `batch_start::Int`: Specifies the starting index for `BatchParam`s that particles in a 
    bunch will see. E.g., if `batch_start=3`, then particle 1 sees index 3 in all 
    `BatchParam`s, particle 2 sees index 4, etc. Indexing of `BatchParam`s will loop 
    around to the start if the particle index exceeds the length of the `BatchParams`. 
    Default is `1`. WARNING: setting this argument will caused unaligned memory access on 
    the GPU, which may reduce performance. In this case, consider `circshift`-ing all 
    `BatchParam`s instead.
- `verbose::Bool`: If `true`, print the the status of tracking. Default is `false`
- `groupsize::Int`: Designates a group of threads acting in parallel, preferably in 
    lockstep. If tracking on a CUDA GPU, this is equal to the "block" size. On the CPU, 
    would be a multiple of the vector register width. Default is `0` which lets 
    `KernelAbstractions` decide.
- `use_cpu_multithreading::Bool `: If `true`, then CPU multithreading will be used. Only 
    beneficial for a large number of particles, as there is a cost to launching threads. 
    Default is `false`.
- `use_KA::Bool`: If `true`, uses `KernelAbstractions` to launch tracking kernels. Default 
    is `true` if `v0` is a GPU array, else `false`.
- `use_explicit_SIMD::Bool`: If `true`, uses `SciBmad`'s explicit CPU SIMD tracking, which
    often guarantees vectorization of tracking on the CPU. Default is `!use_KA`
- `config`: Instead of specifying `n_turns`, `save_every_n_turns`, `in_body_coordinates`,
    `save_reference`, `scalar_params`,
    `ramp_particle_energy_without_rf`, `ramp_update_each_particle`, `verbose`, `groupsize`, 
    `use_cpu_multithreading`, `use_KA`, and `use_explicit_SIMD`, a single `TrackConfig` 
    struct can be provided instead. Setting any of these keyword arguments will override 
    that in `config`.
"""
function track(
    bl::Beamline;

    # User can either specify kwargs to construct a bunch:
    v0::AbstractMatrix=zeros(1, 6),
    spin::Bool=false,
    q0::Union{AbstractMatrix,Nothing}=spin ? (q = similar(v0, (size(v0, 1), 4)); q .= 0; q[:,1] .= 1; q) : nothing,
    weight::Union{AbstractArray,Number,Nothing}=nothing,
    callbacks::Tuple=(),
    species=bl.species_ref,

    # Or explicitly provide a Bunch:
    bunch::Bunch=Bunch(; 
      v=(v = similar(v0); v .= copy.(v0); v), spin=spin,
      q=(!isnothing(q0) ? copy.(q0) : nothing), 
      weight=(!isnothing(weight) ? copy.(weight) : nothing), 
      species=species,
      p_over_q_ref=(_p_over_q_ref = bl.p_over_q_ref; _p_over_q_ref isa TimeDependentParam ? _p_over_q_ref(0) : _p_over_q_ref),
      t_ref=begin
        s_end = bl.line[end].s_downstream # (sums the lengths of all elements, 0.8 ms on ESR)
        if bl.p_over_q_ref isa BatchParam
          p = bl.p_over_q_ref.batch
          b = similar(p, length(p))
          b .= 0
          BatchParam(b) 
        elseif s_end isa BatchParam 
          s = s_end.batch
          b = similar(s, length(s))
          b .= 0
          BatchParam(b) 
        else
          zero(s_end)
        end
      end, 
      callbacks=callbacks,
    ),

    start::Union{Integer,LatElement}=1,
    at::Union{Nothing,Colon,AbstractVector}=nothing,

    config=TrackConfig(use_KA=!(KA.get_backend(bunch.v) isa KA.CPU)),
    
    # Tracking customization kwargs:
    n_turns                         = config.n_turns,
    save_every_n_turns              = config.save_every_n_turns,
    in_body_coordinates             = config.in_body_coordinates,
    save_reference                  = config.save_reference,
    scalar_params                   = config.scalar_params,
    rf_on                           = config.rf_on,
    ramp_particle_energy_without_rf = config.ramp_particle_energy_without_rf,
    ramp_update_each_particle       = config.ramp_update_each_particle,
    batch_start                     = config.batch_start,
    verbose                         = config.verbose,

    # Low-level launch! kwargs (these are not considered stable API and may change):
    groupsize                       = config.groupsize,
    use_cpu_multithreading          = config.use_cpu_multithreading,
    use_KA                          = config.use_KA,
    use_explicit_SIMD               = config.use_explicit_SIMD,
  )  

  newconfig = TrackConfig(
    n_turns,
    save_every_n_turns,
    in_body_coordinates,
    save_reference,
    scalar_params,
    rf_on,
    ramp_particle_energy_without_rf,
    ramp_update_each_particle,
    batch_start,
    verbose,
    groupsize,
    use_cpu_multithreading,
    use_KA,
    use_explicit_SIMD,
  )
  n_turns >= 1 || error("n_turns must be at least 1")
  # (the element or one made from it, placed once)
  start = _element_index(bl, start, "start")
  1 <= start <= length(bl.line) || error("start must be a beamline index of an element")
  # Locations where the particles are saved, as in twiss (by default only the end of the
  # beamline), and the steps of the turn before the start element
  if at isa AbstractVector && isempty(at) # nothing to save (1.5 ms less on ESR than resolving it)
    loc_s, loc_names, loc_kinds, loc_idxs = Float64[], String[], String[], Int[]
    loc_steps = Int[]
    loc_ele_steps = Int[]
    n_steps_turn = 0
    step_start = start > 1 ? _line_steps(bl)[4][start] : 0
  else
    ls = _line_steps(bl)
    locs = isnothing(at) ? [(-1, 0)] : _at_locations(bl, (1, length(bl.line) + 1), at, ls)
    loc_s, loc_idxs, loc_names, loc_kinds, loc_steps, loc_ele_steps = _location_info(bl, locs, ls)
    n_steps_turn = last(ls[4])
    step_start = ls[4][start]
  end
  # Function barrier for Bunch and groupsize (needs special handling for KA)
  locations = (loc_s, loc_idxs, loc_names, loc_kinds, loc_ele_steps)
  return @noinline _track(bl, bunch, newconfig, groupsize == 0 ? nothing : groupsize, step_start,
                          locations, loc_steps, n_steps_turn, Val(save_reference), start)
end

# locations are the s, beamline index, name, kind and step within the element of each location.
# loc_steps are the steps (tracking callbacks) of the locations in a turn, counted from the
# beginning of the beamline (step 0), and n_steps_turn is the number of steps of a turn (the end of
# the beamline). The first turn begins after step_start steps. The locations are saved by a
# callback during the turns, as in twiss
function _track(bl, bunch, config, groupsize, step_start, locations, loc_steps, n_steps_turn,
                ::Val{save_reference}, _start=1) where {save_reference}
  state = bunch.coords.state
  v     = bunch.coords.v
  q     = bunch.coords.q
  n_turns                         = config.n_turns
  save_every_n_turns              = config.save_every_n_turns
  in_body_coordinates             = config.in_body_coordinates
  scalar_params                   = config.scalar_params
  rf_on                           = config.rf_on
  ramp_particle_energy_without_rf = config.ramp_particle_energy_without_rf
  ramp_update_each_particle       = config.ramp_update_each_particle
  batch_start                     = config.batch_start
  verbose                         = config.verbose
  use_cpu_multithreading          = config.use_cpu_multithreading
  use_KA                          = config.use_KA
  use_explicit_SIMD               = config.use_explicit_SIMD

  n_particles = size(v, 1)
  n_locations = length(loc_steps)

  # The initial particles, then every save_every_n_turns-th turn
  n_data_pts = div(n_turns, save_every_n_turns) + 1
  slot_of_turn(i) = mod(i, save_every_n_turns) == 0 ? div(i, save_every_n_turns) + 1 : 0

  state_data = similar(state, n_particles, n_locations, n_data_pts)
  v_data = similar(v, n_particles, 6, n_locations, n_data_pts)
  q_data = isnothing(q) ? nothing : similar(q, n_particles, 4, n_locations, n_data_pts)
  # The reference at the locations if saved: one value per location and slot (written by the
  # first particle), or one per particle with BatchParams
  t_data = save_reference ?
           _reference_data(v, bunch.t_ref, n_particles, n_locations, n_data_pts) : nothing
  bg_data = save_reference ?
            _reference_data(v, _beta_gamma_ref(bunch), n_particles, n_locations, n_data_pts) :
            nothing
  data = (state_data, v_data, q_data, t_data, bg_data)
  _track_copy_all!(data, bunch, batch_start, 1:n_locations, 1)

  # The kernel chains of the elements are built once for all turns (not included in the time t)
  chains = BTBL.prepare(bunch, bl; scalar_params, rf_on, ramp_particle_energy_without_rf,
                        ramp_update_each_particle, batch_start)

  # (the steps of the chains, counted by BeamTracking, must be those of the locations)
  n_steps_turn == 0 || isempty(chains) || last(chains).step_exit == n_steps_turn ||
    error("The steps of the chains differ from those of the locations")
  if step_start > 0
    # The first turn begins with the first chain of the start element: the first that ends after
    # step_start steps of the turn. The locations before are not reached in the first turn, so
    # they hold no particles before the second (also not the initial ones)
    first_chain = findfirst(pc -> pc.step_exit > step_start, chains)
    for j in (1, slot_of_turn(1))
      j == 0 && continue
      ls = 1:count(<(step_start), loc_steps) # (the locations are ordered in s)
      state_data[:,ls,j] .= BeamTracking.STATE_PREBORN
      v_data[:,:,ls,j] .= convert(eltype(v_data), NaN)
      if !isnothing(q)
        q_data[:,:,ls,j] .= convert(eltype(q_data), NaN)
      end
      _set_reference_nan!(t_data, ls, j)
      _set_reference_nan!(bg_data, ls, j)
    end
  else
    first_chain = 1
  end

  # location_of_step[p.step] is 0, or the location the callback saves at that step. The callback
  # runs after each integration step, so the location where a turn begins (if any: at step_start
  # in the first turn, then at step 0) is saved before the turn instead
  _location_of_step = zeros(Int, n_steps_turn)
  for (l, step) in enumerate(loc_steps)
    if step > 0
      _location_of_step[step] = l
    end
  end
  location_of_step = similar(v, Int, n_steps_turn)
  copyto!(location_of_step, _location_of_step)
  l_first = findfirst(==(step_start), loc_steps)
  l_begin = findfirst(==(0), loc_steps)
  slot = similar(v, Int, 1) # the slot of the turn in the saved data (or 0 if not saved)
  # (on the CPU through pointers: arrays in the callback would be GC-rooted in every kernel, ~4%
  # slower for one particle on ESR; they are preserved below)
  K = _kernel_array
  cb = _track_make_callback(K(slot), K(location_of_step), in_body_coordinates, K(v_data),
                            K(q_data), K(state_data), K(t_data), K(bg_data))
  # Always tracked with the callback appended to those of the bunch, so that all tracking compiles
  # the same kernels
  coords = BeamTracking.Coords(state, v, q, bunch.coords.weight, (bunch.coords.callbacks..., cb))
  b = Bunch(bunch.species, bunch.p_over_q_ref, bunch.t_ref, coords)

  t = @elapsed GC.@preserve slot location_of_step data for i in 1:n_turns
    j = slot_of_turn(i)
    l = i == 1 ? l_first : l_begin
    if j != 0 && !isnothing(l)
      _track_copy_all!(data, b, batch_start, l, j)
    end
    slot .= j
    track!(b, i == 1 && first_chain > 1 ? chains[first_chain:end] : chains; groupsize, use_KA,
           use_explicit_SIMD, use_cpu_multithreading)
    if verbose
      print("\rFinished turn $i out of $n_turns")
      flush(stdout)
    end
  end
  if verbose
    println()
  end
  bunch.species = b.species
  bunch.p_over_q_ref = b.p_over_q_ref
  bunch.t_ref = b.t_ref
  bunch.s = b.s
  bunch.step = b.step
  loc_s, loc_idxs, loc_names, loc_kinds, loc_ele_steps = locations
  rows = Dict{Tuple{Int,Int},Int}((loc_idxs[l], loc_ele_steps[l]) => l for l in eachindex(loc_idxs))
  saved = (; state=state_data, v=v_data, q=q_data,
           t_ref=_reference_result(t_data, bunch.t_ref, batch_start),
           beta_gamma_ref=_reference_result(bg_data, _beta_gamma_ref(bunch), batch_start))
  return TrackResult(config, t, _start, loc_s, loc_idxs, loc_names, loc_kinds, saved, bunch, bl, rows)
end

function _track_make_callback(_slot, _location_of_step, _in_body_coordinates, _v_data, _q_data,
                              _state_data, _t_data, _bg_data)
  let slot=_slot, location_of_step=_location_of_step, in_body_coordinates=_in_body_coordinates,
    v_data=_v_data, q_data=_q_data, state_data=_state_data, t_data=_t_data, bg_data=_bg_data
    return (i, coords, p) -> begin
      @inbounds begin
        step = p.step
        j = slot[1]
        if j != 0 && step <= length(location_of_step)
          l = location_of_step[step]
          if l != 0
            _track_save!(i, coords, p, in_body_coordinates, v_data, q_data, state_data, t_data,
                         bg_data, l, j)
          end
        end
      end
      return nothing
    end
  end
end

# All particles of bunch at the location(s) l in slot j (on the host)
function _track_copy_all!((state_data, v_data, q_data, t_data, bg_data), bunch, batch_start, l, j)
  state_data[:,l,j] .= bunch.state
  v_data[:,:,l,j] .= _saved.(bunch.v)
  if !isnothing(q_data)
    q_data[:,:,l,j] .= _saved.(bunch.q)
  end
  _set_reference!(t_data, bunch.t_ref, batch_start, l, j)
  _set_reference!(bg_data, _beta_gamma_ref(bunch), batch_start, l, j)
  return nothing
end

# Saved in the lab coordinates by transforming the saved copy, so the tracked coordinates are not
# written. The copy is out of line, and p is only used by the transforms: where they do nothing
# (aligned elements, the exits) the kernels then keep only p.step
@inline function _track_save!(i, coords, p, in_body_coordinates, v_data, q_data, state_data,
                              t_data, bg_data, l, j)
  _call_or_inline(_track_copy!, v_data, i, coords, v_data, q_data, state_data, l, j)
  _track_reference!(i, p, t_data, bg_data, l, j)
  if !in_body_coordinates
    saved = BeamTracking.Coords(_slot(state_data, l, j), _slot(v_data, l, j),
                                _slot(q_data, l, j), coords.weight, coords.callbacks)
    transforms_out!(i, saved, p)
  end
  return nothing
end

@inline function _track_copy!(i, coords, v_data, q_data, state_data, l, j)
  @inbounds begin
    state_data[i,l,j] = coords.state[i]
    for k in 1:6
      v_data[i,k,l,j] = _saved(coords.v[i,k])
    end
    if !isnothing(q_data)
      for k in 1:4
        q_data[i,k,l,j] = _saved(coords.q[i,k])
      end
    end
  end
  return nothing
end

# The reference: one value per location and slot (written by the first particle, all particles
# see the same), or with BatchParams one per particle (particle i sees the batch entry
# mod1(i + batch_start - 1, length of the batch))
# (if saved: without, p.t_ref and p.beta_gamma_ref are not used, so not computed in the kernels)
@inline _track_reference!(i, p, t_data, bg_data, l, j) =
  _call_or_inline(_track_reference!, t_data, i, p.t_ref, p.beta_gamma_ref, t_data, bg_data, l, j)
@inline _track_reference!(i, p, ::Nothing, ::Nothing, l, j) = nothing
@inline function _track_reference!(i, t, bg, t_data, bg_data, l, j)
  _track_reference!(i, t, t_data, l, j)
  _track_reference!(i, bg, bg_data, l, j)
  return nothing
end
@inline function _track_reference!(i, x, data::AbstractArray{<:Any,2}, l, j)
  if _first_particle(i) == 1
    @inbounds data[l,j] = _saved(_first_lane(x))
  end
  return nothing
end
@inline _track_reference!(i, x, data::AbstractArray{<:Any,3}, l, j) =
  (@inbounds data[i,l,j] = x; nothing)
_first_particle(i::Integer) = i
_first_particle(i::BeamTracking.SIMD.VecRange) = i.i
_first_lane(x) = x
_first_lane(x::BeamTracking.SIMD.Vec) = x[1]

_beta_gamma_ref(bunch) = Beamlines.R_to_beta_gamma(bunch.species, bunch.p_over_q_ref)

# The storage of a reference value x (in the kernels it has the number type of the coordinates,
# if they are numbers)
function _reference_data(v, x, n_particles, n_locations, n_slots)
  T = isbitstype(eltype(v)) ? eltype(v) : typeof(x isa BatchParam ? first(x.batch) : x)
  return x isa BatchParam ? similar(v, T, n_particles, n_locations, n_slots) :
                            similar(v, T, n_locations, n_slots)
end
_set_reference_nan!(data::AbstractArray{<:Any,2}, l, j) =
  (view(data, l, j) .= convert(eltype(data), NaN); nothing)
_set_reference_nan!(data::AbstractArray{<:Any,3}, l, j) =
  (view(data, :, l, j) .= convert(eltype(data), NaN); nothing)
_set_reference_nan!(::Nothing, l, j) = nothing

# x at the location(s) l in slot j (on the host; converted first, e.g. to Float32 on a GPU)
_set_reference!(::Nothing, x, batch_start, l, j) = nothing
function _set_reference!(data::AbstractArray{<:Any,2}, x, batch_start, l, j)
  if isbits(x)
    view(data, l, j) .= convert(eltype(data), x)
  else
    for k in l
      data[k,j] = copy(x)
    end
  end
  return nothing
end
function _set_reference!(data::AbstractArray{<:Any,3}, x::BatchParam, batch_start, l, j)
  B = length(x.batch)
  batch = Array(x.batch)
  values = similar(data, size(data, 1))
  copyto!(values, [convert(eltype(data), batch[mod1(i + batch_start - 1, B)]) for i in axes(data, 1)])
  view(data, :, l, j) .= values
  return nothing
end

# The result: the data, or with BatchParams a BatchParam for each location and slot, from the
# first particle that sees each batch entry (NaN for an entry no particle sees)
_reference_result(data, x, batch_start) = data
function _reference_result(data::AbstractArray{<:Any,3}, x::BatchParam, batch_start)
  d = Array(data)
  B = length(x.batch)
  entry(e, l, j) = (i = mod1(e - batch_start + 1, B); i <= size(d, 1) ? d[i,l,j] : eltype(d)(NaN))
  return [BatchParam([entry(e, l, j) for e in 1:B]) for l in axes(d, 2), j in axes(d, 3)]
end

# An Array of bits as seen by the tracking callback: its pointer and size, which the GC does not
# track (the Array must be preserved while it is used). GPU arrays are device arrays in the kernels
# anyway, and arrays of TPS are kept as they are
struct _PtrArray{T,N} <: DenseArray{T,N}
  ptr::Ptr{T}
  size::NTuple{N,Int}
end

Base.size(a::_PtrArray) = a.size
Base.IndexStyle(::Type{<:_PtrArray}) = IndexLinear()
Base.elsize(::Type{<:_PtrArray{T}}) where {T} = sizeof(T)
Base.unsafe_convert(::Type{Ptr{T}}, a::_PtrArray{T}) where {T} = a.ptr
@inline Base.getindex(a::_PtrArray, i::Int) = unsafe_load(a.ptr, i)
@inline Base.setindex!(a::_PtrArray, x, i::Int) = (unsafe_store!(a.ptr, x, i); a)
# A SIMD lane of particles (the first index) at once
@inline Base.getindex(a::_PtrArray{T}, i::BeamTracking.SIMD.VecRange{M}, I::Int...) where {T,M} =
  BeamTracking.SIMD.vload(BeamTracking.SIMD.Vec{M,T},
                          a.ptr + (LinearIndices(a)[i.i, I...] - 1)*sizeof(T))
@inline function Base.setindex!(a::_PtrArray{T}, x::BeamTracking.SIMD.Vec{M,T},
                                i::BeamTracking.SIMD.VecRange{M}, I::Int...) where {T,M}
  BeamTracking.SIMD.vstore(x, a.ptr + (LinearIndices(a)[i.i, I...] - 1)*sizeof(T))
  return a
end

_kernel_array(a::Array{T}) where {T} = isbitstype(T) ? _PtrArray(pointer(a), size(a)) : a
_kernel_array(a) = a

# The saving is a call on the CPU (the callback is in every kernel). In GPU kernels a call costs
# registers also where it is not taken (ESR on Metal: ~20% per turn), so there it is inlined
@inline _call_or_inline(f, ::Union{_PtrArray,Array}, args...) = @noinline f(args...)
@inline _call_or_inline(f, _, args...) = f(args...)

# The saved coordinates at location l and slot j (the last two dimensions of a)
@inline function _slot(a::_PtrArray{T,N}, l, j) where {T,N}
  sz = Base.front(Base.front(a.size))
  return _PtrArray(a.ptr + ((j-1)*a.size[N-1] + l-1)*prod(sz)*sizeof(T), sz)
end
@inline _slot(a::AbstractArray{T,N}, l, j) where {T,N} =
  view(a, ntuple(_ -> Colon(), Val{N-2}())..., l, j)
_slot(::Nothing, l, j) = nothing

# A copy of a coordinate that the tracking changes in place (TPS), else the value itself
@inline _saved(x) = isbits(x) ? x : copy(x)

"""
    track_spin(q::TrackColumn, s0::AbstractVecOrMat)
    track_spin(q::AbstractArray, s0::AbstractVecOrMat)

Given initial spin(s) `s0` and the quaternions `res.q` of a [`TrackResult`](@ref) `res`,
returns a [`TrackColumn`](@ref) `s` of the particles' spin 3-vectors, read like `res.q`: e.g.
`s[turn=n, particle=i]` is the spin of particle `i` after turn `n` (at the only saved location),
`s[qd, turn=n, particle=i]` at the beginning of `qd`.

Given the quaternion array `res.data.q` instead, returns an array `s` of size
`(n_particles, 3, n_locations, n_saved_turns)` of the spins at each location and saved turn
(any further dimensions of `q` after the second are kept as they are).

If `s0` is a vector (of length 3), then all particles are assumed to have initial spin 
`s0`. If `s0` is a matrix (of size `(n_particles, 3)`), then the i-th particle will have
initial spin `s0[i,:]`.

See `track_spin!` for the in-place version for a pre-allocated output tensor `s`.
"""
function track_spin(q::AbstractArray, s0::AbstractVecOrMat)
  s = similar(q, (size(q, 1), 3, size(q)[3:end]...))
  track_spin!(s, q, s0)
  return s
end

# (the q of a TrackResult)
function track_spin(q::TrackColumn, s0::AbstractVecOrMat)
  q.name === :q && isnothing(q.data) || error("track_spin needs the quaternions `res.q`")
  return TrackColumn(q.res, :spin, track_spin(_track_raw(getfield(q.res, :data), :q)[1], s0))
end

"""
    track_spin!(s::AbstractArray, q::AbstractArray,  s0::AbstractVecOrMat)

In-place version of `track_spin`. See the documentation for `track_spin`.
"""
function track_spin!(s::AbstractArray, q::AbstractArray,  s0::AbstractVecOrMat)
  @assert size(s, 1) == size(q, 1) "Number of rows (particles) in spin output array s and quaternion input array q not equal"
  @assert size(s, 2) == 3 "Number of columns of spin output array s not equal to 3"
  @assert size(q, 2) == 4 "Number of columns of quaternion input array q not equal to 4"
  @assert size(s)[3:end] == size(q)[3:end] "Sizes of s and q after the 2nd dimension not equal"
  if s0 isa AbstractMatrix
    @assert size(s, 1) == size(q, 1) "Number of rows (particles) in spin input array s0 and quaternion input array q not equal"
  else
    @assert length(s0) == 3 "Length of spin input vector not equal to 3"
  end
  device = KA.get_backend(s)
  # The locations and saved turns are one dimension for the kernel
  _rotate_spins!(device)(reshape(s, size(s, 1), 3, :), reshape(q, size(q, 1), 4, :), s0;
                         ndrange=size(s, 1))
  KA.synchronize(device)
  return s
end

@kernel function _rotate_spins!(s, @Const(q), @Const(s0))
  i = @index(Global)
  _rotate_spin!(i, s, q, s0)
end

# On Windows with Julia < 1.11 and KernelAbstractions 0.10, CPU arrays use a loop instead of the
# kernel (see closed_orbit.jl: a CPU kernel compiled in a precompile workload breaks the image)
@static if Sys.iswindows() && VERSION < v"1.11" && pkgversion(KA) >= v"0.10"
  function _rotate_spins_cpu!(s, q, s0; ndrange)
    for i in 1:ndrange
      _rotate_spin!(i, s, q, s0)
    end
    return nothing
  end
  _rotate_spins!(::KA.CPU) = _rotate_spins_cpu!
end

@inline function _rotate_spin!(i, s, q, s0)
  @inbounds begin

    # Set initial condition:
    if s0 isa AbstractVector
      sx = s0[1]
      sy = s0[2]
      sz = s0[3]
    else
      sx = s0[i,1]
      sy = s0[i,2]
      sz = s0[i,3]
    end

    # Now the loop:
    for j in 1:size(q, 3)
      a  = q[i,1,j]; bx = q[i,2,j]; by = q[i,3,j]; bz = q[i,4,j]

      tx = 2 * (by*sz - bz*sy)
      ty = 2 * (bz*sx - bx*sz)
      tz = 2 * (bx*sy - by*sx)

      s[i,1,j] = sx + a*tx + (by*tz - bz*ty)
      s[i,2,j] = sy + a*ty + (bz*tx - bx*tz)
      s[i,3,j] = sz + a*tz + (bx*ty - by*tx)
    end
  end
end

