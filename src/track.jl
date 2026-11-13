"""
    struct TrackingConfig

Defines configuration settings for a particles tracking run.

## Properties
- `n_turns::Int`: Number of turns, defaults to `1`
- `save_every_n_turns::Int`: Multiple of turn number which specifies the turns tracking 
    data is saved, defaults to `1`
- `in_body_coordinates::Bool`: If `true`, particles saved inside misaligned elements are not
    transformed back to the reference curve coordinates. Default is `false`
- `scalar_params::Bool`: If `true`, only uses the primal values of any `LatElement` 
    properties set as `GTPSA.TPS`, `ForwardDiff.Dual`, or `ReverseDiff.TrackedReal` 
    (ignores derivatives part, useful for closed orbit finding prior to parametric 
    normal form). Default is `false`
- `rf_on::Bool`: If `false`, then any `RFParams` in any `LatElement`s are ignored for the 
    entirety of the tracking. Default is `true`
- `ramp_particle_energy_without_rf::Bool`: If `true` when ramping (`Beamline`'s reference 
    energy is a `TimeFunction`), then particle energies will be artificially ramped with 
    the reference energy. Default is `false`
- `ramp_update_each_particle::Bool`: If `true` when ramping, then the z-offset/time-offset 
    of each individual particle at the element entrances is taken into account when 
    tracking particles through the time-dependent element parameters. Usually only needed 
    if ramping is fast compared to the time scale of bunch passage through elements. 
    Default is `false`.
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
@kwdef struct TrackingConfig
  n_turns::Int                          = 1
  save_every_n_turns::Int               = 1
  in_body_coordinates::Bool             = false
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

function Base.show(io::IO, config::TrackingConfig)
  fields = fieldnames(typeof(config))
  width = maximum(length, String.(fields))
  println(io, nameof(typeof(config)))
  for field in fields
    println(io, " ", rpad(String(field), width), " = ", repr(getproperty(config, field)))
  end
  return
end

"""
    struct TrackingResult{L,S0,V0,Q0,S,V,Q}

Contains all information for a given particle tracking run.

## Properties
- `config`: A `TrackingConfig` struct containing configuration settings
- `elapsed_time`: The tracking execution time in seconds 
- `s`: The s of each location where the particles are saved (see `at` in `track`), in the order
    of the `location` index of `state`, `v` and `q`
- `index`: The beamline index of the element of each location (`-1` for the end of the
    beamline)
- `name`: The name of the element of each location
- `kind`: The kind of the element of each location
- `state0`: The initial particle states
- `v0`: The initial particle phase space coordinates
- `q0`: If spin tracking was enabled, the initial particle quaternions
- `state`: A tensor of particle states at each location and saved turn
- `v`: A tensor of particle phase space coordinates at each location and saved turn
- `q`: If spin tracking was enabled, a tensor of particle quaternions at each location and
    saved turn
- `bunch`: The `Bunch` at the end of tracking
"""
struct TrackingResult{L,S0,V0,Q0,S,V,Q}
  config::TrackingConfig
  elapsed_time::Float64 # in seconds
  s::L
  index::Vector{Int}
  name::Vector{String}
  kind::Vector{String}
  state0::S0
  v0::V0
  q0::Q0
  state::S
  v::V
  q::Q
  bunch::Bunch
end

function Base.show(io::IO, res::TrackingResult)
  println(io, typeof(res), ":")
  config=res.config
  fields = fieldnames(typeof(config))
  width = maximum(length, String.(fields))
  println(io, "config:")
  for field in fields
    println(io, " ", rpad(String(field), width), " = ", repr(getproperty(config, field)))
  end
  println(io)
  println(io, "elapsed_time = ", res.elapsed_time, " seconds")
  println(io)
  if res.config.save_every_n_turns == 1
    turnstr = "turn"
  else
    turnstr = "Int((turn-1)/$(res.config.save_every_n_turns))+1"
  end
  n_particles = size(res.state, 1)
  println(io, "state0, v0", isnothing(res.q0) ? "" : ", q0", ":\tInitial particles")
  n_locations = size(res.state, 2)
  n_saved_turns = size(res.state, 3)
  println(io, "s, index, name, kind:\tLocations (s, beamline index, name, kind of the element)")
  println(io, "state[1:$n_particles, 1:$n_locations, 1:$n_saved_turns]:\tStates indexable as\t\t ",
          "state[particle, location, $(turnstr)]")
  println(io, "v[1:$n_particles, 1:6, 1:$n_locations, 1:$n_saved_turns]:",
          "\tPhase space coords indexable as\t v[particle, coordinate, location, $(turnstr)]")
  if !isnothing(res.q)
    println(io, "q[1:$n_particles, 1:4, 1:$n_locations, 1:$n_saved_turns]:",
            "\tQuaternions indexable as\t q[particle, quat_coordinate, location, $(turnstr)]")
  end
  println(io, "")
  println(io, "bunch:\tBunch at the end of tracking")
  return
end

"""
    track(bl::Beamline; kwargs...) -> TrackingResult

All-encompassing function to track particles through the beamline `bl` using the configured 
settings specified by the keyword arguments `kwargs`.

## Keyword Arguments
- `v0`: A matrix of size `(n_particles, 6)` storing the initial particle phase space coordinates
- `spin`: If `true`, spin tracking will be enabled with identity quaternions as initial quaternions
- `q0`: A matrix of size `(n_particles, 4)` storing the initial particle spin quaternions if spin tracking
- `weight`: Optional vector of length `(n_particles)` specifying macroparticle weights per particle, default 
    is `nothing` corresponding to uniform weights for all particles
- `bunch`: Instead of specifying `v0`, `spin`, `q0`, and `weight`, an initial `Bunch` may be explicitly provided
- `start::Union{Integer,LatElement}`: The element (or its beamline index) at whose beginning the
    tracking starts. Default is `1`
- `at::Union{Nothing,Colon,AbstractVector}`: Where the particles are saved, as in `twiss`: a
    vector containing `LatElement`s, element beamline indexes, and/or tuples of s-ranges. An
    element gives its beginning, an s-range the integrator steps inside it (and the end of the
    beamline if it contains it), and `:` every integrator step. Default is `nothing`, only the
    end of the beamline. The locations are ordered in s (from the beginning of the beamline).
    In the first turn, the locations before `start` are not reached and are `NaN` (`state`
    `STATE_PREBORN`)
- `n_turns::Int`: Number of turns, each ending at the end of the beamline (the first turn
    begins at `start`), defaults to `1`
- `save_every_n_turns::Int`: The turns `1`, `1+save_every_n_turns`, `1+2*save_every_n_turns`,
    ... are saved, defaults to `1`
- `in_body_coordinates::Bool`: If `true`, particles saved inside misaligned elements are not
    transformed back to the reference curve coordinates. Default is `false`
- `scalar_params::Bool`: If `true`, only uses the primal values of any `LatElement` 
    properties set as `GTPSA.TPS`, `ForwardDiff.Dual`, or `ReverseDiff.TrackedReal` 
    (ignores derivatives part, useful for closed orbit finding prior to parametric 
    normal form). Default is `false`
- `rf_on::Bool`: If `false`, then any `RFParams` in any `LatElement`s are ignored for the 
    entirety of the tracking. Default is `true`
- `ramp_particle_energy_without_rf::Bool`: If `true` when ramping (`Beamline`'s reference 
    energy is a `TimeFunction`), then particle energies will be artificially ramped with 
    the reference energy. Default is `false`
- `ramp_update_each_particle::Bool`: If `true` when ramping, then the z-offset/time-offset 
    of each individual particle at the element entrances is taken into account when 
    tracking particles through the time-dependent element parameters. Usually only needed 
    if ramping is fast compared to the time scale of bunch passage through elements. 
    Default is `false`.
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
    `scalar_params`,
    `ramp_particle_energy_without_rf`, `ramp_update_each_particle`, `verbose`, `groupsize`, 
    `use_cpu_multithreading`, `use_KA`, and `use_explicit_SIMD`, a single `TrackingConfig` 
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

    # Or explicitly provide a Bunch:
    bunch::Bunch=Bunch(; 
      v=(v = similar(v0); v .= copy.(v0); v), spin=spin,
      q=(!isnothing(q0) ? copy.(q0) : nothing), 
      weight=(!isnothing(weight) ? copy.(weight) : nothing), 
      species=bl.species_ref, 
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

    config=TrackingConfig(use_KA=!(KA.get_backend(bunch.v) isa KA.CPU)),
    
    # Tracking customization kwargs:
    n_turns                         = config.n_turns,
    save_every_n_turns              = config.save_every_n_turns,
    in_body_coordinates             = config.in_body_coordinates,
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

  newconfig = TrackingConfig(
    n_turns,
    save_every_n_turns,
    in_body_coordinates,
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
  if start isa LatElement
    # (as in _twiss_assemble_locations, the element or the element it was copied from)
    ks = findall(bl.line) do ele
      ele == start || (haskey(getfield(ele, :pdict), InheritParams) ?
                       start == getfield(ele, :pdict)[InheritParams].parent : false)
    end
    length(ks) == 1 || error("The start element must be in the beamline exactly once")
    start = only(ks)
  end
  1 <= start <= length(bl.line) || error("start must be a beamline index of an element")
  # Locations where the particles are saved, as in twiss (the first entry is the beginning of the
  # beamline unless steps of the first element are in an s-range, the last the end of the beamline)
  # (by default only the end of the beamline)
  s, names, kinds, idxs, step_save, include_start, include_end = _twiss_assemble_locations(bl,
                                                                   isnothing(at) ? [] : at)
  include_end |= isnothing(at)
  sel = (include_start ? 1 : 2):(include_end ? length(s) : length(s)-1)
  loc_s, loc_names, loc_kinds, loc_idxs = s[sel], names[sel], kinds[sel], idxs[sel]
  loc_steps = step_save[sel]
  n_steps_turn = last(step_save)
  # Function barrier for Bunch and groupsize (needs special handling for KA)
  locations = (loc_s, loc_idxs, loc_names, loc_kinds)
  return @noinline _track(bl, bunch, newconfig, groupsize == 0 ? nothing : groupsize, start,
                          locations, loc_steps, n_steps_turn)
end

# loc_steps are the steps (tracking callbacks) of the locations in a turn, counted from the
# beginning of the beamline (step 0), and n_steps_turn is the number of steps of a turn (the end of
# the beamline). The locations are saved by a callback during the turns, as in twiss
function _track(bl, bunch, config, groupsize, start, locations, loc_steps, n_steps_turn)
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

  # Saved turns: 1, 1+save_every_n_turns, ...
  n_data_pts = div(n_turns-1, save_every_n_turns) + 1
  slot_of_turn(i) = (i <= n_turns && mod(i-1, save_every_n_turns) == 0) ?
                    div(i-1, save_every_n_turns) + 1 : 0

  state0 = copy(state)
  v0 = copy(v)
  q0 = isnothing(q) ? nothing : copy(q)
  state_data = similar(state, n_particles, n_locations, n_data_pts)
  v_data = similar(v, n_particles, 6, n_locations, n_data_pts)
  q_data = isnothing(q) ? nothing : similar(q, n_particles, 4, n_locations, n_data_pts)

  # The kernel chains of the elements are built once for all turns (not included in the time t)
  chains = BTBL.prepare(bunch, bl; scalar_params, rf_on, ramp_particle_energy_without_rf,
                        ramp_update_each_particle, batch_start)

  if start > 1
    # The first turn begins with the first chain of the start element, after step_start steps of
    # the turn. The locations before are not reached in the first turn
    chain_steps, chain_ele = _chain_steps(bl)
    length(chain_steps) == length(chains) ||
      error("The number of chains differs from that of the slices")
    first_chain = findfirst(==(start), chain_ele)
    step_start = sum(view(chain_steps, 1:first_chain-1))
    for (l, step) in enumerate(loc_steps)
      if step < step_start
        state_data[:,l,1] .= BeamTracking.STATE_PREBORN
        v_data[:,:,l,1] .= convert(eltype(v_data), NaN)
        if !isnothing(q)
          q_data[:,:,l,1] .= convert(eltype(q_data), NaN)
        end
      end
    end
  else
    first_chain = 1
    step_start = 0
  end
  # The start holds the initial particles in the first turn (there is no callback before it)
  i_start = findfirst(==(step_start), loc_steps)
  if !isnothing(i_start)
    state_data[:,i_start,1] .= state
    v_data[:,:,i_start,1] .= v
    if !isnothing(q)
      q_data[:,:,i_start,1] .= q
    end
  end

  # The callback finds the location from its step in the turn (loc_of_step[cur_step] is the
  # location, or 0). At the last step (the end of the beamline), it also saves the beginning of the
  # beamline (location i_begin, or 0) of the next turn
  loc_of_step = zeros(Int, n_steps_turn)
  for (l, step) in enumerate(loc_steps)
    if step > 0
      loc_of_step[step] = l
    end
  end
  i_begin = something(findfirst(==(0), loc_steps), 0)
  with_cb = any(!=(0), loc_of_step) || i_begin != 0
  if with_cb
    slots = similar(v, Int, 2) # slots of the turn and of the next turn in the saved data (or 0)
    _loc_of_step = similar(v, Int, n_steps_turn)
    copyto!(_loc_of_step, loc_of_step)
    cb = _track_make_callback(slots, _loc_of_step, i_begin, in_body_coordinates, v_data, q_data,
                              state_data)
    # Tracked with the callback appended to those of the bunch
    coords = BeamTracking.Coords(state, v, q, bunch.coords.weight, (bunch.coords.callbacks..., cb))
    b = Bunch(bunch.species, bunch.p_over_q_ref, bunch.t_ref, coords)
  else
    b = bunch
  end

  t = @elapsed for i in 1:n_turns
    if with_cb
      copyto!(slots, [slot_of_turn(i), slot_of_turn(i+1)])
    end
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
  if b !== bunch
    bunch.species = b.species
    bunch.p_over_q_ref = b.p_over_q_ref
    bunch.t_ref = b.t_ref
    bunch.s = b.s
    bunch.step = b.step
  end
  return TrackingResult(config, t, locations..., state0, v0, q0, state_data, v_data, q_data, bunch)
end

# The number of steps (tracking callbacks) of each chain of bl, i.e. n_steps for each of the
# n_slices of each element (as counted in _twiss_assemble_locations), and the beamline index of the
# element of each chain
function _chain_steps(bl)
  chain_steps = Int[]
  chain_ele = Int[]
  for (k, ele) in enumerate(bl.line)
    up = (ele.UniversalParams)::UniversalParams
    if up.L isa DefExpr
      up = Beamlines.deval(up, bl.context)
    end
    for _ in 1:up.n_slices
      push!(chain_steps, up.n_steps)
      push!(chain_ele, k)
    end
  end
  return chain_steps, chain_ele
end

function _track_make_callback(_slots, _loc_of_step, _i_begin, _in_body_coordinates, _v_data,
                              _q_data, _state_data)
  let slots=_slots, loc_of_step=_loc_of_step, i_begin=_i_begin,
    in_body_coordinates=_in_body_coordinates, v_data=_v_data, q_data=_q_data, state_data=_state_data
    return (i, coords, cur_s, cur_t_ref, step, cur_beta_gamma_ref, last_ds_step, last_g,
            transforms_out!, transforms_in!) -> begin
      @inbounds begin
        n = length(loc_of_step)
        j = slots[1]
        if j != 0 && step <= n
          l = loc_of_step[step]
          if l != 0
            _track_save!(i, coords, transforms_out!, in_body_coordinates, v_data, q_data, state_data,
                         l, j)
          end
        end
        if step == n && i_begin != 0
          j_next = slots[2]
          if j_next != 0
            _track_save!(i, coords, transforms_out!, in_body_coordinates, v_data, q_data, state_data,
                         i_begin, j_next)
          end
        end
      end
      return nothing
    end
  end
end

@inline function _track_save!(i, coords, transforms_out!, in_body_coordinates, v_data, q_data,
                              state_data, l, j)
  @inbounds begin
    if in_body_coordinates
      _track_copy!(i, coords, v_data, q_data, state_data, l, j)
    else
      # Saved in the lab coordinates, and restored after (saving does not change the tracking)
      v = coords.v
      q = coords.q
      vs = ntuple(k -> v[i,k], Val{6}())
      qs = isnothing(q) ? nothing : ntuple(k -> q[i,k], Val{4}())
      ss = coords.state[i]
      transforms_out!(i, coords)
      _track_copy!(i, coords, v_data, q_data, state_data, l, j)
      for k in 1:6
        v[i,k] = vs[k]
      end
      if !isnothing(q)
        for k in 1:4
          q[i,k] = qs[k]
        end
      end
      coords.state[i] = ss
    end
  end
  return nothing
end

@inline function _track_copy!(i, coords, v_data, q_data, state_data, l, j)
  @inbounds begin
    state_data[i,l,j] = coords.state[i]
    for k in 1:6
      v_data[i,k,l,j] = coords.v[i,k]
    end
    if !isnothing(q_data)
      for k in 1:4
        q_data[i,k,l,j] = coords.q[i,k]
      end
    end
  end
  return nothing
end

"""
    track_spin(q::AbstractArray, s0::AbstractVecOrMat)

Given initial spin(s) `s0` and the quaternion output tensor `q` from `track`, 
returns a tensor `s` of size `(n_particles, 3, n_locations, n_saved_turns)` of the particles' 
spins at each location and stored turn (any further dimensions of `q` after the second are 
kept as they are).

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

