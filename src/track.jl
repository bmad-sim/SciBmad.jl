"""
    struct TrackingConfig

Defines configuration settings for a particles tracking run.

## Properties
- `n_turns::Int`: Number of turns, defaults to `1`
- `save_every_n_turns::Int`: Multiple of turn number which specifies the turns tracking 
    data is saved, defaults to `1`
- `scalar_params::Bool`: If `true`, only uses the primal values of any `LineElement` 
    properties set as `GTPSA.TPS`, `ForwardDiff.Dual`, or `ReverseDiff.TrackedReal` 
    (ignores derivatives part, useful for closed orbit finding prior to parametric 
    normal form). Default is `false`
- `ramp_particle_energy_without_rf::Bool`: If `true` when ramping (`Beamline`'s reference 
    energy is a `TimeFunction`), then particle energies will be artificially ramped with 
    the reference energy. Default is `false`
- `ramp_update_each_particle::Bool`: If `true` when ramping, then the z-offset/time-offset 
    of each individual particle at the element entrances is taken into account when 
    tracking particles through the time-dependent element parameters. Usually only needed 
    if ramping is fast compared to the time scale of bunch passage through elements. 
    Default is `false`.
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
  scalar_params::Bool                   = false
  ramp_particle_energy_without_rf::Bool = false
  ramp_update_each_particle::Bool       = false
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
    struct TrackingResult{S,V,Q}

Contains all information for a given particle tracking run.

## Properties
- `config`: A `TrackingConfig` struct containing configuration settings
- `elapsed_time`: The tracking execution time in minutes 
- `state`: A matrix of particle states at each saved turn
- `v`: A tensor of particle phase space coordinates at each saved turn
- `q`: If spin tracking was enabled, a tensor of particle quaternions at each saved turn
- `bunch`: The `Bunch` at the end of tracking
"""
struct TrackingResult{S,V,Q}
  config::TrackingConfig
  elapsed_time::Float64 # in minutes
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
    turnstr = "Int(turn/$(res.config.save_every_n_turns))"
  end
  n_particles = size(res.state, 1)
  n_saved_turns = size(res.state, 2)
  println(io, "state[1:$n_particles, 1:$n_saved_turns]:\tStates indexable as\t\t state[particle, $(turnstr)+1]")
  println(io, "v[1:$n_particles, 1:6, 1:$n_saved_turns]:\tPhase space coords indexable as\t v[particle, coordinate, $(turnstr)+1]")
  if !isnothing(res.q)
    println(io, "q[1:$n_particles, 1:4, 1:$n_saved_turns]:\tQuaternions indexable as\t q[particle, quat_coordinate, $(turnstr)+1]")
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
- `n_turns::Int`: Number of turns, defaults to `1`
- `save_every_n_turns::Int`: Multiple of turn number which specifies the turns tracking 
    data is saved, defaults to `1`
- `scalar_params::Bool`: If `true`, only uses the primal values of any `LineElement` 
    properties set as `GTPSA.TPS`, `ForwardDiff.Dual`, or `ReverseDiff.TrackedReal` 
    (ignores derivatives part, useful for closed orbit finding prior to parametric 
    normal form). Default is `false`
- `ramp_particle_energy_without_rf::Bool`: If `true` when ramping (`Beamline`'s reference 
    energy is a `TimeFunction`), then particle energies will be artificially ramped with 
    the reference energy. Default is `false`
- `ramp_update_each_particle::Bool`: If `true` when ramping, then the z-offset/time-offset 
    of each individual particle at the element entrances is taken into account when 
    tracking particles through the time-dependent element parameters. Usually only needed 
    if ramping is fast compared to the time scale of bunch passage through elements. 
    Default is `false`.
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
- `config`: Instead of specifying `n_turns`, `save_every_n_turns`, `scalar_params`, 
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
    weight::Union{AbstractMatrix,Nothing}=nothing,

    # Or explicitly provide a Bunch:
    bunch::Bunch=Bunch(; 
      v=copy.(v0), spin=spin, 
      q=(!isnothing(q0) ? copy.(q0) : nothing), 
      weight=(!isnothing(weight) ? copy.(weight) : nothing), 
      species=bl.species_ref, 
      p_over_q_ref=(_p_over_q_ref = bl.p_over_q_ref; _p_over_q_ref isa TimeDependentParam ? _p_over_q_ref(0) : _p_over_q_ref)
    ),

    config=TrackingConfig(use_KA=!(KA.get_backend(bunch.v) isa KA.CPU)),
    
    # Tracking customization kwargs:
    n_turns                         = config.n_turns,
    save_every_n_turns              = config.save_every_n_turns,
    scalar_params                   = config.scalar_params,
    ramp_particle_energy_without_rf = config.ramp_particle_energy_without_rf,
    ramp_update_each_particle       = config.ramp_update_each_particle,
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
    scalar_params,
    ramp_particle_energy_without_rf,
    ramp_update_each_particle,
    verbose,
    groupsize,
    use_cpu_multithreading,
    use_KA,
    use_explicit_SIMD,
  )
  # Function barrier for Bunch and groupsize (needs special handling for KA)
  return @noinline _track(bl, bunch, newconfig, groupsize == 0 ? nothing : groupsize)
end

function _track(bl, bunch, config, groupsize)
  state = bunch.coords.state
  v     = bunch.coords.v
  q     = bunch.coords.q
  n_turns                         = config.n_turns
  save_every_n_turns              = config.save_every_n_turns
  scalar_params                   = config.scalar_params
  ramp_particle_energy_without_rf = config.ramp_particle_energy_without_rf
  ramp_update_each_particle       = config.ramp_update_each_particle
  verbose                         = config.verbose
  use_cpu_multithreading          = config.use_cpu_multithreading
  use_KA                          = config.use_KA
  use_explicit_SIMD               = config.use_explicit_SIMD

  n_particles = size(v, 1)

  n_data_pts = div(n_turns, save_every_n_turns) + 1 # + 1 for initial 

  state_data = similar(state, n_particles, n_data_pts)
  v_data = similar(v, n_particles, 6, n_data_pts)
  q_data = isnothing(q) ? nothing : similar(q, n_particles, 4, n_data_pts)
  
  state_data[:,:,1] .= state
  v_data[:,:,1] .= v
  if !isnothing(q)
    q_data[:,:,1] .= q
  end

  t = @elapsed for i in 1:n_turns
    track!(bunch, bl; scalar_params, ramp_particle_energy_without_rf, ramp_update_each_particle, groupsize, use_KA, use_explicit_SIMD, use_cpu_multithreading)
    if mod(i, save_every_n_turns) == 0
      idx = div(i,save_every_n_turns)+1
      state_data[:,idx] .= state
      v_data[:,:,idx] .= v
      if !isnothing(q)
        q_data[:,:,idx] .= q
      end
    end
    if verbose
      print("\rFinished turn $i out of $n_turns")
      flush(stdout) 
    end
  end
  if verbose
    println()
  end
  return TrackingResult(config, t, state_data, v_data, q_data, bunch)
end

"""
    track_spin(s0::AbstractVecOrMat, q::AbstractArray{<:Any,3})

Given initial spin(s) `s0` and the quaternion output tensor `q` from `track`, 
returns a tensor `s` of size `(n_particles, 3, n_saved_turns)` of the particles' 
spins at each stored turn.

If `s0` is a vector (of length 3), then all particles are assumed to have initial spin 
`s0`. If `s0` is a matrix (of size `(n_particles, 3)`), then the i-th particle will have
initial spin `s0[i,:]`.

See `track_spin!` for the in-place version for a pre-allocated output tensor `s`.
"""
function track_spin(q::AbstractArray{<:Any,3}, s0::AbstractVecOrMat)
  s = similar(q, (size(q, 1), 3, size(q,3)))
  track_spin!(s, q, s0)
  return s
end

"""
    track_spin!(s::AbstractArray{<:Any,3}, q::AbstractArray{<:Any,3},  s0::AbstractVecOrMat)

In-place version of `track_spin`. See the documentation for `track_spin`.
"""
function track_spin!(s::AbstractArray{<:Any,3}, q::AbstractArray{<:Any,3},  s0::AbstractVecOrMat)
  @assert size(s, 1) == size(q, 1) "Number of rows (particles) in spin output array s and quaternion input array q not equal"
  @assert size(s, 2) == 3 "Number of columns of spin output array s not equal to 3"
  @assert size(q, 2) == 4 "Number of columns of quaternion input array q not equal to 4"
  @assert size(s, 3) == size(q, 3) "Size of 3rd dimension (number of saved turns) of spin output array s and quaternion input array q not equal"
  if s0 isa AbstractMatrix
    @assert size(s, 1) == size(q, 1) "Number of rows (particles) in spin input array s0 and quaternion input array q not equal"
  else
    @assert length(s0) == 3 "Length of spin input vector not equal to 3"
  end
  device = KA.get_backend(s)
  _rotate_spins!(device)(s, q, s0; ndrange=size(s, 1))
  KA.synchronize(device)
  return s
end

@kernel function _rotate_spins!(s, @Const(q), @Const(s0))
  i = @index(Global)
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

