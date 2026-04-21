@kwdef struct TrackingConfig
  n_turns::Int                          = 1
  save_every_n_turns::Int               = 1
  scalar_params::Bool                   = false
  ramp_particle_energy_without_rf::Bool = false
  verbose::Bool                         = false
  groupsize::Int                        = 0 # autoselect
  use_cpu_multithreading::Bool          = (Threads.nthreads() > 1 ? true : false)
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
    turnstr = "Int(turn/$(res.config.save_every_n_turns))+1"
  end
  println(io, "state:\tStates indexable as state[particle, $(turnstr)]")
  println(io, "v:\tPhase space coordinates indexable as v[particle, coordinate, $(turnstr)]")
  if !isnothing(res.q)
    println(io, "q:\tQuaternions indexable as q[particle, quat_coordinate, $(turnstr)]")
  end
  println(io, "")
  println(io, "bunch:\tBunch at the end of tracking")
  return
end


function track(
    bl::Beamline;

    # User can either specify kwargs to construct a bunch:
    v0::AbstractMatrix,
    spin::Bool=false,
    q0::Union{AbstractMatrix,Nothing}=spin ? (q = similar(v0, (size(v0, 1), 4)); q .= 0; q[:,1] .= 1; q) : nothing,
    weight::Union{AbstractMatrix,Nothing}=nothing,

    # Or explicitly provide a Bunch:
    bunch::Bunch=Bunch(; 
      v=v0, spin=spin, q=q0, weight=weight, p_over_q_ref=bl.p_over_q_ref, species=bl.species_ref
    ),

    config=TrackingConfig(use_KA=!(KA.get_backend(bunch.v) isa KA.CPU)),
    
    # Tracking customization kwargs:
    n_turns                         = config.n_turns,
    save_every_n_turns              = config.save_every_n_turns,
    scalar_params                   = config.scalar_params,
    ramp_particle_energy_without_rf = config.ramp_particle_energy_without_rf,
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
    track!(bunch, bl; scalar_params, ramp_particle_energy_without_rf, groupsize, use_KA, use_explicit_SIMD, use_cpu_multithreading)
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
"""
function track_spin(q::AbstractArray{<:Any,3}, s0::AbstractVecOrMat)
  s = similar(q, (size(q, 1), 3, size(q,3)))
  track_spin!(s, q, s0)
  return s
end

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

    s[i,1,1] = sx
    s[i,2,1] = sy
    s[i,3,1] = sz

    # Now the loop:
    for j in 2:size(q, 3)
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

