@kwdef struct TrackingConfig
  n_turns::Int                          = 1
  save_every_n_turns::Int               = 1
  scalar_params::Bool                   = false
  ramp_particle_energy_without_rf::Bool = false
  verbose::Bool                         = false
  groupsize::Int                        = 0 # autoselect
  multithread_threshold::Int            = (Threads.nthreads() > 1 ? 1750*Threads.nthreads() : typemax(Int))
  use_KA::Bool                          = true
  use_explicit_SIMD::Bool               = !use_KA
end


struct TrackingResult{S,V,Q}
  config::TrackingConfig
  state::S
  v::V
  q::Q
  bunch::Bunch{<:Any,<:Any,<:BeamTracking.Coords{S,V,Q}}
end

function track(
    bl::Beamline;

    # User can either specify kwargs to construct a bunch:
    v0::AbstractMatrix,
    spin::Bool=false,
    q0::Union{AbstractMatrix,Nothing}=nothing,
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
    multithread_threshold           = config.multithread_threshold,
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
    multithread_threshold,
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
  multithread_threshold           = config.multithread_threshold
  use_KA                          = config.use_KA
  use_explicit_SIMD               = config.use_explicit_SIMD

  n_particles = size(v, 1)

  n_data_pts = div(n_turns, save_every_n_turns) + 1 # + 1 for initial 

  state_data = similar(state, n_particles, 1, n_data_pts)
  v_data = similar(v, n_particles, 6, n_data_pts)
  q_data = isnothing(q) ? nothing : similar(q, n_particles, 4, n_data_pts)
  
  state_data[:,:,1] .= state
  v_data[:,:,1] .= v
  if !isnothing(q)
    q_data[:,:,1] .= q
  end

  for i in 1:n_turns
    track!(bunch, bl; scalar_params, ramp_particle_energy_without_rf, groupsize, use_KA, use_explicit_SIMD, multithread_threshold)
    if mod(i, save_every_n_turns) == 0
      idx = div(i,save_every_n_turns)+1
      state_data[:,:,idx] .= state
      v_data[:,:,idx] .= v
      if !isnothing(q)
        q_data[:,:,1] .= q
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
  return TrackingResult(config, state, v, q, bunch)
end