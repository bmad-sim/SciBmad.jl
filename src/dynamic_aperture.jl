"""
    dynamic_aperture(ring::Beamline; kwargs...) -> NTuple{2,Vector{Float64}}

Computes the acceptance of the ring by pushing a polar grid in `x/sig_x` and `y/sig_y` 
space, for each of the provided `deltas`. Returns a tuple of two vectors defining the 
first particle loss along the radius for a given angle on the polar grid, where the 
first index corresponds to the line position in `x/sig_x` or `y/sig_y` space, and the 
second index corresponds to that in `deltas`.

## Required Keyword arguments
- `n_r::Int`: Number of radial points to sample per angle on the polar grid
- `n_theta::Int`: Number of angles to sample on the polar grid
- `deltas::AbstractArray`: Array of each δ to compute the transverse acceptance
- `max_sig_x::Real`: Maximum sigma-x of the polar grid, in units of sigma (e.g. 
    `max_sigma_x=20` to check up to 20 sigma)
- `max_sig_y::Real`: Maximum sigma-y of the polar grid, in units of sigma (e.g. 
    `max_sigma_y=20` to check up to 20 sigma)
- `emit_1::Real`: Horizontal-like emittance to assume
- `emit_2::Real`: Vertical-like emittance to assume
- `n_turns::Int`: Number of turns to track

## Optional Keyword Arguments
- `backend`: A CPU/GPU device backend to use, e.g. `CPU()`, or `CUDA.CUDABackend()` to 
    use the GPU. Default is `CPU()`
- `coordinates_number_type::Type`: The type of the phase space coordinates to use in 
    tracking. Default is `Float64`
- `delta_dependent_orbits::Bool`: If `true`, then for each δ in `deltas`, the 
    computed acceptance line will be centered around that δ-dependent closed orbit 
    (computed with `rf_on=false`). Else, all acceptance lines are centered around the 
    6D closed orbit. Default is `true` if coasting beam is detected, `false` if 
    longitudinal motion is detected.
- `theta_lims`: A tuple specifying the range of thetas to scan on the polar grid. Default 
    is `(0, pi)`
- `sig_pz::Real`: sigma-δ of the beam to assume, default is 0
- `emit_3::Real`: Longitudinal-like emittance to assume
- `output_file`: If provided, then a file containing the initial particle coordinates (w.r.t. 
    the closed orbit(s)) as rows, with the last column specifying the particle state at the 
    end of tracking, will be written to the file. Default is `nothing`.
- `track_kwargs...`: Any of the settings in `TrackingConfig` except for `n_turns` and 
    `save_every_n_turns` may be provided to configure the tracking.
"""
function dynamic_aperture(
    bl::Beamline;

    # Required kwargs:
    n_r::Int,
    n_theta::Int,
    deltas::AbstractArray,
    max_sig_x::Real,
    max_sig_y::Real,
    emit_1::Real,
    emit_2::Real,
    n_turns::Int,

    # Optional kwargs:
    backend=KA.CPU(),
    coordinates_number_type::Type=Float64, 
    sig_pz::Real=0,
    emit_3::Real=0,
    delta_dependent_orbits::Bool=coast_check(bl),
    output_file=nothing,
    theta_lims=(0, pi),
    track_kwargs... # Get passed to track!
  )
  Base.require_one_based_indexing(deltas)
  if delta_dependent_orbits && emit_3 != 0
    error("delta_dependent_orbits = true, but a nonzero emit_3 was provided. Instead specify sig_pz")
  elseif !delta_dependent_orbits && sig_pz != 0
    error("delta_dependent_orbits = true, but a nonzero emit_3 was provided. Instead specify sig_pz")
  end
  
  n_deltas = length(deltas)
  co = zeros(n_deltas, 6)
  co[:,6] .= deltas
  
  if delta_dependent_orbits
    tw = twiss(bl, at=[1], cols=["E1", "E2", "dx", "dy"], rf_on=false)
    if (:q3 in propertynames(tw))
      error("
        To compute delta_dependent_orbits, the beam must be coasting. We tried turning off 
        all cavities, but still did not detect coasting beam. Please turn off any elements 
        that cause longitudinal motion.
      ")
    end

    # Now compute sigmas at first element, just first order:
    sig_x = tw.E1[1][1,1]*emit_1 + tw.E2[1][1,1]*emit_2 
    sig_y = tw.E1[1][3,3]*emit_1 + tw.E2[1][3,3]*emit_2 
    sig_x += (tw.dx[1]*sig_pz)^2
    sig_y += (tw.dy[1]*sig_pz)^2
    sig_x = sqrt(sig_x)
    sig_y = sqrt(sig_y)

    # Compute delta-dependent closed orbits (with RF off)
    sol = find_closed_orbit(bl; v0=co, batch=Val{true}(), coasting_beam=true, rf_on=false)
    if any(sol.sol.retcode .!= BatchSolve.RETCODE_SUCCESS) # If any failed:
      error(
        """
        Unable for find delta-dependent closed orbits (with RF off) for deltas = $(deltas[findall(sol.sol.retcode .!= 0x0)]).
        Please remove these deltas from the input.
        """
      )
    end
  else
    tw = twiss(bl, at=[1], cols=["E1", "E2", "E3"])
    # Now compute sigmas at first element, just first order:
    sig_x = tw.E1[1][1,1]*emit_1 + tw.E2[1][1,1]*emit_2 + tw.E3[1][1,1]*emit_3
    sig_y = tw.E1[1][3,3]*emit_1 + tw.E2[1][3,3]*emit_2 + tw.E3[1][3,3]*emit_3
    sig_x = sqrt(sig_x)
    sig_y = sqrt(sig_y)
    sol = find_closed_orbit(bl)
    if sol.sol.retcode != BatchSolve.RETCODE_SUCCESS
        error("Unable to find closed orbit")
    end
    for i in 1:n_deltas
      co[i,:] = sol.v0
      co[i,6] += deltas[i]
    end
  end

  thetas = range(theta_lims[1], theta_lims[2], length=n_theta)
  rs = range(0, 1, length=n_r)[2:end]

  n_particles = n_deltas*(1+length(rs)*length(thetas))
  println("Initializing dynamic_aperture with $n_particles particles")
  v0 = zeros(n_particles, 6)
  v = zeros(n_particles, 6)
  idx_particle = 1
  for i in 1:n_deltas
    delta = deltas[i]
    # Initialize v0 in closed orbit basis, v in integration basis:
    v0[idx_particle,:] = [0, 0, 0, 0, 0, delta]
    v[idx_particle,:] = co[i,:]
    idx_particle += 1
    for theta in thetas
      for r in rs
        x_grid = max_sig_x * r * cos(theta)
        y_grid = max_sig_y * r * sin(theta)
        x = x_grid * sig_x
        y = y_grid * sig_y
        v0[idx_particle,:] = [x, 0, y, 0, 0, delta]
        v[idx_particle,:] = co[i,:] + [x, 0, y, 0, 0, 0]
        idx_particle += 1
      end
    end
  end

  if backend isa KA.GPU
    println("Initializing bunch on GPU")
    vt = KA.zeros(backend, coordinates_number_type, size(v))
    copy!(vt, v)
  else
    vt = v
  end

  b0 = Bunch(vt; p_over_q_ref=bl.p_over_q_ref, species=bl.species_ref)
  for i in 1:n_turns
    track!(b0, bl; scalar_params=true, track_kwargs...)
    print("\rFinished turn $i out of $n_turns")
    flush(stdout) 
  end
  println("\nTracking complete")

  if backend isa KA.GPU
    copy!(v, vt)
  end

  state = Array(b0.coords.state)

  # each column is a DA line
  x_norm_da = zeros(length(thetas), n_deltas)
  y_norm_da = zeros(length(thetas), n_deltas)

  # Loop thru the thetas, find max for each along r
  idx_particle = 1
  for i in LinearIndices(deltas)
      if state[idx_particle] != 0x1
          idx_particle += length(thetas)*length(rs)+1
          continue
      end
      idx_particle += 1
      for j in LinearIndices(thetas)
          
        # Sanity check:
          x = v0[idx_particle:idx_particle+length(rs)-1,1]./(max_sig_x.*sig_x)
          y = v0[idx_particle:idx_particle+length(rs)-1,3]./(max_sig_y.*sig_y)
          for (xi, yi) in zip(x,y)
            
              if !(atan(yi,xi) ≈ thetas[j])
                  writedlm("error.dlm", state)
                  writedlm("error.dlm", v)
                  println()
                  error("Something went wrong with the analysis. Submit an issue including output files.")
              end
              
          end
          
          if !isnothing(findfirst(t->t != 0x1, state[idx_particle:idx_particle+length(rs)-1]))
            idx_da = idx_particle-1 + findfirst(t->t != 0x1, state[idx_particle:idx_particle+length(rs)-1])

            # Sanity check:
            
            if idx_da-(idx_particle-1) != 1 && state[idx_da-1] != 0x1
                error("Something went wrong")
            end
            
            x_norm_da[j,i] = v0[idx_da,1]/sig_x
            y_norm_da[j,i] = v0[idx_da,3]/sig_y
            idx_particle += length(rs)
          else
            x_norm_da[j,i] = Inf
            y_norm_da[j,i] = Inf
            idx_particle += length(rs)
          end
      end
  end

  # output file will have first 6 columns as INITIAL coordinates wrt
  # closed orbit, followed by the state (alive or dead)
  if !isnothing(output_file)
    hcat(v0, state)
    #drow = vcat(deltas, deltas)
    #da_norms = hcat(x_norm_da, y_norm_da)
    #output_matrix = vcat(drow', da_norms)
    writedlm(output_file, hcat(v0, state), ';')
  end

  return x_norm_da, y_norm_da
end