"""
    dynamic_aperture(ring::Beamline; kwargs...) -> NTuple{2,Matrix{Float64}}

Computes the acceptance of the ring by pushing a polar grid in `x/sig_x` and `y/sig_y`
space, for each of the provided `deltas`. Returns a tuple of two matrices whose first
index corresponds to the angle on the polar grid and whose second index corresponds to
the entry in `deltas`.

For each angle the returned value is the **largest sampled amplitude that survived** all
`n_turns` turns, so the reported acceptance is never larger than the true one. The true
boundary lies between that amplitude and the next radial sample, i.e. the result carries
an uncertainty of one radial step: `max_sig_x/(n_r-1)` in `x/sig_x` and `max_sig_y/(n_r-1)`
in `y/sig_y`. Choose `max_sig_x`, `max_sig_y` and `n_r` so that several radial samples fall
inside the aperture; the radial step in metres is printed when the scan starts.

Two sentinel values are returned when the grid fails to bracket the aperture, and each
raises a warning:
- `NaN` -- even the innermost sampled amplitude was lost, so the grid is too coarse to
  resolve the aperture at that angle. Reduce `max_sig_x`/`max_sig_y`, or increase `n_r`.
  `NaN` is also returned for a `delta` whose closed orbit could not survive the tracking.
- `Inf` -- no sampled amplitude was lost, so the aperture lies beyond the grid. Increase
  `max_sig_x`/`max_sig_y`.

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
  if n_r < 2
    error("n_r must be at least 2 (got n_r = $n_r): the radial grid holds n_r-1 sampled amplitudes.")
  end
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

  # Radial resolution of the scan. The result is only meaningful if several of these steps
  # fit inside the aperture, so report it up front.
  dr_x = max_sig_x*sig_x*first(rs)
  dr_y = max_sig_y*sig_y*first(rs)

  n_particles = n_deltas*(1+length(rs)*length(thetas))
  println("Initializing dynamic_aperture with $n_particles particles")
  println("  sig_x = $sig_x m, sig_y = $sig_y m")
  println("  radial step = $dr_x m in x, $dr_y m in y")
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
  n_unresolved = 0   # innermost sampled amplitude already lost -> grid too coarse
  n_unbounded  = 0   # nothing lost along the ray -> aperture lies beyond the grid
  n_co_lost    = 0   # closed orbit itself did not survive
  idx_particle = 1
  for i in LinearIndices(deltas)
      if state[idx_particle] != 0x1
          n_co_lost += 1
          x_norm_da[:,i] .= NaN
          y_norm_da[:,i] .= NaN
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

          # The aperture is bracketed by the last surviving sample and the first lost one.
          # Report the last SURVIVING amplitude: reporting the first LOST one overstates the
          # acceptance by up to one radial step, and on a grid too coarse to resolve the
          # aperture it degenerates into reporting the grid itself rather than the machine.
          idx_first_lost = findfirst(t->t != 0x1, state[idx_particle:idx_particle+length(rs)-1])

          if isnothing(idx_first_lost)
              n_unbounded += 1
              x_norm_da[j,i] = Inf
              y_norm_da[j,i] = Inf
          elseif idx_first_lost == 1
              # Nothing survived, not even the innermost ring: the aperture lies somewhere
              # inside the first radial step and this scan cannot say where.
              n_unresolved += 1
              x_norm_da[j,i] = NaN
              y_norm_da[j,i] = NaN
          else
              idx_da = idx_particle-1 + idx_first_lost - 1
              x_norm_da[j,i] = v0[idx_da,1]/sig_x
              y_norm_da[j,i] = v0[idx_da,3]/sig_y
          end
          idx_particle += length(rs)
      end
  end

  n_points = length(thetas)*n_deltas
  if n_co_lost > 0
      @warn "dynamic_aperture: the closed orbit itself was lost for $n_co_lost of $n_deltas deltas; " *
            "those columns are returned as NaN."
  end
  if n_unresolved > 0
      @warn "dynamic_aperture: the innermost sampled amplitude was already lost at $n_unresolved of " *
            "$n_points (angle, delta) points, so the aperture is unresolved there and is reported as NaN. " *
            "The radial step is $dr_x m in x and $dr_y m in y. Reduce max_sig_x/max_sig_y, or increase " *
            "n_r, so that several radial samples fall inside the aperture."
  end
  if n_unbounded > 0
      @warn "dynamic_aperture: no sampled amplitude was lost at $n_unbounded of $n_points " *
            "(angle, delta) points, so the aperture lies beyond the grid and is reported as Inf. " *
            "Increase max_sig_x/max_sig_y to bracket it."
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