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
    error("You specified delta_dependent_orbits = true but a nonzero emit_3. Instead specify sig_pz")
  elseif !delta_dependent_orbits && sig_pz != 0
    error("You specified delta_dependent_orbits = true but a nonzero emit_3. Instead specify sig_pz")
  end

  if !issorted(deltas)
    deltas = sort(deltas)
  end
  n_deltas = length(deltas)
  co = zeros(n_deltas, 6)
  co[:,6] .= deltas
  
  if delta_dependent_orbits
    # First, turn off all the cavities and store their strengths in 
    # an array
    cavities = filter(x->!isnothing(x.RFParams), bl.line)
    rfps = map(x->x.RFParams, cavities)
    # Turn them all off (doing this way to ensure inheritance + DefExpr remains):
    foreach(x->x.RFParams=nothing, cavities)

    local t
    try
      tw = twiss(bl, at=[1], cols=["E1", "E2", "dx", "dy"])
      if (:q3 in propertynames(tw))
        error("
          To compute delta_dependent_orbits, the beam must be coasting. We tried turning off 
          all cavities, but still did not detect coasting beam. Please turn off any elements 
          that cause longitudinal motion.
        ")
      end
      t = tw
    catch e
      # Put cavities back
      foreach((cavity,rfp)->cavity.RFParams=rfp, cavities, rfps)
      rethrow()
    end

    # Now compute sigmas at first element, just first order:
    sig_x = t.E1[1][1,1]*emit_1 + t.E2[1][1,1]*emit_2 
    sig_y = t.E1[1][3,3]*emit_1 + t.E2[1][3,3]*emit_2 
    sig_x += (t.dx[1]*sig_pz)^2
    sig_y += (t.dy[1]*sig_pz)^2
    sig_x = sqrt(sig_x)
    sig_y = sqrt(sig_y)

    # Compute delta-dependent closed orbits (with RF off)
    sol = find_closed_orbit(bl; v0=co, batch=Val{true}(), coasting_beam=true)
    # OK now we can turn the cavities back on:
    foreach((cavity,rfp)->cavity.RFParams=rfp, cavities, rfps)
    if any(sol.sol.retcode .!= BatchSolve.RETCODE_SUCCESS) # If any failed:
      error(
        """
        Unable for find delta-dependent closed orbits (with RF off) for deltas = $(deltas[findall(sol.sol.retcode .!= 0x0)]).
        Please remove these deltas from the input.
        """
      )
    end
  else
    t = twiss(bl, at=[1], cols=["E1", "E2", "E3"])
    # Now compute sigmas at first element, just first order:
    sig_x = t.E1[1][1,1]*emit_1 + t.E2[1][1,1]*emit_2 + t.E3[1][1,1]*emit_3
    sig_y = t.E1[1][3,3]*emit_1 + t.E2[1][3,3]*emit_2 + t.E3[1][3,3]*emit_3
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