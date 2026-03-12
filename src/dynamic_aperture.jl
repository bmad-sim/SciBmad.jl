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
  coordinates_number_type::Type=Float32, 
  sig_pz::Real=0,
  delta_dependent_orbits::Bool=true,
  output_file=nothing,
  theta_lims=(0, pi),
  track_kwargs... # Get passed to track!
)
  Base.require_one_based_indexing(deltas)

  # First, turn off all the cavities and store their strengths in 
  # an array
  cavities = filter(x->!isnothing(x.RFParams), bl.line)
  rfps = map(x->x.RFParams, cavities)
  # Turn them all off (doing this way to ensure inheritance + DefExpr remains):
  foreach(x->x.RFParams=nothing, cavities)

  tw = twiss(bl, at=[first(bl.line)], de_moivre=true)
  t = tw.table

  # Now compute sigmas at first element, just first order:
  E = t.E[1]
  sig_x = E[1][1,1]*emit_1 + E[2][1,1]*emit_2 
  sig_y = E[1][3,3]*emit_1 + E[2][3,3]*emit_2 
  eta_x = t.orbit_x[1][6]
  eta_y = t.orbit_y[1][6]
  sig_x += (eta_x*sig_pz)^2
  sig_y += (eta_y*sig_pz)^2
  sig_x = sqrt(sig_x)
  sig_y = sqrt(sig_y)

  co = zeros(length(deltas), 6)
  if delta_dependent_orbits
    # Compute delta-dependent closed orbits (with RF off)
    for i in 1:length(deltas)
      co[i,6] = deltas[i]
      sol = find_closed_orbit(bl, v0=co[i,:]')
      if sol.converged == false
        error("Unable for find delta-dependent closed orbit (with RF off) for delta = $delta.
              Please remove this delta from the input deltas.")
      end
      co[i,:] = sol.u
    end
    # OK now we can turn the cavities back on:
    foreach((cavity,rfp)->cavity.RFParams=rfp, cavities, rfps)
  else
    # Turn RF back on, get the only closed orbit
    foreach((cavity,rfp)->cavity.RFParams=rfp, cavities, rfps)
    sol = find_closed_orbit(bl)
    if sol.converged == false
        error("Unable to find closed orbit")
    end
    for i in 1:length(deltas)
      co[i,:] = sol.u
      co[i,6] += deltas[i]
    end
  end

  thetas = range(theta_lims[1], theta_lims[2], length=n_theta)
  rs = range(0, 1, length=n_r)[2:end]

  n_particles = length(deltas)*(1+length(rs)*length(thetas))
  println("Initializing dynamic_aperture with $n_particles particles")
  v0 = zeros(n_particles, 6)
  v = zeros(n_particles, 6)
  idx_particle = 1
  for i in 1:length(deltas)
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
 # @show vt
 # @show v0

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
  x_norm_da = zeros(length(thetas), length(deltas))
  y_norm_da = zeros(length(thetas), length(deltas))

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
          @show x
          @show y
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