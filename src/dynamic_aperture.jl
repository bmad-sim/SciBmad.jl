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
  coordinates_number_type::Type=Float32, # Default to Float32 for performance
  emit_3::Real=0,
  sig_pz::Real=0,
  output_file=nothing,
  theta_lims=(0, pi),
  track_kwargs... # Get passed to track!
)
  Base.require_one_based_indexing(deltas)
  tw = twiss(bl, at=[first(bl.line)], de_moivre=true)
  t = tw.table

  # Now compute sigmas at first element, just first order:
  E = t.E[1]
  sig_x = E[1][1,1]*emit_1 + E[2][1,1]*emit_2 
  sig_y = E[1][3,3]*emit_1 + E[2][3,3]*emit_2 

  if length(E) == 3 # rf is on
    if emit_3 == 0 && sig_pz != 0
      error("You specified sig_pz, but the longitudinal plane has pseudo-harmonic oscillations.
             Please specify emit_3 instead.")
    end
    sig_x += E[3][1,1]*emit_3
    sig_y += E[3][3,3]*emit_3
  else
    if sig_pz == 0 && emit_3 != 0
      error("You specified emit_3, but the longitudinal plane is coasting.
             Please specify sig_pz instead.")
    end
    eta_x = t.orbit_x[1][6]
    eta_y = t.orbit_y[1][6]
    sig_x += (eta_x*sig_pz)^2
    sig_y += (eta_y*sig_pz)^2
  end

  sig_x = sqrt(sig_x)
  sig_y = sqrt(sig_y)

  thetas = range(theta_lims[1], theta_lims[2], length=n_theta)
  rs = range(0, 1, length=n_r)[2:end]

  n_particles = length(deltas)*(1+length(rs)*length(thetas))
  println("Initializing dynamic_aperture with $n_particles particles")
  v0 = zeros(n_particles, 6)
  idx_particle = 1
  for delta in deltas
      v0[idx_particle,:] = [0, 0, 0, 0, 0, delta]
      idx_particle += 1
      for theta in thetas
          for r in rs
              x_grid = max_sig_x * r * cos(theta)
              y_grid = max_sig_y * r * sin(theta)
              x = x_grid * sig_x
              y = y_grid * sig_y
              v0[idx_particle,:] = [x, 0, y, 0, 0, delta]
              idx_particle += 1
          end
      end
  end

  # These coordinates are in closed orbit basis, need to put in integration frame
  co =  scalar.([t.orbit_x[1], t.orbit_px[1], t.orbit_y[1], t.orbit_py[1], t.orbit_z[1], t.orbit_pz[1]])
  v = zeros(n_particles, 6)
  for i in 1:n_particles
      v[i,:] = co + v0[i,:]
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
  x_norm_da = zeros(length(thetas), length(deltas))
  y_norm_da = zeros(length(thetas), length(deltas))

  # Loop thru the thetas, find max for each along r
  idx_particle = 1
  for i in LinearIndices(deltas)
      if state[idx_particle] != 0x1
          idx_particle += length(thetas)*length(rs)
          continue
      end
      idx_particle += 1
      for j in LinearIndices(thetas)
          
        # Sanity check:
          x = v0[idx_particle:idx_particle+length(rs)-1,1]./(max_sig_x.*sig_x)
          y = v0[idx_particle:idx_particle+length(rs)-1,3]./(max_sig_y.*sig_y)
          for (xi, yi) in zip(x,y)
            #=
              if !(atan(yi,xi) ≈ thetas[j])
                  writedlm("error.dlm", state)
                  writedlm("error.dlm", v)
                  println()
                  error("Something went wrong with the analysis. Submit an issue including output files.")
              end
              =#
          end
          
          if !isnothing(findfirst(t->t != 0x1, state[idx_particle:idx_particle+length(rs)-1]))
            idx_da = idx_particle-1 + findfirst(t->t != 0x1, state[idx_particle:idx_particle+length(rs)-1])

            # Sanity check:
            #=
            if idx_da-(idx_particle-1) != 1 && state[idx_da-1] != 0x1
                error("Something went wrong")
            end
            =#
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