function dynamic_aperture(
  bl::Beamline;
  n_r::Int,
  n_theta::Int,
  deltas::AbstractArray,
  max_sig_x::Real,
  max_sig_y::Real,
  emit_1::Real,
  emit_2::Real,
  emit_3::Real,
  n_turns::Int,
  output_file=nothing,
  theta_lims=(0, pi),
)
  Base.require_one_based_indexing(deltas)
  tw = twiss(bl, at=[first(bl.line)], de_moivre=true)
  t = tw.table

  # Now compute sigmas at first element
  E = t.E[1]
  sig_x = sqrt(E[1][1,1]*emit_1 + E[2][1,1]*emit_2) #+ E[3][1,1]*emit_3)
  sig_y = sqrt(E[1][3,3]*emit_1 + E[2][3,3]*emit_2) # + E[3][3,3]*emit_3)

  if length(E) == 3 # rf is on
    sig_x += E[3][1,1]*emit_3
    sig_y += E[3][3,3]*emit_3
  end

  thetas = range(theta_lims[1], theta_lims[2], length=n_theta)
  rs = range(0, 1, length=n_r)[2:end]

  n_particles = length(deltas)*(1+length(rs)*length(thetas))
  println("Initializing dynamic_aperture with n_particles = $n_particles")
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

  b0 = Bunch(v; p_over_q_ref=bl.p_over_q_ref, species=bl.species_ref)
  for i in 1:n_turns
    print("\rTracking turn: $i out of $n_turns")
    flush(stdout) 
    track!(b0, bl, scalar_params=true)
  end
  println("\nTracking complete")

  writedlm("state.dlm", b0.coords.state)
  writedlm("v.dlm", b0.coords.v)
  println()

  # each column is a DA line
  x_norm_da = zeros(length(thetas), length(deltas))
  y_norm_da = zeros(length(thetas), length(deltas))

  # Loop thru the thetas, find max for each along r
  idx_particle = 1
  for i in LinearIndices(deltas)
      if b0.coords.state[idx_particle] != 0x1
          idx_particle += length(thetas)*length(rs)
          continue
      end
      idx_particle += 1
      for j in LinearIndices(thetas)
          
        # Sanity check:
          x = v0[idx_particle:idx_particle+length(rs)-1,1]./(max_sig_x.*sig_x)
          y = v0[idx_particle:idx_particle+length(rs)-1,3]./(max_sig_y.*sig_y)
          for (xi, yi) in zip(x,y)
              if !(atan(yi,xi) ≈ thetas[j])
                  error("Something went wrong")
              end
          end
          
          if !isnothing(findfirst(t->t != 0x1, b0.coords.state[idx_particle:idx_particle+length(rs)-1]))
            idx_da = idx_particle-1 + findfirst(t->t != 0x1, b0.coords.state[idx_particle:idx_particle+length(rs)-1])

            # Sanity check:
            if idx_da-(idx_particle-1) != 1 && b0.coords.state[idx_da-1] != 0x1
                error("Something went wrong")
            end
            
            x_norm_da[j,i] = v0[idx_da,1]/sig_x
            y_norm_da[j,i] = v0[idx_da,3]/sig_y
            idx_particle += length(rs)
          else
            x_norm_da[j,i] = Inf #v0[idx_da,1]/sig_x
            y_norm_da[j,i] = Inf #v0[idx_da,3]/sig_y
            idx_particle += length(rs)
          end
      end
  end

  if !isnothing(output_file)
    drow = vcat(deltas, deltas)
    da_norms = hcat(x_norm_da, y_norm_da)
    output_matrix = vcat(drow', da_norms)
    writedlm(output_file, output_matrix, ';')
  end

  return x_norm_da, y_norm_da
end