using FundamentalFrequencies, Statistics, LinearAlgebra, SparseArrays
import KernelAbstractions as KA
import NonlinearNormalForm: S
import NonlinearNormalForm as NNF
using StaticArrays
using DelimitedFiles
using BatchSolve

function dynamic_aperture(
    bl::Beamline;

    # Required kwargs:
    n_r::Int,
    n_theta::Int,
    deltas::AbstractArray,
    nsig_1::Real,
    nsig_2::Real,
    emit_1::Real,
    emit_2::Real,
    n_turns::Int,

    # Optional kwargs:
    backend=KA.CPU(),
    coordinates_number_type::Type=Float64, 
    delta_dependent_orbits::Bool=SciBmad.coast_check(bl), 
    verbose=true,
   # output_prefix="da_",
  )
  
  Base.require_one_based_indexing(deltas)
  if !issorted(deltas)
    deltas = sort(deltas)
  end
  n_deltas = length(deltas)

  # Linear normalizing maps at each delta
  as = zeros(4, 4, n_deltas) # (row, col, delta)
  co = zeros(n_deltas, 6)
  co[:,6] .= deltas

  if delta_dependent_orbits
    # This may be set to true by the user even if not coasting beam
    # so we have to turn off all cavities. 
    cavities = filter(x->!isnothing(x.RFParams), bl.line)
    rfps = map(x->x.RFParams, cavities)
    # Turn them all off (doing this way to ensure inheritance + DefExpr remains):
    foreach(x->x.RFParams=nothing, cavities)

    # Around each orbit do linear normal form 
    sol = find_closed_orbit(bl; v0=co, batch=Val{true}(), coasting_beam=true)
    if any(sol.sol.retcode .!= BatchSolve.RETCODE_SUCCESS) # If any failed:
      error(
        """
        Unable for find delta-dependent closed orbits (with RF off) for deltas = $(deltas[findall(sol.sol.retcode .!= 0x0)]).
        Please remove these deltas from the input.
        """
      )
    end

    # OK now we can turn the cavities back on:
    foreach((cavity,rfp)->cavity.RFParams=rfp, cavities, rfps)
    
    # Get the last Jacobians, these will be delta-dependent maps
    ms = reshape(findnz(sol.sol.jac)[end], 4, n_deltas, 4) # (row, delta, col)
    c = 1/sqrt(2)*SA[1 1 0 0; -im im 0 0; 0 0 1 1; 0 0 -im im]

    # Now we want to get linear normal forms:
    for i in 1:n_deltas
      m = symplectify(I-view(ms, 1:4, i, 1:4))
      F = NNF.mat_eigen(transpose(m), phase_modes=false)
      ai = real(c*transpose(F.vectors))
      as[:,:,i] = inv(ai) # transformation from Floquet variables to real space
    end
  else
    # In this case we need to search for the 1-torus that passes through z=0, delta=delta
    # We'll do this delta-by-delta in order to encourage continuous dynamic aperture
    # from minimum delta to maximum delta
    # this will be done on the CPU, because single particle tracking
    out = walk_J3(bl, deltas; verbose=verbose)
    co .= out[1]
    as .= out[2]
  end

  # Now we have the closed orbits and A at each point
  # Construct distribution
  thetas = range(0, pi/2, length=n_theta)
  rs = range(0, 1, length=n_r)[2:end]

  n_particles = n_deltas*(1+length(rs)*length(thetas))
  println("Initializing dynamic_aperture with $n_particles particles")
  Jt1 = zeros(coordinates_number_type, (1+length(rs)*length(thetas)), n_deltas) # Horizontal-like action
  Jt2 = zeros(coordinates_number_type, (1+length(rs)*length(thetas)), n_deltas) # Vertical-like action
  v = zeros(coordinates_number_type, n_particles, 6)  # Coordinates in integration frame for tracking
  idx_particle = 1
  for i in 1:n_deltas
    v[idx_particle,:] = co[i,:]
    idx_particle += 1
    for theta in thetas
      for r in rs
        sqrttwoJ_1i = nsig_1 * r * cos(theta) * sqrt(emit_1) # Number from 0 to n_sig_1*sqrt(emit_1)
        sqrttwoJ_2i = nsig_2 * r * sin(theta) * sqrt(emit_2) # From 0 to n_sig_2*sqrt(emit_2)
        reshape(Jt1, :)[idx_particle] = sqrttwoJ_1i/sqrt(emit_1)
        reshape(Jt2, :)[idx_particle] = sqrttwoJ_2i/sqrt(emit_2)
        # Use phi=0 -> cos=1, sin=0 in Floquet variables
        # Transform back to real variables with a
        v[idx_particle,:] = view(co, i, :)
        v[idx_particle,1:4] += view(as, :, :, i) * SA[sqrttwoJ_1i, 0, sqrttwoJ_2i, 0] 
        idx_particle += 1
      end
    end
  end

  # v0 has shape (1+length(rs)*length(thetas)) x 6

  if backend isa KA.GPU
    println("Initializing bunch on GPU")
    vt = KA.zeros(backend, coordinates_number_type, size(v))
    copy!(vt, v)
  else
    vt = v
  end

  turns_survivedt = similar(vt, Int, size(vt, 1))
  turns_survivedt .= n_turns+1

  b0 = Bunch(vt; p_over_q_ref=bl.p_over_q_ref, species=bl.species_ref)
  for i in 1:n_turns
    track!(b0, bl; scalar_params=true)
    @. turns_survivedt = ifelse(b0.coords.state != BeamTracking.STATE_ALIVE && turns_survivedt == n_turns+1, i, turns_survivedt)
    if verbose
      print("\rFinished turn $i out of $n_turns")
      flush(stdout) 
    end
  end
  if verbose
    println()
  end

  if backend isa KA.GPU
    copy!(v, vt)
  end


  turns_survived = similar(v, Int, div(size(vt, 1), n_deltas), n_deltas)
  copyto!(reshape(turns_survived, :), turns_survivedt)
#=
  if !isnothing(output_file)
    writedlm(output_prefix*"", hcat(v, reshape(turns_survived, :)), ';')
  end
  =#
  # Return X, Y, and Z for heatmap plotting
  # this will be sqrt(2J_1/<J_1>), sqrt(2J_2/<J_2>), and turns_survived
  return Jt1, Jt2, turns_survived
end

# Last step is to just vectorize this function now
# It should keep going up to the max delta 
function walk_J3(bl::Beamline, deltas; verbose=true, as=nothing, maxiter=10)
  n_deltas = size(deltas, 2)    # Number of cols
  n_particles = size(deltas, 1) # Number of rows
  tw = twiss(bl; at=[first(bl.line)], normalizing_map=true)
  Q_guess = similar(deltas, n_particles, 3)
  copyto!(Q_guess, -reshape(repeat(tw.tunes, inner=n_particles), n_particles, 3))
  y = similar(deltas, n_particles, 4) # sol
  v = similar(deltas, n_particles, 4) # guess, will change for each delta
  v_cache = similar(v, n_particles, 6)
  co = similar(v, n_particles, 6, n_deltas)
  if !isnothing(as)
    if size(as) != (n_particles, 4, 4, n_deltas)
      error("Invalid size for `as` keyword argument. Received $(size(as)), requires $((n_particles, 4, 4, n_deltas))")
    else
      as .= 0
    end
  end
  co .= 0
  y .= 0
  v .= 0 # Assume closed orbit is zero orbit for all, in general should solve for all

  if verbose
    println("Computing J₁=J₂=0 coordinate for each J₃")
  end
  for i in 1:n_deltas
    delta = view(deltas, :, i)
    sol = newton!(
      transverse_frequencies!, 
      y,
      v, 
      Cache(v_cache),
      Cache(Q_guess),
      Constant(bl),
      Constant(delta); 
      abstol=1e-12, 
      reltol=1e-12,
      batchdim=1,
      autodiff=AutoBatch(AutoFiniteDiff(fdjtype=Val(:central))),
      maxiter=maxiter,
      verbose=verbose,
    )

    # This will act on each row (batch)
    @. v = ifelse(sol.retcode != BatchSolve.RETCODE_SUCCESS, Inf, v) 
    # Jacobian will always be zero then
    # And Q_guess should always be bad

    co[:,1:4,i] .= sol.u
    co[:,6,i] .= delta
    if !isnothing(as)
      if delta == 0 # use twiss a for all inputs
        as[:,:,:,i] .= reshape(view(NNF.jacobian(tw.table.a[1]), 1:4, 1:4), 1, 4, 4)
      else # compute from Fourier modes
        if n_particles > 1
          error("Returning `as` for n_particles > 1 not implemented yet")
        end
        ai = symplectify(reshape(findnz(sol.jac)[end], 4, 4))
        as[1,:,:,i] = inv(ai) # transformation from Floquet variables to real space
      end
    end
    if verbose
      print("\r", rpad("Finished iteration $i", 20))
      flush(stdout)
    end
  end
  if verbose
    println()
  end
  return co, Q_guess
end

function transverse_frequencies!(
    y,       # Output, n_particles x 4
    v,       # Input, n_particles x 4
    v_cache, # DI.Cache, n_particles x 6
    Q,       # DI.Cache, n_particles x 3
    bl,      # DI.Constant, Beamline
    deltas;  # DI.Constant, n_particles x 1
    n_frequencies=20,
    n_turns=500,
    order=3,
    verbose=true,
    window_order=5,
    reltol=0.005,#,=0.01,
    abstol=1e-5, 
  )

  n_particles = size(v, 1)
  v_cache .= 0
  v_cache[:,6] .= deltas
  v_cache[:,1:4] .= v
  res = track(bl; v0=v_cache, n_turns, verbose, use_explicit_SIMD=false)
  coords = res.v
  data = reshape(permutedims(coords, (2,1,3)), 6*n_particles, n_turns+1) 

  # Now this makes it x + im*px, etc:
  data = reinterpret(ComplexF64, data)

  # Remove the mean:
  data = data .- mean(data, dims=2) 

  # Batched NAFF
  frequencies, amplitudes = naff(data, n_frequencies; window_order, warnings=false)
  frequencies = reshape(frequencies, 3, n_particles, n_frequencies)
  amplitudes = reshape(amplitudes, 3, n_particles, n_frequencies)
  Q3 = reshape(view(Q, :, 3), 1, n_particles, 1)
  #@show frequencies[1:3,1,:]
  #@show frequencies[3,1,:]
  # Select dominant frequency (longitudinal mode)
  f3, idx3 = findmin(abs.(Q3 .- frequencies) .* (1 .+ 1 ./ abs.(amplitudes)), dims=3)
  f3tru, idx3tru = findmin(f3, dims=1)
  good = f3tru ./ (1 .+ 1 ./ abs.(amplitudes[idx3][idx3tru])) .< reltol
  Q3 .= good .* frequencies[idx3][idx3tru] .+ .!good .* Q3
  #@show Q3
  #@show good
  #@show frequencies[idx3][idx3tru]
  # amp3 = good .* amplitudes[idx3][idx3tru]

  # "Remove this" and all integer multiples from the result 
  # by setting those frequencies to gigantic. This will make sure 
  # we don't accidentally find them, however does not 
  # handle the cross terms yet.
  # Each column of Q3 corresponds to a chunk of 3-rows in frequencies 
  # multol = 1e-1
  j_vals = reshape(0:order, 1, 1, 1, order+1)  # (1, 1, 1, n_orders)

  ismul = any(
      abs.(frequencies .- j_vals .* Q3) .< abstol .||
      abs.(frequencies .+ j_vals .* Q3) .< abstol,
      dims=4
  )  # (3, n_particles, n_frequencies, 1)

  frequencies .+= dropdims(ismul, dims=4) .* 1e10

  # Do the next dominant frequency. We have to determine this
  # by checking which difference/amplitude is the largest of 1 and 2. Kill this and 
  # all integer multiples. In this case, we actually need to check if 
  # the result is "good" or not, because close to the end, we will 
  # only have Q3 +- integer multiples. If no good guess, then 
  # we can just assume the amplitude is zero 
  # CHECK:
  Q1 = reshape(view(Q, :, 1), 1, n_particles, 1)
  Q2 = reshape(view(Q, :, 2), 1, n_particles, 1)

  f1, idx1 = findmin(abs.(Q1 .- frequencies) .* (1 .+ 1 ./ abs.(amplitudes)), dims=3)
  f1tru, idx1tru = findmin(f1, dims=1)

  f2, idx2 = findmin(abs.(Q2 .- frequencies) .* (1 .+ 1 ./ abs.(amplitudes)), dims=3)
  f2tru, idx2tru = findmin(f2, dims=1)

  # Whichever is smaller, do that first (closer match):
  if first(f2tru) < first(f1tru)
      next = 2
      final = 1
      good = f2tru ./ (1 .+ 1 ./ abs.(amplitudes[idx2][idx2tru])) .< reltol
      Qnext = good .* frequencies[idx2][idx2tru] .+ .!good .* Q2
      amp2 = good .* amplitudes[idx2][idx2tru] # amp is zero if not good
      Q2 .= Qnext
      #@show Q2
      #@show good
      #@show frequencies[idx2][idx2tru]
  else
      next = 1
      final = 2
      good = f1tru ./ (1 .+ 1 ./ abs.(amplitudes[idx1][idx1tru])) .< reltol
      Qnext = good .* frequencies[idx1][idx1tru]  .+ .!good .* Q1
      amp1 = good .* amplitudes[idx1][idx1tru] # amp is zero if not good
      Q1 .= Qnext
      #@show Q1
      #@show good
      #@show frequencies[idx1][idx1tru]
  end
  # Remove the next one
  j_vals = reshape(0:order, 1, 1, 1, order+1, 1)        # (1,1,1,n_j,1)
  k_vals = reshape(0:order, 1, 1, 1, 1, order+1)        # (1,1,1,1,n_k)
  Q3_bc   = reshape(Q3,   1, n_particles, 1, 1, 1)      # already (1,n_p,1) -> add 2 more
  Qnext_bc = reshape(Qnext, 1, n_particles, 1, 1, 1)
  fv = reshape(frequencies, 3, n_particles, n_frequencies, 1, 1)

  ismul2_full = (
    abs.(fv .- j_vals .* Q3_bc .- k_vals .* Qnext_bc) .< abstol .||
    abs.(fv .- j_vals .* Q3_bc .+ k_vals .* Qnext_bc) .< abstol .||
    abs.(fv .+ j_vals .* Q3_bc .- k_vals .* Qnext_bc) .< abstol .||
    abs.(fv .+ j_vals .* Q3_bc .+ k_vals .* Qnext_bc) .< abstol
  )  # (3, n_particles, n_frequencies, order+1, order+1)
  ismul2 = any(any(ismul2_full, dims=5), dims=4)
  #=
  ismul2 = any(
      abs.(fv .- j_vals .* Q3_bc .- k_vals .* Qnext_bc) .< abstol .||
      abs.(fv .- j_vals .* Q3_bc .+ k_vals .* Qnext_bc) .< abstol .||
      abs.(fv .+ j_vals .* Q3_bc .- k_vals .* Qnext_bc) .< abstol .||
      abs.(fv .+ j_vals .* Q3_bc .+ k_vals .* Qnext_bc) .< abstol,
      dims=(4,5)
  )  # (3, n_particles, n_frequencies, 1, 1)
=#
  frequencies .+= dropdims(ismul2, dims=(4,5)) .* 1e10

  # Finally get our guess for the last. Same idea as before. Here however, 
  # if the tune is too close to any other
  Qfinal = reshape(view(Q, :, final), 1, n_particles, 1)
  ff, idxf = findmin(abs.(Qfinal .- frequencies) .* (1 .+ 1 ./ abs.(amplitudes)), dims=3)
  fftru, idxftru = findmin(ff, dims=1)
  good = fftru ./ (1 .+ 1 ./ abs.(amplitudes[idxf][idxftru])) .< reltol
  Qfinal .= good .* frequencies[idxf][idxftru] .+ .!good .* Qfinal
  if final == 1
      amp1 = good .* amplitudes[idxf][idxftru]
  else
      amp2 = good .* amplitudes[idxf][idxftru] # amp is zero if not good
  end
  #@show Qfinal
  #@show good
  #@show frequencies[idxf][idxftru]

  y .=  hcat(
    permutedims(reinterpret(Float64, reshape(amp1, 1, n_particles)), (2,1)),
    permutedims(reinterpret(Float64, reshape(amp2, 1, n_particles)), (2,1)),
    #permutedims(reinterpret(Float64, reshape(amp3, 1, n_particles)), (2,1))
  )

  return y
end

#=
function transverse_frequencies!(
    y,       # Output, n_particles x 4
    v,       # Input, n_particles x 4
    v_cache, # DI.Cache, n_particles x 6
    Q,       # DI.Cache, n_particles x 3
    bl,      # DI.Constant, Beamline
    deltas;  # DI.Constant, n_particles x 1
    multol=1e-4, 
    n_frequencies=20,
    n_turns=500,
    order=3,
    verbose=true,
    window_order=5,
  )

  n_particles = size(v, 1)
  v_cache .= 0
  v_cache[:,6] .= deltas
  v_cache[:,1:4] .= v
  res = track(bl; v0=v_cache, n_turns, verbose, use_explicit_SIMD=false)
  coords = res.v
  data = reshape(permutedims(coords, (2,1,3)), 6*n_particles, n_turns+1) 

  # Now this makes it x + im*px, etc:
  data = reinterpret(ComplexF64, data)

  # Remove the mean:
  data = data .- mean(data, dims=2) 

  # Batched NAFF
  frequencies, amplitudes = naff(data, n_frequencies; window_order, warnings=false)

  frequencies = reshape(frequencies, 3, n_particles, n_frequencies)
  amplitudes = reshape(amplitudes, 3, n_particles, n_frequencies)
  Q3 = reshape(view(Q, :, 3), 1, n_particles, 1)

  # Select dominant frequency (longitudinal mode)
  f3, idx3 = findmin(abs.((Q3 .- frequencies) ./ Q3) ./ abs.(amplitudes), dims=3)
  f3tru, idx3tru = findmin(f3, dims=1)
  good = f3tru .< 1/multol
  Q3 .= good .* frequencies[idx3][idx3tru] .+ .!good .* Q3
  @show good
  # amp3 = good .* amplitudes[idx3][idx3tru]

  # "Remove this" and all integer multiples from the result 
  # by setting those frequencies to gigantic. This will make sure 
  # we don't accidentally find them, however does not 
  # handle the cross terms yet.
  # Each column of Q3 corresponds to a chunk of 3-rows in frequencies 
  # multol = 1e-1
  j_vals = reshape(0:order, 1, 1, 1, order+1)  # (1, 1, 1, n_orders)

  ismul = any(
      abs.(frequencies .- j_vals .* Q3) .< multol .||
      abs.(frequencies .+ j_vals .* Q3) .< multol,
      dims=4
  )  # (3, n_particles, n_frequencies, 1)

  frequencies .+= dropdims(ismul, dims=4) .* 1e10

  # Do the next dominant frequency. We have to determine this
  # by checking which difference/amplitude is the largest of 1 and 2. Kill this and 
  # all integer multiples. In this case, we actually need to check if 
  # the result is "good" or not, because close to the end, we will 
  # only have Q3 +- integer multiples. If no good guess, then 
  # we can just assume the amplitude is zero 
  # CHECK:
  Q1 = reshape(view(Q, :, 1), 1, n_particles, 1)
  Q2 = reshape(view(Q, :, 2), 1, n_particles, 1)

  f1, idx1 = findmin(abs.((Q1 .- frequencies) ./ Q1) ./ abs.(amplitudes), dims=3)
  f1tru, idx1tru = findmin(f1, dims=1)

  f2, idx2 = findmin(abs.((Q2 .- frequencies) ./ Q2) ./ abs.(amplitudes), dims=3)
  f2tru, idx2tru = findmin(f2, dims=1)

  # Whichever is smaller, do that first (closer match):
  if first(f2tru) < first(f1tru)
      next = 2
      final = 1
      good = f2tru .< 1/multol
      Qnext = good .* frequencies[idx2][idx2tru] .+ .!good .* Q2
      amp2 = good .* amplitudes[idx2][idx2tru] # amp is zero if not good
      Q2 .= Qnext
      @show good
  else
      next = 1
      final = 2
      good = f1tru .< 1/multol # this may need fiddling
      Qnext = good .* frequencies[idx1][idx1tru]  .+ .!good .* Q1
      amp1 = good .* amplitudes[idx1][idx1tru] # amp is zero if not good
      Q1 .= Qnext
      @show good
  end

  # Remove the next one
  j_vals = reshape(0:order, 1, 1, 1, order+1, 1)        # (1,1,1,n_j,1)
  k_vals = reshape(0:order, 1, 1, 1, 1, order+1)        # (1,1,1,1,n_k)
  Q3_bc   = reshape(Q3,   1, n_particles, 1, 1, 1)      # already (1,n_p,1) -> add 2 more
  Qnext_bc = reshape(Qnext, 1, n_particles, 1, 1, 1)
  fv = reshape(frequencies, 3, n_particles, n_frequencies, 1, 1)

  ismul2 = any(
      abs.(fv .- j_vals .* Q3_bc .- k_vals .* Qnext_bc) .< multol .||
      abs.(fv .- j_vals .* Q3_bc .+ k_vals .* Qnext_bc) .< multol .||
      abs.(fv .+ j_vals .* Q3_bc .- k_vals .* Qnext_bc) .< multol .||
      abs.(fv .+ j_vals .* Q3_bc .+ k_vals .* Qnext_bc) .< multol,
      dims=(4,5)
  )  # (3, n_particles, n_frequencies, 1, 1)

  frequencies .+= dropdims(ismul2, dims=(4,5)) .* 1e10

  # Finally get our guess for the last. Same idea as before. Here however, 
  # if the tune is too close to any other
  Qfinal = reshape(view(Q, :, final), 1, n_particles, 1)
  ff, idxf = findmin(abs.((Qfinal .- frequencies) ./ Qfinal) ./ abs.(amplitudes), dims=3)
  fftru, idxftru = findmin(ff, dims=1)
  good = fftru .< 1/multol
  Qfinal .= good .* frequencies[idxf][idxftru] .+ .!good .* Qfinal
  if final == 1
      amp1 = good .* amplitudes[idxf][idxftru]
  else
      amp2 = good .* amplitudes[idxf][idxftru] # amp is zero if not good
  end
  @show good
  y .=  hcat(
    permutedims(reinterpret(Float64, reshape(amp1, 1, n_particles)), (2,1)),
    permutedims(reinterpret(Float64, reshape(amp2, 1, n_particles)), (2,1)),
    #permutedims(reinterpret(Float64, reshape(amp3, 1, n_particles)), (2,1))
  )

  return y
end
=#

#=
function transverse_frequencies!(
    y,       # Output, n_particles x 4
    v,       # Input, n_particles x 4
    v_cache, # DI.Cache, n_particles x 6
    Q,       # DI.Cache, n_particles x 3
    bl,      # DI.Constant, Beamline
    deltas;  # DI.Constant, n_particles x 1
    n_frequencies=20,
    n_turns=500,
    order=3,
    verbose=true,
    window_order=5,
    reltol=0.02,
    abstol=1e-5, 
  )

  n_particles = size(v, 1)
  v_cache .= 0
  v_cache[:,6] .= deltas
  v_cache[:,1:4] .= v
  res = track(bl; v0=v_cache, n_turns, verbose, use_explicit_SIMD=false)
  coords = res.v
  if any(res.state[:,end] .!= 0x1)
    error("fuck")
  end
  data = reshape(permutedims(coords, (2,1,3)), 6*n_particles, n_turns+1) 

  # Now this makes it x + im*px, etc:
  data = reinterpret(ComplexF64, data)

  # Remove the mean:
  data = data .- mean(data, dims=2) 

  # Batched NAFF
  frequencies, amplitudes = naff(data, n_frequencies; window_order, warnings=false)

  frequencies = reshape(frequencies, 3, n_particles, n_frequencies)
  amplitudes = reshape(amplitudes, 3, n_particles, n_frequencies)
  Q3 = reshape(view(Q, :, 3), 1, n_particles, 1)

  # Select dominant frequency (longitudinal mode)
  f3, idx3 = findmin(abs.(Q3 .- frequencies), dims=3)
  f3tru, idx3tru = findmin(f3, dims=1)
  good = f3tru .< reltol
  Q3 .= good .* frequencies[idx3][idx3tru] .+ .!good .* Q3
  @show Q3
  @show good
  @show frequencies[idx3][idx3tru]
  # amp3 = good .* amplitudes[idx3][idx3tru]

  # "Remove this" and all integer multiples from the result 
  # by setting those frequencies to gigantic. This will make sure 
  # we don't accidentally find them, however does not 
  # handle the cross terms yet.
  # Each column of Q3 corresponds to a chunk of 3-rows in frequencies 
  # multol = 1e-1
  j_vals = reshape(0:order, 1, 1, 1, order+1)  # (1, 1, 1, n_orders)

  ismul = any(
      abs.(frequencies .- j_vals .* Q3) .< abstol .||
      abs.(frequencies .+ j_vals .* Q3) .< abstol,
      dims=4
  )  # (3, n_particles, n_frequencies, 1)

  frequencies .+= dropdims(ismul, dims=4) .* 1e10

  # Do the next dominant frequency. We have to determine this
  # by checking which difference/amplitude is the largest of 1 and 2. Kill this and 
  # all integer multiples. In this case, we actually need to check if 
  # the result is "good" or not, because close to the end, we will 
  # only have Q3 +- integer multiples. If no good guess, then 
  # we can just assume the amplitude is zero 
  # CHECK:
  Q1 = reshape(view(Q, :, 1), 1, n_particles, 1)
  Q2 = reshape(view(Q, :, 2), 1, n_particles, 1)

  f1, idx1 = findmin(abs.(Q1 .- frequencies), dims=3)
  f1tru, idx1tru = findmin(f1, dims=1)

  f2, idx2 = findmin(abs.(Q2 .- frequencies), dims=3)
  f2tru, idx2tru = findmin(f2, dims=1)

  # Whichever is smaller, do that first (closer match):
  if first(f2tru) < first(f1tru)
      next = 2
      final = 1
      good = f2tru .< reltol
      Qnext = good .* frequencies[idx2][idx2tru] .+ .!good .* Q2
      amp2 = good .* amplitudes[idx2][idx2tru] # amp is zero if not good
      Q2 .= Qnext
      @show Q2
      @show good
      @show frequencies[idx2][idx2tru]
  else
      next = 1
      final = 2
      good = f1tru .< reltol # this may need fiddling
      Qnext = good .* frequencies[idx1][idx1tru]  .+ .!good .* Q1
      amp1 = good .* amplitudes[idx1][idx1tru] # amp is zero if not good
      Q1 .= Qnext
      @show Q1
      @show good
      @show frequencies[idx1][idx1tru]
  end
  # Remove the next one
  j_vals = reshape(0:order, 1, 1, 1, order+1, 1)        # (1,1,1,n_j,1)
  k_vals = reshape(0:order, 1, 1, 1, 1, order+1)        # (1,1,1,1,n_k)
  Q3_bc   = reshape(Q3,   1, n_particles, 1, 1, 1)      # already (1,n_p,1) -> add 2 more
  Qnext_bc = reshape(Qnext, 1, n_particles, 1, 1, 1)
  fv = reshape(frequencies, 3, n_particles, n_frequencies, 1, 1)

  ismul2 = any(
      abs.(fv .- j_vals .* Q3_bc .- k_vals .* Qnext_bc) .< abstol .||
      abs.(fv .- j_vals .* Q3_bc .+ k_vals .* Qnext_bc) .< abstol .||
      abs.(fv .+ j_vals .* Q3_bc .- k_vals .* Qnext_bc) .< abstol .||
      abs.(fv .+ j_vals .* Q3_bc .+ k_vals .* Qnext_bc) .< abstol,
      dims=(4,5)
  )  # (3, n_particles, n_frequencies, 1, 1)

  frequencies .+= dropdims(ismul2, dims=(4,5)) .* 1e10
  # Finally get our guess for the last. Same idea as before. Here however, 
  # if the tune is too close to any other
  Qfinal = reshape(view(Q, :, final), 1, n_particles, 1)
  ff, idxf = findmin(abs.(Qfinal .- frequencies), dims=3)
  fftru, idxftru = findmin(ff, dims=1)
  good = fftru .< reltol
  Qfinal .= good .* frequencies[idxf][idxftru] .+ .!good .* Qfinal
  if final == 1
      amp1 = good .* amplitudes[idxf][idxftru]
  else
      amp2 = good .* amplitudes[idxf][idxftru] # amp is zero if not good
  end
  @show Qfinal
  @show good
  @show frequencies[idxf][idxftru]

  y .=  hcat(
    permutedims(reinterpret(Float64, reshape(amp1, 1, n_particles)), (2,1)),
    permutedims(reinterpret(Float64, reshape(amp2, 1, n_particles)), (2,1)),
    #permutedims(reinterpret(Float64, reshape(amp3, 1, n_particles)), (2,1))
  )

  return y
end
=#
function symplectify(m)
  @assert size(m, 1) == size(m, 2) "Matrix must be square"
  @assert mod(size(m, 1), 2) == 0 "Matrix must have even number of rows"
  nd = div(size(m, 1), 2)

  inner_prod(a, b) = dot(a, S * b)

  function normalize_pair(e1, e2)
    f = inner_prod(e1, e2)
    e1 /= sqrt(abs(f))
    e2 /= (sign(f) * sqrt(abs(f)))
    return e1, e2
  end

  m_out = copy(m)
  for i in 1:nd
    # Symplectify this pair
    v, w = normalize_pair(m_out[:,2*i-1], m_out[:,2*i])
    m_out[:,2*i-1] = v
    m_out[:,2*i]    = w
    
    # Project out from all remaining pairs
    for j in i+1:nd
      vj = m_out[:,2*j-1]
      wj = m_out[:,2*j]
      
      vj = vj - inner_prod(vj, w)*v + inner_prod(vj, v)*w
      wj = wj - inner_prod(wj, w)*v + inner_prod(wj, v)*w
      
      m_out[:,2*j-1] = vj
      m_out[:,2*j]   = wj
    end
  end
  return m_out

end