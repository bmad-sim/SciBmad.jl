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

# This will return only the maximum one torus for a given delta/x/etc.
# uses bisection once limit is hit. Does not return A (found that 
# approximating next step with linear Ai doesn't help at all)
function max_one_torus(
  bl::Beamline, 
  mode,         # n_particles x 1, which mode?
  mode_coord,   # n_particles x 1, which coordinate of the mode? (1 or 2)
  stp,         # n_particles x 1, what initial step size?
  n_steps::Integer, # number of steps to do on ALL particles
  setter! = (p)->nothing, # Closure of beamline
  params...; # args to pass to setter
  verbose=true, 
  maxiter=10, 
  autodiff=AutoBatch(AutoFiniteDiff(fdjtype=Val(:central))),
  Q_guess = begin
    tunes = twiss(bl; at=[]).tunes;
    n_particles = size(mode, 1);
    q = similar(stp, n_particles, 3);
    copyto!(q, -reshape(repeat(tunes, inner=n_particles), n_particles, 3));
    q
  end,
  v=(zero(similar(stp, size(mode, 1), 4))),    # Initial 4-coordinate guess, n_particles x 4
  delta=zero(stp), # Initial delta/x/etc guess, n_particles x 1
  )
  if size(mode, 1) != size(mode_coord, 1)
    error("Invalid input sizes: size(mode, 1) must equal size(mode_coord, 1). 
      Received $(size(mode, 1)) and $(size(mode_coord, 1)) respectively.")
  end
  if size(stp, 1) != size(mode, 1)
    error("Invalid input sizes: size(stp, 1) must equal size(mode, 1). 
      Received $(size(stp, 1)) and $(size(mode, 1)) respectively.")
  end
  if size(mode_coord, 2) != 1
    error("size(mode_coord, 2) must equal 2")
  end
  if size(mode, 2) != 1
    error("size(mode, 2) must equal 1")
  end
  if size(stp, 2) != 1
    error("size(stp, 2) must equal 1")
  end

  n_particles = size(mode, 1)
  y = similar(v, n_particles, 4) # sol
  # Here we also want to store the previous solution that worked.
  # Then if the newton fails, we can go back to this.
  v_prev = similar(v, n_particles, 4)
  v_cache = similar(v, n_particles, 6)
  y .= 0
  v_prev .= v
  coord = similar(v, n_particles, 2) # contains both canonical conjugates of mode
  coord .= 0
  Q_guess_prev = similar(Q_guess)
  Q_guess_prev .= Q_guess

  # Solution:
  co = similar(v, n_particles, 6)
  co .= NaN
  if verbose
    println("Computing one tori...")
  end
  for i in 1:n_steps
    # Bump the coord and try to solve:
    @. coord[:,1] = ifelse(mode_coord == 1, delta+stp, 0)
    @. coord[:,2] = ifelse(mode_coord == 2, delta+stp, 0)
    sol = newton!(
      freq_res!, 
      y,
      v, 
      Cache(v_cache),
      Cache(Q_guess),
      Constant(bl),
      Constant(mode),
      Constant(coord),
      Constant(setter!),
      (Constant.(params))...; 
      abstol=1e-12, 
      reltol=1e-12,
      batchdim=1,
      autodiff=autodiff,
      maxiter=maxiter,
      verbose=verbose,
    )
    # For those that failed, we need to take a step back to v_prev
    fail = (@. sol.retcode != BatchSolve.RETCODE_SUCCESS) .| any(isnan, sol.u; dims=2)
    # if success, update v_prev:
    @. Q_guess_prev = ifelse(fail, Q_guess_prev, Q_guess)
    @. v_prev = ifelse(fail, v_prev, v)
    @. delta = ifelse(fail, delta, delta+stp)

    # If fail, reset to prev:
    @. v = ifelse(fail, v_prev, v)
    @. Q_guess = ifelse(fail, Q_guess_prev, Q_guess)
    @. stp = ifelse(fail, stp/2, stp) # If success, keep going forward
    
    # Set solution
    #=
    Dominant mode: 1
      a-mode = 2
      b-mode = 3
    Dominant mode: 2
      a-mode = 3
      b-mode = 1
    Dominant mode: 3
      a-mode = 1
      b-mode = 2
    =#

    if verbose
      print("\r", rpad("Finished iteration $i", 20))
      flush(stdout)
      println()
    end
  end

  # Now set the solution:
  m1 = mode .== 1
  m2 = mode .== 2
  mc1 = @. ifelse(mode_coord == 1, delta, 0)
  mc2 = @. ifelse(mode_coord == 2, delta, 0)
  v1 = @view v[:,1]
  v2 = @view v[:,2]
  v3 = @view v[:,3]
  v4 = @view v[:,4]
  @. co[:,1] = ifelse(m1, mc1, ifelse(m2, v3, v1))
  @. co[:,2] = ifelse(m1, mc2, ifelse(m2, v4, v2))
  @. co[:,3] = ifelse(m1, v1,  ifelse(m2, mc1, v3))
  @. co[:,4] = ifelse(m1, v2,  ifelse(m2, mc2, v4))
  @. co[:,5] = ifelse(m1, v3,  ifelse(m2, v1,  mc1))
  @. co[:,6] = ifelse(m1, v4,  ifelse(m2, v2,  mc2))
  return co, Q_guess
end

# deltas was n_particles x n_deltas array

# We want to generalize this to allow also z inputs
# So it would be n_particles x 2 x n_coord_points (since deltas are slowest moving)

# Last step is to just vectorize this function now
# It should keep going up to the max delta 
function walk_one_torus(
  bl::Beamline, 
  mode,         # n_particles x 1
  mode_coords,  # n_particles x 2 x n_coord_pts 
  setter! = (p)->nothing, # Closure of beamline
  params...; # args to pass to setter
  verbose=true, 
  as=nothing, 
  maxiter=10, 
  autodiff=AutoBatch(AutoFiniteDiff(fdjtype=Val(:central))),
  Q_guess = begin
    tunes = twiss(bl; at=[]).tunes;
    n_particles = size(mode, 1);
    q = similar(mode_coords, n_particles, 3);
    copyto!(q, -reshape(repeat(tunes, inner=n_particles), n_particles, 3));
    q
  end,
  v0=nothing, # Initial guess
  )
  if ndims(mode_coords) != 3
    error("mode_coords must be of size n_particles x 2 x n_coord_pts")
  end
  if size(mode, 1) != size(mode_coords, 1)
    error("Invalid input sizes: size(mode, 1) must equal size(mode_coords, 1). 
      Received $(size(mode, 1)) and $(size(mode_coords, 1)) respectively.")
  end
  if size(mode_coords, 2) != 2
    error("size(mode_coords, 2) must equal 2")
  end
  if size(mode, 2) != 1
    error("size(mode, 2) must equal 1")
  end
  n_coord_pts = size(mode_coords, 3)   
  n_particles = size(mode, 1)
  y = similar(mode_coords, n_particles, 4) # sol
  v = similar(mode_coords, n_particles, 4) # guess, will change for each delta
  v_cache = similar(v, n_particles, 6)
  co = similar(v, n_particles, 6, n_coord_pts)
  if !isnothing(as)
    if size(as) != (n_particles, 4, 4, n_coord_pts)
      error("Invalid size for `as` keyword argument. Received $(size(as)), requires $((n_particles, 4, 4, n_coord_points))")
    else
      as .= 0
    end
  end
  co .= NaN
  y .= 0
  v .= 0 # Assume closed orbit is zero orbit for all, in general should solve for all
  if !isnothing(v0)
    v .= v0
  end
  if verbose
    println("Computing one tori...")
  end
  for i in 1:n_coord_pts
    coord = view(mode_coords, :, :, i)
    sol = newton!(
      freq_res!, 
      y,
      v, 
      Cache(v_cache),
      Cache(Q_guess),
      Constant(bl),
      Constant(mode),
      Constant(coord),
      Constant(setter!),
      (Constant.(params))...; 
      abstol=1e-12, 
      reltol=1e-12,
      batchdim=1,
      autodiff=autodiff,
      maxiter=maxiter,
      verbose=verbose,
    )

    #@show Q_guess
    #@show v
    #@show y
    # This will act on each row (batch)
    @. v = ifelse(sol.retcode != BatchSolve.RETCODE_SUCCESS || isnan(sol.u), NaN, v) 
    if all(sol.retcode .!= BatchSolve.RETCODE_SUCCESS) # Then break the loop
      if verbose
        print("\r", rpad("All fail at iter $i", 20))
        flush(stdout)
        println()
      end
      break
    end
    #@show v
    # Jacobian will always be zero then
    # And Q_guess should always be bad

   
    # Set solution
    #=
    Dominant mode: 1
      a-mode = 2
      b-mode = 3
    Dominant mode: 2
      a-mode = 3
      b-mode = 1
    Dominant mode: 3
      a-mode = 1
      b-mode = 2
    =#
    m1 = mode .== 1
    m2 = mode .== 2
    mc1 = @view coord[:,1]
    mc2 = @view coord[:,2]
    v1 = @view sol.u[:,1]
    v2 = @view sol.u[:,2]
    v3 = @view sol.u[:,3]
    v4 = @view sol.u[:,4]
    @. co[:,1,i] = ifelse(m1, mc1, ifelse(m2, v3, v1))
    @. co[:,2,i] = ifelse(m1, mc2, ifelse(m2, v4, v2))
    @. co[:,3,i] = ifelse(m1, v1,  ifelse(m2, mc1, v3))
    @. co[:,4,i] = ifelse(m1, v2,  ifelse(m2, mc2, v4))
    @. co[:,5,i] = ifelse(m1, v3,  ifelse(m2, v1,  mc1))
    @. co[:,6,i] = ifelse(m1, v4,  ifelse(m2, v2,  mc2))

    if !isnothing(as)
      if delta == 0 # use twiss a for all inputs
        tw = twiss(bl; at=[first(bl.line)], normalizing_map=true)
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
      println()
    end
  end
  return co, Q_guess
end

function freq_res!(
    y,       # Output, n_particles x 4
    v,       # Input, n_particles x 4
    v_cache, # DI.Cache, n_particles x 6
    Q,       # DI.Cache, n_particles x 3
    bl,      # DI.Constant, Beamline
    mode,    # DI.Constant, n_particles x 1
    coord,   # DI.Constant, n_particles x 2
    setter! = (p)->nothing, # Closure of beamline, DI.Constant
    params...; # DI.Contexts
    n_frequencies=20,
    n_turns=500,
    order=3,
    verbose=true,
    window_order=5,
    reltol=eltype(v) == Float64 ? 0.005 : 0.005,#,=0.01,
    abstol=eltype(v) == Float64 ? 1e-5 : 1e-3, 
  )
  # Set lattice parameters
  setter!(params)

  n_particles = size(v, 1)
  v_cache .= 0
    
  #=
  Dominant mode: 1
    a-mode = 2
    b-mode = 3
  Dominant mode: 2
    a-mode = 3
    b-mode = 1
  Dominant mode: 3
    a-mode = 1
    b-mode = 2
  =#
  #mode = reshape(mode, 1, n_particles, 1)
  m1 = mode .== 1
  m2 = mode .== 2

  mc1 = @view coord[:,1]
  mc2 = @view coord[:,2]

  v1 = @view v[:,1]
  v2 = @view v[:,2]
  v3 = @view v[:,3]
  v4 = @view v[:,4]

  @. v_cache[:,1] = ifelse(m1, mc1, ifelse(m2, v3, v1))
  @. v_cache[:,2] = ifelse(m1, mc2, ifelse(m2, v4, v2))
  @. v_cache[:,3] = ifelse(m1, v1,  ifelse(m2, mc1, v3))
  @. v_cache[:,4] = ifelse(m1, v2,  ifelse(m2, mc2, v4))
  @. v_cache[:,5] = ifelse(m1, v3,  ifelse(m2, v1,  mc1))
  @. v_cache[:,6] = ifelse(m1, v4,  ifelse(m2, v2,  mc2))

  res = track(bl; v0=v_cache, n_turns, verbose, use_explicit_SIMD=false)
  coords = res.v
  data = reshape(permutedims(coords, (2,1,3)), 6*n_particles, n_turns+1) 

  # Now this makes it x + im*px, etc:
  data = reinterpret(complex(eltype(data)), data)

  # Remove the mean:
  data = data .- mean(data, dims=2) 

  # Batched NAFF
  frequencies, amplitudes = naff(data, n_frequencies; window_order, warnings=false)
  frequencies = reshape(frequencies, 3, n_particles, n_frequencies)
  amplitudes = reshape(amplitudes, 3, n_particles, n_frequencies)
  #@show frequencies[1:3,1,:]
  #@show frequencies[3,1,:]

  # Frequency views
  Q1 = reshape(view(Q, :, 1), 1, n_particles, 1)
  Q2 = reshape(view(Q, :, 2), 1, n_particles, 1)
  Q3 = reshape(view(Q, :, 3), 1, n_particles, 1)
  
  #=
  Dominant mode: 1
    a-mode = 2
    b-mode = 3
  Dominant mode: 2
    a-mode = 3
    b-mode = 1
  Dominant mode: 3
    a-mode = 1
    b-mode = 2
  =#
  mode = reshape(mode, 1, n_particles, 1)
  a = @. mod1(mode+1, 3) 
  b = @. mod1(mode+2, 3)

  Qdom = @. ifelse(mode == 1, Q1, ifelse(mode == 2, Q2, Q3)) 
  Qa   = @. ifelse(a == 1, Q1, ifelse(a == 2, Q2, Q3)) 
  Qb   = @. ifelse(b == 1, Q1, ifelse(b == 2, Q2, Q3)) 


  # Select dominant frequency
  fdom, idxdom = findmin(abs.(Qdom .- frequencies) .* (1 .+ 1 ./ abs.(amplitudes)), dims=3)
  fdomtru, idxdomtru = findmin(fdom, dims=1)
  good = fdomtru ./ (1 .+ 1 ./ abs.(amplitudes[idxdom][idxdomtru])) .< reltol
  Qdom .= good .* frequencies[idxdom][idxdomtru] .+ .!good .* Qdom

  # Update tunes
  @. Q1 = ifelse(mode == 1, Qdom, Q1)
  @. Q2 = ifelse(mode == 2, Qdom, Q2)
  @. Q3 = ifelse(mode == 3, Qdom, Q3)

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
      abs.(frequencies .- j_vals .* Qdom) .< abstol .||
      abs.(frequencies .+ j_vals .* Qdom) .< abstol,
      dims=4
  )  # (3, n_particles, n_frequencies, 1)

  frequencies .+= dropdims(ismul, dims=4) .* 1e10

  # Do the next dominant frequency. We have to determine this
  # by checking which difference/amplitude is the largest of 1 and 2. Kill this and 
  # all integer multiples. In this case, we actually need to check if 
  # the result is "good" or not, because close to the end, we will 
  # only have Qdom +- integer multiples. If no good guess, then 
  # we can just assume the amplitude is zero 
  fa, idxa = findmin(abs.(Qa .- frequencies) .* (1 .+ 1 ./ abs.(amplitudes)), dims=3)
  fatru, idxatru = findmin(fa, dims=1)

  fb, idxb = findmin(abs.(Qb .- frequencies) .* (1 .+ 1 ./ abs.(amplitudes)), dims=3)
  fbtru, idxbtru = findmin(fb, dims=1)

  # Whichever is smaller (closer match), do that first:
  next = @. ifelse(fbtru < fatru, b, a)
  fnexttru = @. ifelse(next == b, fbtru, fatru)
  final = @. ifelse(next == b, a, b)
  idxnext = @. ifelse(next == b, idxb, idxa)
  idxnexttru = @. ifelse(next == b, idxbtru, idxatru)
  good = fnexttru ./ (1 .+ 1 ./ abs.(amplitudes[idxnext][idxnexttru])) .< reltol
  Qnext = good .* frequencies[idxnext][idxnexttru] .+ .!good .* (ifelse.(next .== b, Qb, Qa))
  ampnext =  good .* amplitudes[idxnext][idxnexttru] # amp is zero if not good

  #@show next
  #@show final
  #@show good

  # Update tunes
  @. Q1 = ifelse(next == 1, Qnext, Q1)
  @. Q2 = ifelse(next == 2, Qnext, Q2)
  @. Q3 = ifelse(next == 3, Qnext, Q3)

  
  #=
  Set output, n_particles x 4
  a-mode goes in cols 1:2, b-mode in cols 3:4
  a and b modes for different dominant modes:

  Dominant mode: 1
    a-mode = 2
    b-mode = 3
  Dominant mode: 2
    a-mode = 3
    b-mode = 1
  Dominant mode: 3
    a-mode = 1
    b-mode = 2

  As long as the selected dominant mode is the same each iteration (which it is), then 
  the mode writing will always be consistent each iteration.
  =#


  y[:,1:2] .= ifelse.(reshape(next .== a, n_particles, 1), reinterpret(real(eltype(ampnext)), reshape(ampnext, 1, n_particles))', view(y, :, 1:2))
  y[:,3:4] .= ifelse.(reshape(next .== b, n_particles, 1), reinterpret(real(eltype(ampnext)), reshape(ampnext, 1, n_particles))', view(y, :, 3:4))

  # Remove the next one
  j_vals = reshape(0:order, 1, 1, 1, order+1, 1)        # (1,1,1,n_j,1)
  k_vals = reshape(0:order, 1, 1, 1, 1, order+1)        # (1,1,1,1,n_k)
  Qdom_bc   = reshape(Qdom,   1, n_particles, 1, 1, 1)      # already (1,n_p,1) -> add 2 more
  Qnext_bc = reshape(Qnext, 1, n_particles, 1, 1, 1)
  fv = reshape(frequencies, 3, n_particles, n_frequencies, 1, 1)

  # Compute each combination separately, accumulate with |=
  # This keeps each broadcast result concretely Bool
  ismul2 = abs.(fv .- j_vals .* Qdom_bc .- k_vals .* Qnext_bc) .< abstol
  ismul2 .|= abs.(fv .- j_vals .* Qdom_bc .+ k_vals .* Qnext_bc) .< abstol
  ismul2 .|= abs.(fv .+ j_vals .* Qdom_bc .- k_vals .* Qnext_bc) .< abstol
  ismul2 .|= abs.(fv .+ j_vals .* Qdom_bc .+ k_vals .* Qnext_bc) .< abstol# (3, n_particles, n_frequencies, order+1, order+1)
  ismul2 = any(any(ismul2, dims=5), dims=4)

  # Remove:
  frequencies .+= dropdims(ismul2, dims=(4,5)) .* 1e10

  # Finally get our guess for the last. Same idea as before. Here however, 
  # if the tune is too close to any other
  Qfinal = @. ifelse(final == 1, Q1, ifelse(final == 2, Q2, Q3)) #reshape(view(Q, :, final), 1, n_particles, 1)
  ffinal, idxfinal = findmin(abs.(Qfinal .- frequencies) .* (1 .+ 1 ./ abs.(amplitudes)), dims=3)
  ffinaltru, idxfinaltru = findmin(ffinal, dims=1)
  good = ffinaltru ./ (1 .+ 1 ./ abs.(amplitudes[idxfinal][idxfinaltru])) .< reltol
  Qfinal .= good .* frequencies[idxfinal][idxfinaltru] .+ .!good .* Qfinal
  ampfinal =  good .* amplitudes[idxfinal][idxfinaltru] 
  # Update tunes
  @. Q1 = ifelse(final == 1, Qfinal, Q1)
  @. Q2 = ifelse(final == 2, Qfinal, Q2)
  @. Q3 = ifelse(final == 3, Qfinal, Q3)

  #@show good

  # Set output, n_particles x 4
  y[:,1:2] .= ifelse.(reshape(final .== a, n_particles, 1), reinterpret(real(eltype(ampfinal)), reshape(ampfinal, 1, n_particles))', view(y, :, 1:2))
  y[:,3:4] .= ifelse.(reshape(final .== b, n_particles, 1), reinterpret(real(eltype(ampfinal)), reshape(ampfinal, 1, n_particles))', view(y, :, 3:4))

  return y
end

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