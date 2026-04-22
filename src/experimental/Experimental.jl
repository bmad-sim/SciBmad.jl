module Experimental
using ..SciBmad, FundamentalFrequencies, Statistics, LinearAlgebra
import NonlinearNormalForm: S

function transverse_frequencies(
    bl,
    v0, # DI.Cache
    Q;  # DI.Cache
    multol=1e-4, 
    n_frequencies=20,
    n_turns=500,
    order=3,
    verbose=true,
    window_order=5,
  )
  #@show Q
  #@show v0
  n_particles = size(v0, 1)
  res = track(bl; v0=copy(v0), n_turns, verbose, use_explicit_SIMD=false)
  coords = res.v
  data = reshape(permutedims(coords, (2,1,3)), 6*n_particles, n_turns+1) 
#@show data
  # Now this makes it x + im*px, etc:
  data = reinterpret(ComplexF64, data)
#@show data
  # Remove the mean:
  data = data .- mean(data, dims=2) 
  @show data
  # Batched NAFF
  frequencies, amplitudes = naff(data, n_frequencies; window_order, warnings=false)
  @show frequencies
  frequencies = reshape(frequencies, 3, n_particles, n_frequencies)
  amplitudes = reshape(amplitudes, 3, n_particles, n_frequencies)
  Q3 = reshape(view(Q, :, 3), 1, n_particles, 1)

  # Select dominant frequency (longitudinal mode)
  f3, idx3 = findmin(abs.((Q3 .- frequencies) ./ Q3) ./ abs.(amplitudes), dims=3)
  f3tru, idx3tru = findmin(f3, dims=1)
  good = f3tru .< 1/multol
  Q3 .= good .* frequencies[idx3][idx3tru] .+ .!good .* Q3
  amp3 = good .* amplitudes[idx3][idx3tru]

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
  )  # → (3, n_particles, n_frequencies, 1)

  frequencies .+= dropdims(ismul, dims=4) .* 1e10
#=
  for i in 1:length(Q3) #TODO: VECTORIZE THIS!
      fv = view(frequencies, ((i-1)*3+1):((i-1)*3+3),:)
      for j in 0:order
          ismul = abs.(fv .- j .* Q3[i]) .< multol .|| abs.(fv .+ j .* Q3[i]) .< multol
          fv .+= (ismul .* 1e10) # Make gigantic if is multiple
      end
  end
=#

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
  else
      next = 1
      final = 2
      good = f1tru .< 1/multol # this may need fiddling
      Qnext = good .* frequencies[idx1][idx1tru]  .+ .!good .* Q1
      amp1 = good .* amplitudes[idx1][idx1tru] # amp is zero if not good
      Q1 .= Qnext
  end

  # Remove the next one
  j_vals = reshape(0:order, 1, 1, 1, order+1, 1)        # (1,1,1,n_j,1)
  k_vals = reshape(0:order, 1, 1, 1, 1, order+1)        # (1,1,1,1,n_k)
  Q3_bc   = reshape(Q3,   1, n_particles, 1, 1, 1)      # already (1,n_p,1) → add 2 more
  Qnext_bc = reshape(Qnext, 1, n_particles, 1, 1, 1)
  fv = reshape(frequencies, 3, n_particles, n_frequencies, 1, 1)

  ismul2 = any(
      abs.(fv .- j_vals .* Q3_bc .- k_vals .* Qnext_bc) .< multol .||
      abs.(fv .- j_vals .* Q3_bc .+ k_vals .* Qnext_bc) .< multol .||
      abs.(fv .+ j_vals .* Q3_bc .- k_vals .* Qnext_bc) .< multol .||
      abs.(fv .+ j_vals .* Q3_bc .+ k_vals .* Qnext_bc) .< multol,
      dims=(4,5)
  )  # → (3, n_particles, n_frequencies, 1, 1)

  frequencies .+= dropdims(ismul2, dims=(4,5)) .* 1e10

#=
  # Remove the next one
  for i in 1:length(Q3)
      fv = view(frequencies, ((i-1)*3+1):((i-1)*3+3), :)
      for j in 0:order
          for k in 0:(order-j)
              ismul =  abs.(fv .- j .* Q3[i] .- k .* Qnext[i]) .< multol .|| 
                          abs.(fv .- j .* Q3[i] .+ k .* Qnext[i]) .< multol .|| 
                          abs.(fv .+ j .* Q3[i] .- k .* Qnext[i]) .< multol .|| 
                          abs.(fv .+ j .* Q3[i] .+ k .* Qnext[i]) .< multol 
              fv .+= (ismul .* 1e10) # Make gigantic if is multiple
          end
      end
  end
  =#

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
  
  y =  hcat(
    permutedims(reinterpret(Float64, reshape(amp1, 1, n_particles)), (2,1)),
    permutedims(reinterpret(Float64, reshape(amp2, 1, n_particles)), (2,1)),
    permutedims(reinterpret(Float64, reshape(amp3, 1, n_particles)), (2,1))
  )

  return y
end

function symplectify(m::AbstractMatrix)
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


end