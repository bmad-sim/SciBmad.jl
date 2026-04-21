module Experimental
using ..SciBmad, FundamentalFrequencies

function transverse_frequencies(
    bl,
    v0, 
    Q;
    multol=1e-4, 
    n_frequencies=20,
    n_turns=500,
    order=3,
    verbose=true,
    window_order=5,
  )
  n_particles = size(v0, 1)
  res = track(bl; v0, n_turns, verbose)
  coords = res.v
  data = reshape(permutedims(coords, (2,1,3)), n_particles*6, n_turns+1) 
  
  # Now this makes it x + im*px, etc:
  data = reinterpret(ComplexF64, data)
  # Remove the mean:
  data = data .- mean(data, dims=2)
  
  # Batched NAFF
  frequencies, amplitudes = naff(data, n_frequencies; window_order, warnings=false)


  # Select dominant frequency (longitudinal mode)
  f3, idx3 = findmin(abs.((view(Q, :, 3) .- frequencies) ./ view(Q[3], :, 3)) ./ abs.(amplitudes), dims=2)
  f3tru, idx3tru = findmin(reshape(f3, 3, n_particles), dims=1)
  good = f3tru .< 1/multol
  Q3 = good .* (reshape(frequencies[idx3], 3, n_particles)[idx3tru]) .+ .!good .* view(Q, :, 3)
  amp3 = good .* reshape(amplitudes[idx3], 3, n_particles)[idx3tru] # amp is zero if not good

  # "Remove this" and all integer multiples from the result 
  # by setting those frequencies to gigantic. This will make sure 
  # we don't accidentally find them, however does not 
  # handle the cross terms yet.
  # Each column of Q3 corresponds to a chunk of 3-rows in frequencies 
  # multol = 1e-1
  for i in 1:length(Q3)
      fv = view(frequencies, ((i-1)*3+1):((i-1)*3+3),:)
      for j in 0:order
          ismul = abs.(fv .- j .* Q3[i]) .< multol .|| abs.(fv .+ j .* Q3[i]) .< multol
          fv .+= (ismul .* 1e10) # Make gigantic if is multiple
      end
  end
end

function obj_val_jac!(y, jac, x, p)
    Q = p[1]
        
    multol = 1e-4
    n_frequencies = 20
    order = 3
    N_turns = 500
    dx = zero(x)
    n_particles = 1 + 2*length(x)

    # The coords struct should make it easy for SIMD-writing
    coords = zeros(n_particles, 6, N_turns+1)
    v = repeat([x[1] x[2] x[3] x[4] x[5] x[6]], outer=n_particles)

    # Set the finite difference
    dd = eps(eltype(x))^(1/3)
    for i in 1:length(x)
        dx[i] = max(dd * abs(x[i]), dd)
        v[2*i,i] += dx[i]
        v[2*i+1,i] -= dx[i]
    end
    
    b0 = Bunch(v, species=ring.species_ref, p_over_q_ref=ring.p_over_q_ref)
    coords[:,:,1] = b0.coords.v
    for i in 1:N_turns
        track!(b0, ring, use_explicit_SIMD=false)
        coords[:,:,i+1] = b0.coords.v
    end

    # NAFF it all together
    # This puts it into (n_particles*6) x N_turns matrix
    # HOWEVER THIS IS STORED AS X1, X2, PX1, PX2 ! 
    # so really we need to permutedims
    data = reshape(permutedims(coords, (2,1,3)),n_particles*6, N_turns+1)  #reshape(coords, n_particles*6, N_turns+1)
    # Now this makes it x + im*px, etc:
    data = reinterpret(ComplexF64, data)
    # Remove the mean:
    data = data .- mean(data, dims=2)
    
    # Batched NAFF
    frequencies, amplitudes = naff(data, n_frequencies, window_order=5, warnings=false)
    
    #return frequencies, amplitudes
    # Select dominant frequency (longitudinal mode)
    # Assume this guess is correct
    #return frequencies, amplitudes
    f3, idx3 = findmin(abs.((Q[3] .- frequencies) ./ Q[3]) ./ abs.(amplitudes), dims=2)
    f3tru, idx3tru = findmin(reshape(f3, 3, n_particles), dims=1)
    good = f3tru .< 1/multol
    Q3 = good .* (reshape(frequencies[idx3], 3, n_particles)[idx3tru]) .+ .!good .* Q[3]
    amp3 = good .* reshape(amplitudes[idx3], 3, n_particles)[idx3tru] # amp is zero if not good
    #Q3 = reshape(frequencies[idx3], 3, n_particles)[idx3tru]
    #amp3 = reshape(amplitudes[idx3], 3, n_particles)[idx3tru]
    Q[3] = first(Q3)

    # "Remove this" and all integer multiples from the result 
    # by setting those frequencies to gigantic. This will make sure 
    # we don't accidentally find them, however does not 
    # handle the cross terms yet.
    # Each column of Q3 corresponds to a chunk of 3-rows in frequencies 
    # multol = 1e-1
    for i in 1:length(Q3)
        fv = view(frequencies, ((i-1)*3+1):((i-1)*3+3),:)
        for j in 0:order
            ismul = abs.(fv .- j .* Q3[i]) .< multol .|| abs.(fv .+ j .* Q3[i]) .< multol
            fv .+= (ismul .* 1e10) # Make gigantic if is multiple
        end
    end

    # Do the next dominant frequency. We have to determine this
    # by checking which difference/amplitude is the largest of 1 and 2. Kill this and 
    # all integer multiples. In this case, we actually need to check if 
    # the result is "good" or not, because close to the end, we will 
    # only have Q3 +- integer multiples. If no good guess, then 
    # we can just assume the amplitude is zero 
    # CHECK:
    f1, idx1 = findmin(abs.((Q[1] .- frequencies) ./ Q[1]) ./ abs.(amplitudes), dims=2)
    f1tru, idx1tru = findmin(reshape(f1, 3, n_particles), dims=1)
    #@show hcat(f1tru', (reshape(frequencies[idx1], 3, n_particles)[idx1tru])')
    
    f2, idx2 = findmin(abs.((Q[2] .- frequencies) ./ Q[2]) ./ abs.(amplitudes), dims=2)
    f2tru, idx2tru = findmin(reshape(f2, 3, n_particles), dims=1)
    #@show hcat(f2tru', (reshape(frequencies[idx2], 3, n_particles)[idx2tru])')

    # Whichever is smaller, do that first (closer match):
    if first(f2tru) < first(f1tru)
        next = 2
        final = 1
        good = f2tru .< 1/multol
        Qnext = good .* (reshape(frequencies[idx2], 3, n_particles)[idx2tru]) .+ .!good .* Q[2]
        amp2 = good .* reshape(amplitudes[idx2], 3, n_particles)[idx2tru] # amp is zero if not good
        Q[2] = first(Qnext)
        #@show Q[2]
    else
        next = 1
        final = 2
        good = f1tru .< 1/multol # this may need fiddling
        Qnext = good .* (reshape(frequencies[idx1], 3, n_particles)[idx1tru]) .+ .!good .* Q[1]
        amp1 = good .* reshape(amplitudes[idx1], 3, n_particles)[idx1tru] # amp is zero if not good
        Q[1] = first(Qnext)
        #@show Q[1]
    end

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
    
    #@show Q[1]

    # Finally get our guess for the last. Same idea as before. Here however, 
    # if the tune is too close to any other
    ff, idxf = findmin(abs.((Q[final] .- frequencies) ./ Q[final]) ./ abs.(amplitudes), dims=2)
    fftru, idxftru = findmin(reshape(ff, 3, n_particles), dims=1)
    good = fftru .< 1/multol
    Qf = good .* (reshape(frequencies[idxf], 3, n_particles)[idxftru]) .+ .!good .* Q[final]
    if final == 1
        amp1 = good .* reshape(amplitudes[idxf], 3, n_particles)[idxftru]
    else
        amp2 = good .* reshape(amplitudes[idxf], 3, n_particles)[idxftru] # amp is zero if not good
    end
    Q[final] = first(Qf)
    
    # val:
    y .= [real(first(amp1)), imag(first(amp1)), real(first(amp2)), imag(first(amp2)), real(first(amp3)), imag(first(amp3))]
    
    # Now compute the Jacobian
    for i in 1:length(x) # column
        amp1p = amp1[1+(i-1)*2+1]
        amp2p = amp2[1+(i-1)*2+1]
        amp3p = amp3[1+(i-1)*2+1]
        amp1m = amp1[1+(i-1)*2+2]
        amp2m = amp2[1+(i-1)*2+2]
        amp3m = amp3[1+(i-1)*2+2]
        yp = [real(amp1p), imag(amp1p), real(amp2p), imag(amp2p), real(amp3p), imag(amp3p)]
        ym = [real(amp1m), imag(amp1m), real(amp2m), imag(amp2m), real(amp3m), imag(amp3m)]
        jac[:,i] .= (yp .- ym) ./ (2*dx[i])
    end
    @show norm(y)
    @show Q
    return y, jac
end



end