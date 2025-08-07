function track(
  v0::Union{AbstractMatrix,AbstractVector},
  bl::Beamline;
  N_turns=1,
)
  if v0 isa AbstractVector
    length(v0) == 6 || error("Track accepts a N x 6 matrix of N particle coordinates,
                          or alternatively a single particle as a vector. Received 
                          a vector of length $(length(v))")
    v0 = reshape(v0, (1, 6))
  end
  N_particles = size(v0, 1)
  b0 = Bunch(v0)
  BTBL.check_bl_bunch!(bl, b0, false) # Do not notify
  res = Array{eltype(v0),3}(undef, N_particles, N_turns+1, 6)
  res[:,1,:] .= b0.coords.v
  for i in 1:N_turns
    track!(b0, bl)
    res[:,i+1,:] .= b0.coords.v
  end
  return res
end