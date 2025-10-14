function track(
  bl::Beamline,
  v0::Union{AbstractMatrix,AbstractVector},
  q0::Union{AbstractMatrix,AbstractVector,UniformScaling,Nothing}=nothing;
  n_turns=1,
  save_every_n_turns=1,
)
  if v0 isa AbstractVector
    length(v0) == 6 || error("track accepts a n x 6 matrix of n particle coordinates,
                          or alternatively a single particle as a vector of length 6. 
                          Received a vector of length $(length(v))")
    v0 = reshape(v0, (1, 6))
  else
    size(v0, 2) == 6 || error("track accepts a n x 6 matrix of n particle coordinates,
                          or alternatively a single particle as a vector of length 6. 
                          Received a matrix of size $(size(v))")
  end

  if !isnothing(q0) && !(q0 isa UniformScaling)

  end
  N_particles = size(v0, 1)
  v = similar(v0)
  res = similar(v0, N_particles, save_every_n_turns+1, 6)
  v .= v0
  b0 = Bunch(v0)
  BTBL.check_bl_bunch!(bl, b0, false) # Do not notify
  res[:,1,:] .= b0.coords.v
  for i in 1:n_turns
    track!(b0, bl)
    if mod(i, save_every_n_turns) == 0
      res[:,Int(i/save_every_n_turns)+1,:] .= b0.coords.v
    end
  end
  return res
end

