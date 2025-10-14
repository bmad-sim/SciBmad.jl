# No spin tracking
function track(
  bl::Beamline,
  v0::Union{AbstractMatrix,AbstractVector},
  q0::Nothing=nothing;
  n_turns=1,
  save_every_n_turns=1,
)
  # Input sanity checks:
  if v0 isa AbstractVector
    length(v0) == 6 || error("track accepts a n x 6 matrix of n particle coordinates,
                          or alternatively a single particle as a vector of length 6. 
                          Received a vector of length $(length(v))")
    n_particles = 1
  else
    size(v0, 2) == 6 || error("track accepts a n x 6 matrix of n particle coordinates,
                          or alternatively a single particle as a vector of length 6. 
                          Received a matrix of size $(size(v))")
    n_particles = size(v0, 1)
  end

  res = similar(v0, n_particles, div(n_turns, save_every_n_turns)+1, 6)
  b0 = Bunch(copy(v0))
  BTBL.check_bl_bunch!(bl, b0, false) # Do not notify
  res[:,1,1:6] .= b0.coords.v
  for i in 1:n_turns
    track!(b0, bl)
    if mod(i, save_every_n_turns) == 0
      res[:,div(i,save_every_n_turns)+1,1:6] .= b0.coords.v
    end
  end
  return res
end

# Spin tracking
function track(
  bl::Beamline,
  v0::Union{AbstractMatrix,AbstractVector},
  q0::Union{AbstractMatrix,AbstractVector,UniformScaling};
  n_turns=1,
  save_every_n_turns=1,
)
  # Input sanity checks:
  if v0 isa AbstractVector
    length(v0) == 6 || error("track accepts a n x 6 matrix of n particle coordinates,
                          or alternatively a single particle as a vector of length 6. 
                          Received a vector of length $(length(v))")
    n_particles = 1
  else
    size(v0, 2) == 6 || error("track accepts a n x 6 matrix of n particle coordinates,
                          or alternatively a single particle as a vector of length 6. 
                          Received a matrix of size $(size(v))")
    n_particles = size(v0, 1)
  end

  store_spin_vector_only = false
  if q0 isa AbstractVector
    length(q0) in (3, 4) || error("Third positional argument in track must be one of the following:
                                  Uniform scaling (I) for identity quaternions for n particles, 
                                  n x 4 matrix for n particle quaternions, n x 3 matrix for n particle 
                                  spin 3-vectors, a vector of length 4 for single particle quaternion, 
                                  or a vector of length 3 for single particle spin 3-vector
                                  Received a vector of length $(length(q0))")
    eltype(v0) == eltype(q0) || error("Spin array has eltype $(eltype(q0)) but orbital has $(eltype(v0))")
    if length(q0) == 3
      store_spin_vector_only = true
    end
    q0 = reshape(q0, (1,length(q0)))
  elseif q0 isa AbstractMatrix
    size(q0, 2) in (3, 4) || error("Third positional argument in track must be one of the following:
                                  Uniform scaling (I) for identity quaternions for n particles, 
                                  n x 4 matrix for n particle quaternions, n x 3 matrix for n particle 
                                  spin 3-vectors, a vector of length 4 for single particle quaternion, 
                                  or a vector of length 3 for single particle spin 3-vector
                                  Received a matrix of size $(size(q0))")
    size(q0, 1) == size(v0, 1) || error("Third positional argument (spin) has $(size(q0,1)) rows (particles) but 
                                  second positional argument has $(size(v0,1)) rows (particles). These must be the same.")
    eltype(v0) == eltype(q0) || error("Spin array has eltype $(eltype(q0)) but orbital has $(eltype(v0))")
    if size(q0, 2) == 3
      store_spin_vector_only = true
    end
  end
  
  if store_spin_vector_only
    res = similar(v0, n_particles, div(n_turns, save_every_n_turns)+1, 9)
  else
    res = similar(v0, n_particles, div(n_turns, save_every_n_turns)+1, 10)
  end

  if q0 isa UniformScaling || store_spin_vector_only
    q = similar(v0, n_particles, 4)
    q .= 0
    q[:,1] .= 1
    b0 = Bunch(copy(v0), q)
  else
    b0 = Bunch(copy(v0), copy(q0))
  end
  BTBL.check_bl_bunch!(bl, b0, false) # Do not notify
  res[:,1,1:6] .= b0.coords.v
  if store_spin_vector_only
    res[:,1,7:9] .= q0
  else
    res[:,1,7:10] .= b0.coords.q
  end
  for i in 1:n_turns
    track!(b0, bl)
    if mod(i, save_every_n_turns) == 0
      res[:,div(i,save_every_n_turns)+1,1:6] .= b0.coords.v
      if store_spin_vector_only
        res[:,div(i,save_every_n_turns)+1,7:9] .= rotate(q0, b0.coords.q)
      else
        res[:,div(i,save_every_n_turns)+1,7:10] .= b0.coords.q
      end
    end
  end
  return res
end


"""
Rotates vector v by quaternion q.
"""
function rotate(v::Union{AbstractMatrix,AbstractVector}, q::Union{AbstractMatrix,AbstractVector})
  if ndims(v) == 1
    # q = (a,b)
    a  = q[1]; bx = q[2]; by = q[3]; bz = q[4]
    vx = v[1]; vy = v[2]; vz = v[3]

    # t = 2 * (b × v)
    tx = 2 * (by*vz - bz*vy)
    ty = 2 * (bz*vx - bx*vz)
    tz = 2 * (bx*vy - by*vx)

    # v' = v + a*t + b×t
    outx = vx + a*tx + (by*tz - bz*ty)
    outy = vy + a*ty + (bz*tx - bx*tz)
    outz = vz + a*tz + (bx*ty - by*tx)

    return SA[outx outy outz]
  else
    N = size(v, 1)
    out = similar(v)
    @inbounds for i in 1:N
        a  = q[i,1]; bx = q[i,2]; by = q[i,3]; bz = q[i,4]
        vx = v[i,1]; vy = v[i,2]; vz = v[i,3]

        tx = 2 * (by*vz - bz*vy)
        ty = 2 * (bz*vx - bx*vz)
        tz = 2 * (bx*vy - by*vx)

        out[i,1] = vx + a*tx + (by*tz - bz*ty)
        out[i,2] = vy + a*ty + (bz*tx - bx*tz)
        out[i,3] = vz + a*tz + (bx*ty - by*tx)
    end
    return out
  end
end