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

  track_spin_vector = false
  if q0 isa UniformScaling
    q0 = repeat([1.0 0.0 0.0 0.0], size(v0, 1))
  elseif q0 isa AbstractVector
    length(q0) in (3,4) || error("q0 must be a vector of length 3 or 4, a uniform scaling, 
                          a matrix with 3 or 4 columns, or nothing.
                          Received a vector of length $(length(q0))")
    if length(q0) == 3
      track_spin_vector = true
      q0 = reshape(q0, (1, 3))
    else
      q0 = reshape(q0, (1, 4))
    end
  elseif q0 isa AbstractMatrix
    size(q0, 2) in (3,4) || error("q0 must be a matrix with 3 or 4 columns, a uniform scaling, or nothing.
                          Received a matrix of size $(size(q0))")
    size(q0, 1) == size(v0, 1) || error("q0 must have the same number of rows as v0.
                          Received a q0 of size $(size(q0)) and v0 of size $(size(v0))")
    track_spin_vector = (size(q0, 2) == 3)
  end

  N_particles = size(v0, 1)
  if isnothing(q0)
    res = similar(v0, N_particles, div(n_turns, save_every_n_turns)+1, 6)
  elseif track_spin_vector
    res = similar(hcat(v0, q0), N_particles, div(n_turns, save_every_n_turns)+1, 9)
  else
    res = similar(hcat(v0, q0), N_particles, div(n_turns, save_every_n_turns)+1, 10)
  end
  if track_spin_vector
    b0 = Bunch(v0, repeat([1.0 0.0 0.0 0.0], size(v0, 1)))
  else
    b0 = Bunch(v0, q0)
  end
  BTBL.check_bl_bunch!(bl, b0, false) # Do not notify
  res[:,1,1:6] .= b0.coords.v
  if track_spin_vector
    res[:,1,7:9] .= q0
  elseif !isnothing(q0)
    res[:,1,7:10] .= b0.coords.q
  end
  for i in 1:n_turns
    track!(b0, bl)
    if mod(i, save_every_n_turns) == 0
      res[:,Int(i/save_every_n_turns)+1,1:6] .= b0.coords.v
      if track_spin_vector
        res[:,Int(i/save_every_n_turns)+1,7:9] .= rotate(q0, b0.coords.q)
      elseif !isnothing(q0)
        res[:,Int(i/save_every_n_turns)+1,7:10] .= b0.coords.q
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