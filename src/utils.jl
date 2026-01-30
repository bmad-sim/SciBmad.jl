
"""
    Given an N x 4 matrix quaternions `q` (e.g., from the `Bunch` struct),
    and an N x 3 matrix of initial spin 3-vectors `si`, rotates the spin 
    3-vectors by its corresponding row in the quaternion matrix and fills 
    the N x 3 matrix `sf` with the result.
"""
function rotate_spins!(sf::AbstractMatrix, q::AbstractMatrix, si::AbstractMatrix)
  if !(size(sf, 1) == size(si, 1) == size(q, 1))
    error("Invalid sizes: different number of rows received for each matrix")
  end
  if size(q, 2) != 4
    error("Invalid size for quaternions matrix, number of columns must be 4")
  end
  if size(sf, 2) != 3
    error("Invalid size for final spin matrix, number of columns must be 3")   
  end
  if size(si, 2) != 3
    error("Invalid size for initial spin matrix, number of columns must be 3")   
  end
  Base.require_one_based_indexing(sf, q, si)
  N = size(sf, 1)
  @simd for i in 1:N
    @inbounds begin
      a = q[i,1]; bx = q[i,2]; by = q[i,3]; bz = q[i,4]
      vx = si[i,1]; vy = si[i,2]; vz = si[i,3]

      # t = 2 * (b × v)
      tx = 2 * (by*vz - bz*vy)
      ty = 2 * (bz*vx - bx*vz)
      tz = 2 * (bx*vy - by*vx)

      # v' = v + a*t + b×t
      sf[i,1] = vx + a*tx + (by*tz - bz*ty)
      sf[i,2] = vy + a*ty + (bz*tx - bx*tz)
      sf[i,3] = vz + a*tz + (bx*ty - by*tx)
    end
  end
  return sf
end

"""
    Given an N x 4 matrix quaternions `q` (e.g., from the `Bunch` struct),
    and an N x 3 matrix of initial spin 3-vectors `si`, rotates the spin 
    3-vectors by its corresponding row in the quaternion matrix and returns 
    a new spin matrix containing the final spin 3-vectors.
"""
function rotate_spins(q::AbstractMatrix, si::AbstractMatrix)
  sf = zero(si)
  rotate_spins!(sf, q, si)
  return sf
end