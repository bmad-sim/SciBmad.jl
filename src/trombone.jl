
"""
    phase_trombone!(ele::LineElement; phi1=0, phi2=0, phi3=0)

Makes `ele` into a phase trombone - a matrix expanded around the _current_ closed orbit 
(at the time of calling this function) which applies a linear rotation in normal form 
coordinates (which are also computed at the time of calling this function). Useful for 
creating zero-length elements where the phase advance can be tuned without impacting the 
Twiss parameters/linear optics. The `LineElement` must be in a `Beamline` in order to 
use this function.

The `LineElement` property `transport_map` is filled with the corresponding phase trombone 
transport map, and the property `transport_map_params` is set to `(phi1, phi2, phi3)`.

Spin tracking is NOT currently compatible with phase trombones.
  
!!! warn
    If the closed orbit and/or Twiss parameters at `ele` are changed *after* calling 
    this function, then the phase trombone map is no longer valid.
"""
function phase_trombone!(ele::LineElement; phi1=0, phi2=0, phi3=0)
  bl = ele.beamline
  tw = twiss(bl, base_cols=["index"], cols=["a", "x", "px", "y", "py", "z", "pz"], at=[ele])
  a = SMatrix{6,6}(tw.a[1])
  ai = inv(a)
  ele.transport_map = _make_trombone_map(a, ai, tw.x[1], tw.px[1], tw.y[1], tw.py[1], tw.z[1], tw.pz[1])
  ele.transport_map_params = (2*pi*phi1, 2*pi*phi2, 2*pi*phi3)
  return ele
end

function _make_trombone_map(_a, _ai, _x, _px, _y, _py, _z, _pz)
  let a=_a, ai=_ai, x=_x, px=_px, y=_y, py=_py, z=_z, pz=_pz
    return (v, q::Nothing, p) -> begin
      nu1 = p[1]
      nu2 = p[2]
      nu3 = p[3]
      @FastGTPSA begin
        v1n = ai[1,1]*(v[1] - x) + ai[1,2]*(v[2] - px) + ai[1,3]*(v[3] - y) + ai[1,4]*(v[4] - py) + ai[1,5]*(v[5] - z) + ai[1,6]*(v[6] - pz)
        v2n = ai[2,1]*(v[1] - x) + ai[2,2]*(v[2] - px) + ai[2,3]*(v[3] - y) + ai[2,4]*(v[4] - py) + ai[2,5]*(v[5] - z) + ai[2,6]*(v[6] - pz)
        v3n = ai[3,1]*(v[1] - x) + ai[3,2]*(v[2] - px) + ai[3,3]*(v[3] - y) + ai[3,4]*(v[4] - py) + ai[3,5]*(v[5] - z) + ai[3,6]*(v[6] - pz)
        v4n = ai[4,1]*(v[1] - x) + ai[4,2]*(v[2] - px) + ai[4,3]*(v[3] - y) + ai[4,4]*(v[4] - py) + ai[4,5]*(v[5] - z) + ai[4,6]*(v[6] - pz)
        v5n = ai[5,1]*(v[1] - x) + ai[5,2]*(v[2] - px) + ai[5,3]*(v[3] - y) + ai[5,4]*(v[4] - py) + ai[5,5]*(v[5] - z) + ai[5,6]*(v[6] - pz)
        v6n = ai[6,1]*(v[1] - x) + ai[6,2]*(v[2] - px) + ai[6,3]*(v[3] - y) + ai[6,4]*(v[4] - py) + ai[6,5]*(v[5] - z) + ai[6,6]*(v[6] - pz)
        
        v1nn =  cos(nu1)*v1n + sin(nu1)*v2n
        v2nn = -sin(nu1)*v1n + cos(nu1)*v2n
        v3nn =  cos(nu2)*v3n + sin(nu2)*v4n
        v4nn = -sin(nu2)*v3n + cos(nu2)*v4n
        v5nn =  cos(nu3)*v5n + sin(nu3)*v6n
        v6nn = -sin(nu3)*v5n + cos(nu3)*v6n

        v1 = a[1,1]*v1nn + a[1,2]*v2nn + a[1,3]*v3nn + a[1,4]*v4nn + a[1,5]*v5nn + a[1,6]*v6nn + x
        v2 = a[2,1]*v1nn + a[2,2]*v2nn + a[2,3]*v3nn + a[2,4]*v4nn + a[2,5]*v5nn + a[2,6]*v6nn + px
        v3 = a[3,1]*v1nn + a[3,2]*v2nn + a[3,3]*v3nn + a[3,4]*v4nn + a[3,5]*v5nn + a[3,6]*v6nn + y
        v4 = a[4,1]*v1nn + a[4,2]*v2nn + a[4,3]*v3nn + a[4,4]*v4nn + a[4,5]*v5nn + a[4,6]*v6nn + py
        v5 = a[5,1]*v1nn + a[5,2]*v2nn + a[5,3]*v3nn + a[5,4]*v4nn + a[5,5]*v5nn + a[5,6]*v6nn + z
        v6 = a[6,1]*v1nn + a[6,2]*v2nn + a[6,3]*v3nn + a[6,4]*v4nn + a[6,5]*v5nn + a[6,6]*v6nn + pz
      end
      return ((v1, v2, v3, v4, v5, v6), q)  
    end
  end
end