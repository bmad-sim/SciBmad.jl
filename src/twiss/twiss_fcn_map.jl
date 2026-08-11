# If periodic + coasting: slip
# if periodic + not coasting: phi3, slip
# If open: always phi3, slip

const _TWISS_BARE_FCN_MAP = Dict{String,Function}(
  "beta1"  => _beta1  ,
  "beta2"  => _beta2  ,
  "alpha1" => _alpha1 ,
  "alpha2" => _alpha2 ,
  "phi1"   => _phi1   ,
  "phi2"   => _phi2   ,
  "phi3"   => _phi3   ,
  "slip"   => _slip   ,
  "x"      => _x      ,
  "px"     => _px     ,
  "y"      => _y      ,
  "py"     => _py     ,
  "z"      => _z      ,
  "pz"     => _pz     ,
  "dx"     => _dx     ,
  "dpx"    => _dpx    ,
  "dy"     => _dy     ,
  "dpy"    => _dpy    ,
  "zx"     => _zx     ,
  "zpx"    => _zpx    ,
  "zy"     => _zy     ,
  "zpy"    => _zpy    ,
  "nx"     => _nx     ,
  "ny"     => _ny     ,
  "nz"     => _nz     ,
  "n"      => _n      ,
  "N"      => _N      ,
  "Vi"     => _Vi     ,
  "c11"    => _c11    ,
  "c12"    => _c12    ,
  "c21"    => _c21    ,
  "c22"    => _c22    ,
  "gammac" => _gammac ,
  "H1"     => _H1     ,
  "H2"     => _H2     ,
  "H3"     => _H3     ,
)

# things like RDTs and arbitrary-order derivatives must be handled specially

function _twiss_map_fcn(str)
  if str in keys(_TWISS_BARE_FCN_MAP)
    return _TWISS_BARE_FCN_MAP[str]
  else
error(  )
  end
  #=
  elseif str[1] == "d" # derivative

  end
  =#

end