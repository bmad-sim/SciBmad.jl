# If periodic + coasting: slip
# if periodic + not coasting: phi3, slip
# If open: always phi3, slip



const _TWISS_FCN_MAP = Dict{String,Function}(
  "index"  => _index  ,
  "name"   => _name   ,
  "kind"   => _kind   ,
  "s"      => _s      ,    
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
  #"nx"     => _nx     ,
  #"ny"     => _ny     ,
  #"nz"     => _nz     ,
  #"n"      => _n      ,
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
  "B1"     => _B1     ,
  "B2"     => _B2     ,
  "B3"     => _B3     ,
)


const _TWISS_COLLABEL_MAP = Dict{String,String}(
  "index"  => "index"     ,
  "name"   => "name"      ,
  "kind"   => "kind"      ,
  "s"      => "s\n[m]"     ,    
  "beta1"  => "beta1\n[m]" ,
  "beta2"  => "beta2\n[m]" ,
  "alpha1" => "alpha1\n[1]",
  "alpha2" => "alpha2\n[1]",
  "phi1"   => "phi1\n[2π]" ,
  "phi2"   => "phi2\n[2π]" ,
  "phi3"   => "phi3\n[2π]" ,
  "slip"   => "slip\n[m]"  ,
  "x"      => "x\n[m]"     ,
  "px"     => "px\n[1]"    ,
  "y"      => "y\n[m]"     ,
  "py"     => "py\n[1]"    ,
  "z"      => "z\n[m]"     ,
  "pz"     => "pz\n[1]"    ,
  "dx"     => "dx\n[m]"    ,
  "dpx"    => "dpx\n[1]"   ,
  "dy"     => "dy\n[m]"    ,
  "dpy"    => "dpy\n[1]"   ,
  "zx"     => "zx\n[1]"    ,
  "zpx"    => "zpx\n[m⁻¹]" ,
  "zy"     => "zy\n[1]"    ,
  "zpy"    => "zpy\n[m⁻¹]" ,
  #"nx"     => _nx     ,
  #"ny"     => _ny     ,
  #"nz"     => _nz     ,
  #"n"      => _n      ,
  "N"      =>  "N"        ,
  "Vi"     =>  "Vi"       ,
  "c11"    =>  "c11"      ,
  "c12"    =>  "c12"      ,
  "c21"    =>  "c21"      ,
  "c22"    =>  "c22"      ,
  "gammac" =>  "gammac"   ,
  "H1"     =>  "H1"       ,
  "H2"     =>  "H2"       ,
  "H3"     =>  "H3"       ,
  "B1"     =>  "B1"       ,
  "B2"     =>  "B2"       ,
  "B3"     =>  "B3"       ,
)


# things like RDTs and arbitrary-order derivatives must be handled specially

function _twiss_map(cols)
  fcn = Vector{Function}(undef, length(cols))
  collabel = Vector{String}(undef, length(cols))
  for i in 1:length(cols)
    col = cols[i]
    fcn[i] = col in keys(_TWISS_FCN_MAP) ? _TWISS_FCN_MAP[col] : error()
    collabel[i] = col in keys(_TWISS_COLLABEL_MAP) ? _TWISS_COLLABEL_MAP[col] : error()
  end
  return fcn, collabel 
end