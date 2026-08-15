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
  "dx_1"   => _dx     ,
  "dpx_1"  => _dpx    ,
  "dy_1"   => _dy     ,
  "dpy_1"  => _dpy    ,
  "zx"     => _zx     ,
  "zpx"    => _zpx    ,
  "zy"     => _zy     ,
  "zpy"    => _zpy    ,
  "nx"     => _nx     ,
  "ny"     => _ny     ,
  "nz"     => _nz     ,
  "n0x"     => _n0x   ,
  "n0y"     => _n0y   ,
  "n0z"     => _n0z   ,
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
  "w1a"    => _w1a    ,
  "w2a"    => _w2a    ,
  "w1b"    => _w1b    ,
  "w2b"    => _w2b    ,
  "w1"     => _w1     ,
  "w2"     => _w2     ,  
)

const _INVERTED_TWISS_FCN_MAP = Dict(value => key for (key, value) in _TWISS_FCN_MAP)

const _TWISS_COLUNIT_MAP = Dict{String,String}(
  "index"  => ""      ,
  "name"   => ""      ,
  "kind"   => ""      ,
  "s"      => "[m]"   ,    
  "beta1"  => "[m]"   ,
  "beta2"  => "[m]"   ,
  "alpha1" => "[1]"   ,
  "alpha2" => "[1]"   ,
  "phi1"   => "[2π]"  ,
  "phi2"   => "[2π]"  ,
  "phi3"   => "[2π]"  ,
  "slip"   => "[m]"   ,
  "x"      => "[m]"   ,
  "px"     => "[1]"   ,
  "y"      => "[m]"   ,
  "py"     => "[1]"   ,
  "z"      => "[m]"   ,
  "pz"     => "[1]"   ,
  "dx"     => "[m]"   ,
  "dpx"    => "[1]"   ,
  "dy"     => "[m]"   ,
  "dpy"    => "[1]"   ,
  "dx_1"   => "[m]"   ,
  "dpx_1"  => "[1]"   ,
  "dy_1"   => "[m]"   ,
  "dpy_1"  => "[1]"   ,
  "zx"     => "[1]"   ,
  "zpx"    => "[m⁻¹]" ,
  "zy"     => "[1]"   ,
  "zpy"    => "[m⁻¹]" ,
  "nx"     =>  ""     ,
  "ny"     =>  ""     ,
  "nz"     =>  ""     ,
  "n0x"     =>  ""    ,
  "n0y"     =>  ""    ,
  "n0z"     =>  ""    ,
  "N"      =>  ""  ,
  "Vi"     =>  ""  ,
  "c11"    =>  ""  ,
  "c12"    =>  ""  ,
  "c21"    =>  ""  ,
  "c22"    =>  ""  ,
  "gammac" =>  ""  ,
  "H1"     =>  ""  ,
  "H2"     =>  ""  ,
  "H3"     =>  ""  ,
  "B1"     =>  ""  ,
  "B2"     =>  ""  ,
  "B3"     =>  ""  ,
  "w1a"    =>  "[1]" ,
  "w2a"    =>  "[1]" ,
  "w1b"    =>  "[1]" ,
  "w2b"    =>  "[1]" ,
  "w1"     =>  "[1]" ,
  "w2"     =>  "[1]" ,
)

@inline function _chrom_derivative(cfcn, order, override, j, twi, cache, ::Val{as_tps}) where {as_tps}
  if !override && !iscoasting(twi)
    error("
      To compute d$(_INVERTED_TWISS_FCN_MAP[cfcn])_$(order), beam must be coasting (no longitudinal oscillations).
    ")
  end

  x = cfcn(j, twi, cache, Val{true}()) # as_tps=true !!!!!
  dord = no6(twi)
  if cfcn in (_x, _px, _y, _py, _n0x, _n0y, _n0z, _nx, _ny, _nz)
    if order > dord # these guys require order <= no6
      error("Chromatic order must be at least $order to compute d$(_INVERTED_TWISS_FCN_MAP[cfcn])_$(order)")
    end
  elseif cfcn in (_w1, _w2, _w1a, _w1b, _w2a, _w2b) # requires order < no6-1
    if order + 1 >= dord
      error("Chromatic order must be at least $(order+2) to compute d$(_INVERTED_TWISS_FCN_MAP[cfcn])_$(order)")
    end
  #elseif order >= dord # else require order < no6
  #  error("Chromatic order must be at least $(order+1) to compute d$(_INVERTED_TWISS_FCN_MAP[cfcn])_$(order)")
  end
  
  if !(TI.is_tps_type(typeof(x)) isa TI.IsTPSType)
    error("
      Chromatic derivative-getting is currently only compatible with scalar-valued outputs, and 
      $(_INVERTED_TWISS_FCN_MAP[cfcn]) is a matrix/map.

      Try setting the `twiss` keyword argument `as_taylor_series=true` and include \"$(_INVERTED_TWISS_FCN_MAP[cfcn])\" 
      in the `cols`. The desired chromatic derivative can then be extracted
    ")
  end
  if as_tps
    for _ in 1:order
      x = TI.deriv(x, 6)            
    end
    return x
  else
    mono = zeros(UInt8, 6)
    mono[end] = order
    return factorial(order-1)*TI.getm(x, mono)
  end
end

# things like RDTs and arbitrary-order derivatives must be handled specially

function _twiss_map(cols)
  fcn = Vector{Function}(undef, length(cols))
  unit = Vector{String}(undef, length(cols))
  for i in 1:length(cols)
    col = cols[i]
    fcn[i] = _twiss_map_fcn(col) #col in keys(_TWISS_FCN_MAP) ? _TWISS_FCN_MAP[col] : error()
    unit[i] = col in keys(_TWISS_COLUNIT_MAP) ? _TWISS_COLUNIT_MAP[col] : ""
  end
  return fcn, unit
end

function _twiss_map_fcn(col)
  if haskey(_TWISS_FCN_MAP, col)
    return _TWISS_FCN_MAP[col]
  end
  m = match(r"^h([0-9]{4,6})$", col)
  if !isnothing(m)
    mono = [parse(Int, c) for c in m.captures[1]]
    let mono=mono
      return (j, twi, cache, vas_tps) -> _h(j, twi, cache, vas_tps, mono)
    end
  end

  pattern = Regex("^d(" * join([escape_string.(keys(_TWISS_FCN_MAP)); "h([0-9]{4,6})"], "|") * ")(?:_([1-9]))?\$")
  m = match(pattern, col)
  if !isnothing(m)
    ccol = m.captures[1]
    if !isnothing(match(r"^h([0-9]{4,6})$", ccol) )
      order = isnothing(m.captures[3]) ? 1 : parse(Int, m.captures[3])
      mono = [parse(Int, c) for c in m.captures[2]]
      let mono=mono, order=order
        cfcn = (j, twi, cache, vas_tps) -> _h(j, twi, cache, vas_tps, mono)
        return (j, twi, cache, vas_tps) -> _chrom_derivative(cfcn, order, false, j, twi, cache, vas_tps)
      end
    end
    order = isnothing(m.captures[2]) ? 1 : parse(Int, m.captures[2])
    cfcn = _TWISS_FCN_MAP[ccol]
    # note unlike others we can always compute dnx_1, dnx_2, dnx_5, etc. when coasting
    override = (cfcn in (_nx, _ny, _nz))
    let cfcn=cfcn, order=order, override=override
      return (j, twi, cache, vas_tps) -> _chrom_derivative(cfcn, order, override, j, twi, cache, vas_tps)
    end
  end
  error("Unrecognized input col: $col")
end