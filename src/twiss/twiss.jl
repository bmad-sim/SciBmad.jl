const TENG_EDWARDS_COAST = [   
  "beta1"  ,
  "phi1"   ,
  "dx"     ,
  "x"      ,
  "beta2"  ,
  "phi2"   ,
  "dy"     ,
  "y"      ,
  "slip"   ,
  "alpha1" ,
  "alpha2" ,
  "px"     ,
  "py"     ,
  "z"      ,
  "pz"     ,
  "dpx"    ,
  "dpy"    ,
  "gammac" ,
  "c11"    ,
  "c12"    ,
  "c21"    ,
  "c22"    ,
]

const TENG_EDWARDS = [   
  "beta1"  ,
  "phi1"   ,
  "dx"     ,
  "x"      ,
  "beta2"  ,
  "phi2"   ,
  "dy"     ,
  "y"      ,
  "slip"   ,
  "phi3"   ,
  "alpha1" ,
  "alpha2" ,
  "px"     ,
  "py"     ,
  "z"      ,
  "pz"     ,
  "dpx"    ,
  "dpy"    ,
  "gammac" ,
  "c11"    ,
  "c12"    ,
  "c21"    ,
  "c22"    ,
]

base_cols::Vector{String} = ["index", "name", "kind", "s"]

"""
    twiss(bl::Beamline; kwargs...)

General function to compute the (parametric and nonlinear) Twiss parameters of a `Beamline`. 
Returns a `Twiss`struct that contains the (amplitude-dependent) (spin) tunes/slip, and a 
dataframe of the Twiss parameters at each specified integration step. By default, lattice 
functions of the Sagan-Rubin/Edwards-Teng coupling formalism are computed in the dataframe 
at every integration step.

The `cols` keyword argument can be set to customize what quantities will be computed and 
stored as columns in the returned dataframe. E.g., 
```julia
tw = twiss(bl; cols=["beta1"])
```
will compute only the horizontal-like beta function. We can also specify so-called 
"resonance driving terms"/"detune coefficients" of the Bengtsson polynomial, similarly,
```julia
tw = twiss(bl; cols=["beta1", "h3000"], order=2)
```
where the total order of the `twiss` call has also been set appropriately to compute `h3000`.

An arbitrary order chromatic derivative of any scalar-valued column can be taken using the 
syntax `d<column>_<order>`. If `order==1`, then the `_1` can be omitted. E.g., the linear and 
2nd order dispersions, as well as ∂β₁/∂δ can be computed with
```julia
tw = twiss(bl; cols=["dx", "dx_2","dbeta1"], chrom=2)
```
where the `chrom` keyword argument, which specifies solely the truncation order of δ, has been 
set appropriately. 

For chromatic derivative calculation, you may need to set the keyword argument `rf_on=false`.

To customize the output, such as computing higher order (chromatic) lattice functions, spin 
lattice functions/tune, amplitude-dependent tunes, and parametric quantities, see the 
keyword arguments:

## Keyword arguments
- `at::Union{Colon, AbstractVector}`: A vector containing `LineElement`s, element beamline 
    indexes, and/or tuples of s-ranges specifying where to compute the Twiss parameters.
- `in_body_coordinates::Bool`: If `true` Twiss parameters inside misaligned elements are not
    transformed back to the reference curve coordinates. Default is `false`
- `rf_on::Bool`: If `false`, then any `RFParams` in any `LineElement`s are ignored for the 
    entirety of the Twiss tracking and calculation. Default is `true`
- `order::Integer`: TPSA truncation order for all phase space coordinates used in the Twiss 
    calculation, default is 1
- `chrom::Integer`: TPSA truncation order for only δ, default is `order`
- `GTPSA_descriptor::Union{Descriptor,Nothing}`: Custom GTPSA `Descriptor` to use for the Twiss 
    calculation, required to provide to do parametric normal form and lattice function 
    calculations. Default is `Descriptor([order, order, order, order, order, chrom], max(order,chrom))`.
- `a_initial::Union{DAMap,Nothing}`: A `DAMap` of the initial transformation from normal form 
    coordinates to laboratory coordinates, to compute "open" lattice Twiss functions. Default is 
    `nothing`, to compute the periodic ("closed") lattice Twiss functions
- `damping::Union{Bool,Nothing}`: Specifies if radiation damping is included. Default is `nothing`,
    which will auto-detect from `a_initial` or the periodic `a`.
- `delta0`: If the beam is coasting (no longitudinal oscillations), then the Twiss parameters will 
    be computed around this delta-dependent orbit. Else, this will be an initial guess for the 
    6D closed orbit calculation. Default is `0.`. If `a_initial` is provided then this will be ignored, 
    UNLESS `a_initial` is coasting, in which case this will set the value for δ. 
- `v0`: Initial guess for the closed orbit finder if `a_initial` is not provided, else setting this 
    will override the orbit expansion origin coordinates in `a_initial.v0`.
- `spin::Bool`: If `true`, spin is included in the Twiss calculation. Default is `false`.
- `base_cols`: A vector of "base columns", that will always be included in the Twiss dataframe regardless 
    of what is specified in `cols`. Defaults to the global variable `SciBmad.base_cols`, which defaults 
    to `["index", "name", "kind", "s"]`.
- `cols`: A vector of columns to include in the Twiss dataframe. See the extended help for all possible 
    columns.
- `as_taylor_series::Union{Bool,Nothing}`: If `true`, that the lattice functions in the dataframe are 
    returned as Taylor series types - this will be necessary if doiong parametric normal form. Default 
    is `nothing`, which autoselects to `false` if there are no parameters in the GTPSA, and `true` if 
    there are parameters.
- `symplectic_tol::Float64`: Tolerance of symplectic condition violation, above which radiation damping 
    is assumed. Default is `1e-8`.

To see a description of all quantities available to include in `cols`, see the extended help 
section using `??twiss`

# Extended help

Columns to include in `cols` are detailed below. As a reminder, an arbitrary order chromatic derivative 
of *any* scalar-valued output can be computed using the syntax "d<column>_<order" (e.g. "dx_1" is the 
linear dispersion).

- `index`  : Beamline index of the containing `LineElement`
- `name`   : Name of the containing `LineElement`
- `kind`   : Kind of the containing `LineElement`
- `s`      : s-position of the Twiss parameter calculation (may be inside of `LineElement`)
- `beta1`  : Horizontal-like beta function of the Sagan-Rubin/Edwards-Teng coupling formalism
- `beta2`  : Vertical-like beta function of the Sagan-Rubin/Edwards-Teng coupling formalism
- `alpha1` : Horizontal-like alpha function of the Sagan-Rubin/Edwards-Teng coupling formalism
- `alpha2` : Vertical-like alpha function of the Sagan-Rubin/Edwards-Teng coupling formalism
- `phi1`   : Horizontal-like phase advance in units of [2π]
- `phi2`   : Vertical-like phase advance in units of [2π]
- `phi3`   : Longitudinal-like phase advance in units of [2π]
- `slip`   : Slip factor in units cΔt [m]
- `x`      : Orbit canonical x coordinate
- `px`     : Orbit canonical px coordinate
- `y`      : Orbit canonical y coordinate
- `py`     : Orbit canonical py coordinate
- `z`      : Orbit canonical z coordinate
- `pz`     : Orbit canonize pz coordinate
- `zx`     : Adiabatic dependence of x on initial longitudinal position/time (AKA "crab dispersion") 
- `zpx`    : Adiabatic dependence of px on initial longitudinal position/time (AKA "crab dispersion")
- `zy`     : Adiabatic dependence of y on initial longitudinal position/time (AKA "crab dispersion")
- `zpy`    : Adiabatic dependence of py on initial longitudinal position/time (AKA "crab dispersion")
- `nx`     : Invariant spin field x-component (recommended to use with `as_taylor_series=true`)
- `ny`     : Invariant spin field y-component (recommended to use with `as_taylor_series=true`)
- `nz`     : Invariant spin field z-component (recommended to use with `as_taylor_series=true`)
- `n0x`    : Closed orbit periodic spin direction x-component
- `n0y`    : Closed orbit periodic spin direction y-component
- `n0z`    : Closed orbit periodic spin direction z-component
- `N`      : N matrix of the Sagan-Rubin coupling formalism
- `Vi`     : V⁻¹ matrix of the Sagan-Rubin coupling formalism
- `c11`    : [1,1] component of the Sagan-Rubin coupling matrix C
- `c12`    : [1,2] component of the Sagan-Rubin coupling matrix C
- `c21`    : [2,1] component of the Sagan-Rubin coupling matrix C
- `c22`    : [2,2] component of the Sagan-Rubin coupling matrix C
- `gammac` : Coupling factor of the Sagan-Rubin coupling formalism
- `w1a`    : Horizontal-like Montague function "a" component, requires `chrom > 1` and/or `order > 1`
- `w2a`    : Vertical-like Montague function "a" component, requires `chrom > 1` and/or `order > 1`
- `w1b`    : Horizontal-like Montague function "b" component, requires `chrom > 1` and/or `order > 1`
- `w2b`    : Vertical-like Montague function "b" component, requires `chrom > 1` and/or `order > 1`
- `w1`     : Horizontal-like Montague function, requires `chrom > 1` and/or `order > 1`
- `w2`     : Vertical-like Montague function, requires `chrom > 1` and/or `order > 1`
- `H1`     : De Moivre-Ripken H¹ matrix (see E. Forest, _From Tracking Code to Analysis_)
- `H2`     : De Moivre-Ripken H² matrix (see E. Forest, _From Tracking Code to Analysis_)
- `H3`     : De Moivre-Ripken H³ matrix (see E. Forest, _From Tracking Code to Analysis_)
- `B1`     : De Moivre-Ripken B¹ matrix (see E. Forest, _From Tracking Code to Analysis_)
- `B2`     : De Moivre-Ripken B² matrix (see E. Forest, _From Tracking Code to Analysis_)
- `B3`     : De Moivre-Ripken B³ matrix (see E. Forest, _From Tracking Code to Analysis_)
- `E1`     : De Moivre-Ripken E¹ matrix (see E. Forest, _From Tracking Code to Analysis_)
- `E2`     : De Moivre-Ripken E² matrix (see E. Forest, _From Tracking Code to Analysis_)
- `E3`     : De Moivre-Ripken E³ matrix (see E. Forest, _From Tracking Code to Analysis_)
- `a`      : (Nonlinear) map to transform from Floquet variables to laboratory variables
- `a0`     : (Nonlinear) map to transform from to the parameter-dependent fixed point
- `a1`     : Linear normalizing map around the parameter-dependent fixed point, including nonlinear parameter dependence
- `a2`     : Nonlinear part of the normalizing map, around the parameter-dependent fixed point
- `as`     : (Nonlinear) spin normalizing map
- `h<ijkl>`   : The "ijkl" resonance driving term/detune coefficient/Bengtsson monomial for the coasting beam case
- `h<ijklmn>` : The "ijklmn" resonance driving term/detune coefficient/Bengtsson monomial including longitudinal oscillations
"""
function twiss(
  bl::Beamline; 

  at::Union{Colon, AbstractVector}  = :,
  in_body_coordinates::Bool = false, 
  rf_on::Bool = true,

  # GTPSA truncation order sets:
  order::Integer = 1,
  chrom::Integer = order,
  GTPSA_descriptor::Union{Descriptor,Nothing} = nothing,

  #start::Union{Integer,LineElement,Nothing} = nothing, # TODO: Nothing means compute periodic  
  a_initial::Union{Nothing,DAMap}   = nothing, 
  damping::Union{Nothing,Bool} = nothing, # nothing = auto-detect from a_initial

  # Initial input, CO guess if periodic, initial orbit if not periodic
  delta0::Number=0.,
  v0::Matrix     = (if isnothing(a_initial)
        [0. 0. 0. 0. 0. delta0]
    else
      t = zeros(1,6)
      if length(a_initial.v0) == 5
        t[1:5] .= a_initial.v0
        t[6] = delta0
      else
        t .= a_initial.v0
      end
    end
  ), 

  # The lattice functions to compute
  spin::Bool      = isnothing(a_initial) ? false : !isnothing(a_initial.q),
  base_cols       = SciBmad.base_cols,
  cols            = nothing, # (de_moivre ? DE_MOIVRE : TENG_EDWARDS)..., (spin ? SPIN : Function[])...],\
  as_taylor_series::Union{Nothing,Bool} = nothing, # nothing = auto-select (if nn > 6, true, else false)

  symplectic_tol = 1e-8, # Tolerance below which to include damping
  )

  if isnothing(a_initial)
    v0_and_coast = co_and_coast(bl, v0, rf_on)
  else
    v0_and_coast = (v0, isodd(NNF.nvars(a_initial))) 
  end

  coast = v0_and_coast[2]

  if isnothing(cols)
    if coast
      cols = TENG_EDWARDS_COAST
    else
      cols = TENG_EDWARDS
    end
  end

  if spin
    spinregex = r"^(?:nx|ny|nz|n0x|n0y|n0z|d(?:nx|ny|nz|n0x|n0y|n0z)(?:_[1-9])?)$"
    if !any(x->occursin(spinregex, x), cols)
      # Then add default cols:
      cols = vcat(cols, ["n0x", "n0y", "n0z", "dnx", "dny", "dnz"])
    end
  end

  if isnothing(GTPSA_descriptor)
    storedesc = GTPSA.desc_current
    GTPSA_descriptor = Descriptor([order, order, order, order, order, chrom], max(order,chrom))
    GTPSA.desc_current = storedesc # Don't reset the global
  elseif chrom != 1 || order != 1
    @info "`GTPSA_descriptor` has been explicitly provided: ignoring `order`/`chrom` inputs"
  end

  if !coast && chrom != order
    error("""
    You specified `chrom`, but this beamline has synchrotron motion. Please turn off RF 
    cavities to get delta-dependent Twiss functions.
    """)
  end

  if !isnothing(a_initial) && GTPSA.getdesc(first(a_initial.v)) != GTPSA_descriptor
    error("Specified `GTPSA_descriptor` disagrees with that of `a_initial`")
  elseif !isnothing(a_initial)
    if spin && isnothing(a_initial.q)
      error("Unable to propagate spin: `a_initial` does not include spin")
    elseif !spin && !isnothing(a_initial)
      a_initial = DAMap(v0=a_initial.v0, v=a_initial.v, nv=NNF.nvars(a_initial), np=NNF.nparams(a_initial), s=a_initial.s)
    end
    GTPSA_descriptor = GTPSA.getdesc(first(a_initial.v))
  end

  init = TI.InitGTPSA{GTPSA.Dynamic,Descriptor}(; dynamic_descriptor=GTPSA_descriptor)

  # Check if output are TPSA in parameters (delta excluded)
  if isnothing(as_taylor_series)
    if TI.ndiffs(init) > 6
      as_taylor_series = true
    else
      as_taylor_series = false
    end
  end

  # Assemble locations. Note that start and end of the Beamline are ALWAYS included
  s, names, kinds, idxs, step_save, include_start, include_end = _twiss_assemble_locations(bl, at)
  beta_gamma_ref = Vector{Float64}(undef, length(s)) # Store the reference energy at each step
  t_ref = Vector{Float64}(undef, length(s)) # Store the reference time at each step

  # If the GTPSA truncation order is uniform, then we can 
  # cache the maps between saved points and concatenate them
  # In the closed case, we need to do one pass to compute a, 
  # and another pass to push a. If uniform, a can be pushed with cached maps
  # else, a is tracked again. Note that the first pass will tell you 
  # if there is damping or not for the rest of the Twiss.

  # So each element will have the output of factorise with canonise level
  # set accordingly. 
  maps = nothing
  r_and_tunes = nothing
  if isnothing(a_initial)
    # Determine:
    if _check_cachable(GTPSA_descriptor)
      # also fills beta_gamma_ref, t_ref
      a_initial, r_and_tunes, maps = _compute_periodic_a_and_cache!(bl, v0_and_coast[1], init, rf_on, Val{coast}(), Val{spin}(), step_save, beta_gamma_ref, t_ref, in_body_coordinates)
    else
      a_initial, r_and_tunes = _compute_periodic_a(bl, v0_and_coast[1], init, rf_on, Val{coast}(), Val{spin}())
    end
  end

  if isnothing(damping)
    damping = norm(NNF.checksymp(NNF.jacobian(a_initial))) > symplectic_tol
  end

  # Determine canonization level
  canonise, phase, damp = canonise_phase_damp(GTPSA_descriptor, coast, damping)

  a_initial = factorise(a_initial; canonise=canonise, damping=isnothing(damp) ? false : true).a

  # Now we push 
  if isnothing(maps)
    fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3 = _twiss_push_a!(bl, rf_on, step_save, a_initial, canonise, phase, damp, beta_gamma_ref, t_ref, in_body_coordinates)
  else
    fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3 = _twiss_push_a_with_cache(maps, step_save, a_initial, canonise, phase, damp)
  end

  twi = TwissInternal(s, names, kinds, idxs, beta_gamma_ref, t_ref, fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3, r_and_tunes)

  # Finally, construct the summ and the dataframe (with provided columns)
  # And post-process with the provided columns
  # Need to do one row first then can construct the DataFrame
  df, cache = _twiss_df(vcat(base_cols, cols), twi, include_start, include_end, Val{as_taylor_series}())

  # q1, q2, slip factor eta_c, momentum compaction alpha_c, [q3, qspin]
  # only thing is with 3d motion, need to compute 
  summ = _twiss_summ(twi, cache)
  return Twiss(summ, df)
end

function co_and_coast(bl, v0, rf_on)
  co_sol = find_closed_orbit(bl; v0=v0, batch=Val{false}(), rf_on)
  if co_sol.sol.retcode != RETCODE_SUCCESS
    error("Closed orbit finder did not converge.")
  end
  return (co_sol.v0, co_sol.coasting_beam)
end

_twiss_assemble_locations(bl::Beamline, ::Colon) = _twiss_assemble_locations(bl, [(0., Inf)])

function _twiss_assemble_locations(bl::Beamline, at::Vector)
  at_idxs = filter(x->x isa Integer, at)
  at_eles = filter(x->x isa LineElement, at)
  at_ranges = filter(x->x isa Tuple, at)

  if any(x->x[1] > x[2], at_ranges)
    t = at_ranges[findfirst(x->x[1] > x[2], at_ranges)]
    error("Invalid s range ($(t[1]),$(t[2])): start index must be <= end index")
  end
  
  stmp = Vector{Any}(undef, 0)
  names = Vector{String}(undef, 0)
  kinds = Vector{String}(undef, 0)
  idxs = Vector{Int}(undef, 0)
  step_save = Vector{Int}(undef, 0)

  # As a guess assume length equal to number of beamline elements + 1
  # This makes the typical Twiss case hopefully faster
  n_ele = length(bl.line)
  sizehint!(stmp, n_ele+1)
  sizehint!(names, n_ele+1)
  sizehint!(kinds, n_ele+1)
  sizehint!(idxs, n_ele+1)
  sizehint!(step_save, n_ele)

  scur = 0f0
  step_cur = 0
  for ele in bl.line
    idx = ((ele.BeamlineParams)::BeamlineParams).beamline_index
    up = (ele.UniversalParams)::UniversalParams
    name = up.name
    kind = up.kind
    tm = up.tracking_method
    L = up.L
    n_steps, ds_step = BeamTracking.find_steps(tm, L)

    # Check which steps are inside any of the ranges
    found = false
    for _ in 1:n_steps
      if any(x -> x[1] <= scur < x[2], at_ranges)
        push!(stmp, scur)
        push!(names, name)
        push!(kinds, kind)
        push!(idxs, idx)
        push!(step_save, step_cur)
        found = true
      end
      step_cur += 1
      scur += ds_step
    end
    
    # If not in an s-range, check if explicitly provided
    # Always include first element
    if !found && (idx == 1 || (any(x -> x == idx, at_idxs) || any(at_eles) do x
          x == ele || (haskey(getfield(ele, :pdict), InheritParams) ? x == (getfield(ele, :pdict)[InheritParams].parent) : false)
        end
        ))
        push!(stmp, scur - ds_step*n_steps)
        push!(names, name)
        push!(kinds, kind)
        push!(idxs, idx)
        push!(step_save, step_cur-n_steps)
    end
  end

  # Always store the last step
  push!(stmp, scur)
  push!(names, "END")
  push!(kinds, "-"^3)
  push!(idxs, -1)
  push!(step_save, step_cur)

  # Now resolve type of s:
  s = typeof(scur).(stmp)

  include_start = any(x->x[1]<=0<=x[2], at_ranges) || any(x->x==1, at_idxs) || any(at_eles) do x
    x == bl.line[1] || (haskey(getfield(bl.line[1], :pdict), InheritParams) ? x == (getfield(bl.line[1], :pdict)[InheritParams].parent) : false)
  end
  include_end = any(x->x[1]<=scur<=x[2], at_ranges) || any(x->x==length(bl.line), at_idxs) || any(at_eles) do x
    x == bl.line[end] || (haskey(getfield(bl.line[end], :pdict), InheritParams) ? x == (getfield(bl.line[end], :pdict)[InheritParams].parent) : false)
  end

  return s, names, kinds, idxs, step_save, include_start, include_end
end

function _check_cachable(GTPSA_descriptor)
  # check if we can cache_and_concat:
  desc = unsafe_load(GTPSA_descriptor.desc)
  nn = desc.nn
  mo = desc.mo
  po = desc.po
  if all(x->x == mo && (po == 0 || x == po), unsafe_wrap(Vector{UInt8}, desc.no, nn))
    return true
  else
    return false
  end
end

function _twiss_make_identity(v0, init, ::Val{coast}, ::Val{spin}) where {coast,spin}
  nn = TI.ndiffs(init)
  nv = 6
  np = nn-nv
  if coast
    nv -= 1
    np += 1
  end
  return DAMap(init=init, nv=nv, np=np, v0=view(v0, :, 1:nv), v_matrix=I, q=(spin ? I : nothing))
end

function _twiss_setmap!(map, coords)
  nv = NNF.nvars(map)
  NNF.setray!(map.v, v=reshape(coords.v, :))
  
  # Reset coords back to identity
  # This should not touch delta if e.g. delta-dependent twiss:
  for i in 1:nv
    TI.clear!(coords.v[i])
  end
  NNF.setray!(view(coords.v, 1:nv), scalar=NNF.getscalar(map), v_matrix=I)

  # Handle spin too:
  if !isnothing(map.q)
    NNF.setquat!(map.q, q=reshape(coords.q, :))
    for i in 1:4
      TI.clear!(coords.q[i])
    end
    TI.seti!(coords.q[1], 1, 0)
  end
  return map
end

function _twiss_track!(eye, cbs, bl, rf_on)
  if NNF.nvars(eye) == 5
    v = reshape([(i < 5 ? eye.v0[i]+copy(eye.v[i]) : copy(eye.v[i])) for i in 1:6], 1, 6)
  else
    v = reshape([eye.v0[i]+copy(eye.v[i]) for i in 1:6], 1, 6)
  end
  q = isnothing(eye.q) ? nothing : [copy(eye.q[1]) copy(eye.q[2]) copy(eye.q[3]) copy(eye.q[4])]
  b0 = Bunch(v=v, q=q, callbacks=cbs)
  BTBL.check_bl_bunch!(b0, bl, false) # Do not notify
  track!(b0, bl; rf_on)
  return b0
end

function _a_r_tunes(m::DAMap)
  mo = NNF.maxord(m)
  a = normal(m)
  c = c_map(m) # Transform to phasor basis
  r = inv(c) ∘ inv(a) ∘ m ∘ a ∘ c
  # Need to cut highest order
  Q_x = -cutord(angle(NNF.factor_out(r.v[1], 1))/(2*pi), mo)
  Q_y = -cutord(angle(NNF.factor_out(r.v[3], 3))/(2*pi), mo)
  if NNF.nvars(m) == 5
    Q_s = real(r.v[5])
    TI.seti!(Q_s, 0, 5) # subtract time identity
  else
    Q_s = -cutord(angle(NNF.factor_out(r.v[5], 5))/(2*pi), mo)
  end
  if isnothing(m.q)
    return a, r, SA[Q_x, Q_y, Q_s]
  else
    Q_spin = -atan(real(r.q.q2), real(r.q.q0))/pi # not two pi bc quaternion
    return a, r, SA[Q_x, Q_y, Q_s, Q_spin]
  end
end

function _compute_periodic_a(bl::Beamline, v0, init, rf_on, ::Val{coast}, ::Val{spin}) where {coast, spin}
  eye = _twiss_make_identity(v0, init, Val{coast}(), Val{spin}())
  b0 = _twiss_track!(eye, (), bl, rf_on)
  _twiss_setmap!(eye, b0.coords)
  a, r, tunes = _a_r_tunes(eye)
  return a, (r, tunes)
end

function _twiss_cache_preallocate(step_save, map::T) where {T<:DAMap}
  maps = Vector{T}(undef, length(step_save))
  for i in 1:length(step_save)
    if i == 1 && step_save[1] == 0
      maps[1] = one(map)
      NNF.setscalar!(maps[1], map.v0)
    else
      maps[i] = zero(map) # Preallocate
    end
  end
  return maps
end

function _twiss_cache_make_callback(_step_save, _beta_gamma_ref, _t_ref, _in_body_coordinates, _maps)
  # Note: need to handle the first element differently
  if first(_step_save) == 0
    _cur_step_save_idx = 2
  else
    _cur_step_save_idx = 1
  end
  let step_save=_step_save, maps=_maps, curstep=Ref{Int}(0), cur_step_save_idx=Ref{Int}(_cur_step_save_idx), beta_gamma_ref=_beta_gamma_ref, t_ref=_t_ref, in_body_coordinates=_in_body_coordinates
    return (i, coords, cur_s, cur_t_ref, cur_beta_gamma_ref, last_ds_step, last_g, transforms_out!, transforms_in!) -> begin
      curstep[] += 1
      if cur_step_save_idx[] <= length(step_save) && curstep[] == step_save[cur_step_save_idx[]] # Store the current map
        map = maps[cur_step_save_idx[]]
        if !in_body_coordinates
          transforms_out!(i, coords, cur_s, cur_t_ref)
        end
        _twiss_setmap!(map, coords)
        beta_gamma_ref[cur_step_save_idx[]] = cur_beta_gamma_ref
        t_ref[cur_step_save_idx[]] = cur_t_ref
        if !in_body_coordinates
          transforms_in!(i, coords, cur_s, cur_t_ref)
        end
        cur_step_save_idx[] += 1
      end
    end
  end
end

function _compute_periodic_a_and_cache!(bl::Beamline, v0, init, rf_on, ::Val{coast}, ::Val{spin}, step_save, beta_gamma_ref, t_ref, in_body_coordinates) where {coast, spin}
  eye = _twiss_make_identity(v0, init, Val{coast}(), Val{spin}())
  maps = _twiss_cache_preallocate(step_save, eye)
  cb = _twiss_cache_make_callback(step_save, beta_gamma_ref, t_ref, in_body_coordinates, maps)
  _twiss_track!(eye, (cb,), bl, rf_on)
  m_turn = eye
  for map in maps
    m_turn = map ∘ m_turn
  end
  a, r, tunes = _a_r_tunes(m_turn)
  if first(step_save) == 0
    b0 = Bunch(v=zeros(0,6)) # empty bunch just to compute initial reference energy
    BTBL.check_bl_bunch!(b0, bl, false)
    beta_gamma_ref[1] = BeamTracking.R_to_beta_gamma(b0.species, b0.p_over_q_ref)
    t_ref[1] = 0
  end
  return a, (r, tunes), maps
end

function canonise_phase_damp(GTPSA_descriptor, coast, damping)
  desc = unsafe_load(GTPSA_descriptor.desc)
  mo = desc.mo
  no4 = unsafe_wrap(Vector{UInt8}, desc.no, 4)
  if mo == 1
    canonise = 0
    phase = MVector{3,Float64}(0,0,0)
    damp = damping ? MVector{3,Float64}(0,0,0) : nothing
  else
    if coast && no4 == SA[1,1,1,1]
      canonise = 1
    else
      canonise = 2
    end
    zer = TPS64(use=GTPSA_descriptor)
    phase = MVector{3,typeof(zer)}(zer, zero(zer), zero(zer))
    damp = damping ? MVector{3,typeof(zer)}(zero(zer), zero(zer), zero(zer)) : nothing
  end
  return canonise, phase, damp
end

function _store_twiss!(fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3, a, canonise, phase, damp, j)
  damping = !isnothing(damp)
  facj = factorise(a; canonise=canonise, phase=phase, damp=damp, damping=damping)
  fac[j] = facj
  phi1[j] = j != 1 ? copy(phase[1]) : zero(phase[1])
  phi2[j] = j != 1 ? copy(phase[2]) : zero(phase[2])
  phi3_or_slip[j] = j != 1 ? copy(phase[3]) : zero(phase[3])
  if damping
    damp1[j] = j != 1 ? copy(damp[1]) : zero(damp[1])
    damp2[j] = j != 1 ? copy(damp[2]) : zero(damp[2])
    damp3[j] = j != 1 ? copy(damp[3]) : zero(damp[3])
  end
  return
end

function _twiss_make_callback(_step_save, initial_step_save_idx, _in_body_coordinates, _map, _fac, _canonise, _phase, _phi1, _phi2, _phi3_or_slip, _damp, _damp1, _damp2, _damp3, _beta_gamma_ref, _t_ref)
  # stupid let block bc the compiler is very stupid:
  let step_save=_step_save, in_body_coordinates=_in_body_coordinates, fac=_fac, canonise=_canonise, phase=_phase, 
    phi1=_phi1, phi2=_phi2, phi3_or_slip=_phi3_or_slip, damp=_damp, damp1=_damp1, damp2=_damp2, damp3=_damp3,
    curstep=curstep=Ref{Int}(0), cur_step_save_idx=Ref{Int}(initial_step_save_idx), map=_map, beta_gamma_ref=_beta_gamma_ref, t_ref=_t_ref
    
    return (i, coords, cur_s, cur_t_ref, cur_beta_gamma_ref, last_ds_step, last_g, transforms_out!, transforms_in!) -> begin
      curstep[] += 1
      j = cur_step_save_idx[]
      if j <= length(step_save) && curstep[] == step_save[j]
        if !in_body_coordinates
          transforms_out!(i, coords, cur_s, cur_t_ref)
        end
        _twiss_setmap!(map, coords)
        _store_twiss!(fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3, map, canonise, phase, damp, j)
        # Reset coords with canonised a:
        aj = fac[j].a
        beta_gamma_ref[j] = cur_beta_gamma_ref
        t_ref[j] = cur_t_ref
        for k in 1:6
          TI.copy!(coords.v[k], aj.v[k])
        end
        if !isnothing(map.q)
          for k in 1:4
            TI.copy!(coords.q[k], aj.q[k])
          end
        end
        if !in_body_coordinates
          transforms_in!(i, coords, cur_s, cur_t_ref)
        end
        cur_step_save_idx[] += 1
      end
    end
  end
end

function _twiss_make_base_columns(n, a::T, phase, damp) where {T}
  if !isnothing(a.q)
    fac = Vector{@NamedTuple{as::T, a0::T, a1::T, a2::T, a::T, r::T}}(undef, n)
  else
    fac = Vector{@NamedTuple{a0::T, a1::T, a2::T, a::T, r::T}}(undef, n)
  end
  phi1 = Vector{eltype(phase)}(undef, n)
  phi2 = Vector{eltype(phase)}(undef, n)
  phi3_or_slip = Vector{eltype(phase)}(undef, n) 
  if !isnothing(damp)
    damp1 = Vector{eltype(damp)}(undef, n) 
    damp2 = Vector{eltype(damp)}(undef, n) 
    damp3 = Vector{eltype(damp)}(undef, n) 
  else
    damp1 = nothing
    damp2 = nothing
    damp3 = nothing
  end
  return fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3
end

function _twiss_push_a!(bl, rf_on, step_save, a_initial, canonise, phase, damp, beta_gamma_ref, t_ref, in_body_coordinates)
  fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3 = _twiss_make_base_columns(length(step_save), a_initial, phase, damp)
  # Have to treat 0 specially:
  if first(step_save) == 0
    _store_twiss!(fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3, a_initial, canonise, phase, damp, 1) 
    a_initial = fac[1].a
    b0 = Bunch(v=zeros(0,6)) # empty bunch just to compute initial reference energy
    BTBL.check_bl_bunch!(b0, bl, false)
    beta_gamma_ref[1] = BeamTracking.R_to_beta_gamma(b0.species, b0.p_over_q_ref)
    t_ref[1] = 0
    initial_step_save_idx = 2
  else
    initial_step_save_idx = 1
  end
  cb = _twiss_make_callback(step_save, initial_step_save_idx, in_body_coordinates, a_initial, fac, canonise, phase, phi1, phi2, phi3_or_slip, damp, damp1, damp2, damp3, beta_gamma_ref, t_ref)
  _twiss_track!(a_initial, (cb,), bl, rf_on)
  return fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3
end

function _twiss_push_a_with_cache(maps, step_save, a_initial, canonise, phase, damp)
  fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3 = _twiss_make_base_columns(length(step_save), a_initial, phase, damp)
  a = a_initial
  # Have to treat 0 specially:
  if first(step_save) == 0
    _store_twiss!(fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3, a, canonise, phase, damp, 1) 
    a = fac[1].a
  end
  # Now push around, note end is always included
  for j in 1:length(maps)
    map = maps[j]
    a = map ∘ a
    _store_twiss!(fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3, a, canonise, phase, damp, j) 
    a = fac[j].a
  end
  return fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3
end

function _twiss_summ(twi, cache)
  j = length(twi.s)
  q1 = _phi1(j, twi, cache, Val{true}())
  coast = iscoasting(twi)
  oper = x -> (TI.is_tps_type(typeof(x)) isa TI.IsTPSType ? AmplitudeDependentValue(x, coast) : x)
  summ = LittleDict{Symbol,Union{Float64,AmplitudeDependentValue}}()
  summ[:q1] = oper(q1)
  summ[:q2] = oper(_phi2(j, twi, cache, Val{true}()))

  if !iscoasting(twi)
    summ[:q3] = oper(_phi3(j, twi, cache, Val{true}()))
  end

  etac = -_slip(j, twi, cache, Val{true}()) / (C_LIGHT * twi.t_ref[end])
  alphac = -_z_slip(j, twi, cache, Val{true}()) / twi.s[end]
  if coast && TI.is_tps_type(typeof(etac)) isa TI.IsTPSType
    etac = TI.deriv(etac, 6)
  end
  if coast && TI.is_tps_type(typeof(alphac)) isa TI.IsTPSType
    alphac = TI.deriv(alphac, 6)
  end
  summ[:etac] = oper(etac)
  summ[:alphac] = oper(alphac)

  if length(twi.r_and_tunes[2]) == 4
    qspin = twi.r_and_tunes[2][end]
    if !(TI.is_tps_type(typeof(q1)) isa TI.IsTPSType) && !coast
      qspin = scalar(qspin)
    else
      qspin = AmplitudeDependentValue(qspin, coast)
    end
    summ[:qspin] = qspin
  end
  

  if !isnothing(twi.damp1)
    summ[:damp1] = oper(twi.damp1[end])
    summ[:damp2] = oper(twi.damp2[end])
    summ[:damp3] = oper(twi.damp3[end])
  end
  return TwissSummary(summ)
end
