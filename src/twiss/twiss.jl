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
  "dampdec1",
  "dampdec2",
  "dampdec3",
]

base_cols::Vector{String} = ["index", "name", "kind", "s"]

"""
    twiss(bl::Beamline; kwargs...)

General function to compute the (parametric and nonlinear) Twiss parameters of a `Beamline`. 
Returns a `Twiss` struct that contains the (amplitude-dependent) (spin) tunes/slip, and a 
dataframe of the Twiss parameters at each specified integration step. By default, lattice 
functions of the Sagan-Rubin/Edwards-Teng coupling formalism are computed in the dataframe 
at every integration step.

The `cols` keyword argument can be set to customize what quantities will be computed and 
stored as columns in the returned dataframe. E.g., 
```julia
tw = twiss(bl; cols=["beta1"])
```
will compute only the horizontal-like beta function. We can also specify so-called 
"resonance driving terms"/"detune coefficients" of the nonlinear effective Hamiltonian,
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
- `at::Union{Colon, AbstractVector}`: A vector containing `LatElement`s, element beamline 
    indexes, and/or tuples of s-ranges specifying where to compute the Twiss parameters.
- `in_body_coordinates::Bool`: If `true` Twiss parameters inside misaligned elements are not
    transformed back to the reference curve coordinates. Default is `false`
- `rf_on::Bool`: If `false`, then any `RFParams` in any `LatElement`s are ignored for the 
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
- `spin::Bool`: If `true`, spin is included in the Twiss calculation. Default is `false`, or `true` if 
    `a_initial` includes spin.
- `base_cols`: A vector of "base columns", that will always be included in the Twiss dataframe regardless 
    of what is specified in `cols`. Defaults to the global variable `SciBmad.base_cols`, which defaults 
    to `["index", "name", "kind", "s"]`.
- `cols`: A vector of columns to include in the Twiss dataframe. See the extended help for all possible 
    columns.
- `as_taylor_series::Union{Bool,Nothing}`: If `true`, the lattice functions in the dataframe are 
    returned as Taylor series types - this will be necessary if doing parametric normal form. Default 
    is `nothing`, which autoselects to `false` if there are no parameters in the GTPSA, and `true` if 
    there are parameters.
- `coast_tol`: Tolerance for the first-order delta dependences on each of the phase space variables below 
    which coasting beam is assumed, passed to `find_closed_orbit`. Default is `1e-14`.
- `symplectic_tol::Float64`: Tolerance of symplectic condition violation, above which radiation damping 
    is assumed. Default is `1e-8`.
- `batch_index::Int`: If any element parameters are `BatchParam`, sets which index in the batch the `twiss`
    should be evaluated at. Default is `1`.

To see a description of all quantities available to include in `cols`, see the extended help 
section using `??twiss`

# Extended help

Columns to include in `cols` are detailed below. As a reminder, an arbitrary order chromatic derivative 
of *any* scalar-valued output can be computed using the syntax "d<column>_<order>" (e.g. "dx_1" is the 
linear dispersion).

- `index`  : Beamline index of the containing `LatElement`
- `name`   : Name of the containing `LatElement`
- `kind`   : Kind of the containing `LatElement`
- `s`      : s-position of the Twiss parameter calculation (may be inside of `LatElement`)
- `beta1`  : Horizontal-like beta function of the Sagan-Rubin/Edwards-Teng coupling formalism
- `beta2`  : Vertical-like beta function of the Sagan-Rubin/Edwards-Teng coupling formalism
- `alpha1` : Horizontal-like alpha function of the Sagan-Rubin/Edwards-Teng coupling formalism
- `alpha2` : Vertical-like alpha function of the Sagan-Rubin/Edwards-Teng coupling formalism
- `phi1`   : Horizontal-like phase advance in units of [2π]
- `phi2`   : Vertical-like phase advance in units of [2π]
- `phi3`   : Longitudinal-like phase advance in units of [2π]
- `dampdec1`: Horizontal-like damping decrement
- `dampdec2`: Vertical-like damping decrement
- `dampdec3`: Longitudinal-like damping decrement
- `slip`   : Slip factor in units cΔt [m]
- `x`      : Orbit canonical x coordinate
- `px`     : Orbit canonical px coordinate
- `y`      : Orbit canonical y coordinate
- `py`     : Orbit canonical py coordinate
- `z`      : Orbit canonical z coordinate
- `pz`     : Orbit canonical pz coordinate
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
- `a0`     : (Nonlinear) map to transform to the parameter-dependent fixed point
- `a1`     : Linear normalizing map around the parameter-dependent fixed point, including nonlinear parameter dependence
- `a2`     : Nonlinear part of the normalizing map, around the parameter-dependent fixed point
- `as`     : (Nonlinear) spin normalizing map
- `h<ijkl>`   : The "ijkl" resonance driving term/detune coefficient/nonlinear effective Hamiltonian monomial for the coasting beam case
- `h<ijklmn>` : The "ijklmn" resonance driving term/detune coefficient/nonlinear effective Hamiltonian monomial including longitudinal oscillations
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

  #start::Union{Integer,LatElement,Nothing} = nothing, # TODO: Nothing means compute periodic  
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
        t[:] .= a_initial.v0
      end
      t
    end
  ), 

  # The lattice functions to compute
  spin::Bool      = isnothing(a_initial) ? false : !isnothing(a_initial.q),
  base_cols       = SciBmad.base_cols,
  cols            = nothing, # (de_moivre ? DE_MOIVRE : TENG_EDWARDS)..., (spin ? SPIN : Function[])...],\
  as_taylor_series::Union{Nothing,Bool} = nothing, # nothing = auto-select (if nn > 6, true, else false)

  coast_tol = 1e-14, 
  symplectic_tol = 1e-8, # Tolerance below which to include damping
  batch_index::Int=1,
  )

  if isnothing(a_initial)
    v0_and_coast = co_and_coast(bl, v0, rf_on, coast_tol, batch_index)
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

  if !isnothing(a_initial)
    if !isnothing(GTPSA_descriptor) && GTPSA.getdesc(first(a_initial.v)) != GTPSA_descriptor
      error("Specified `GTPSA_descriptor` disagrees with that of `a_initial`")
    end
    if spin && isnothing(a_initial.q)
      error("Unable to propagate spin: `a_initial` does not include spin")
    elseif !spin && !isnothing(a_initial)
      a_initial = DAMap(v0=a_initial.v0, v=a_initial.v, nv=NNF.nvars(a_initial), np=NNF.nparams(a_initial), s=a_initial.s)
    end
    GTPSA_descriptor = GTPSA.getdesc(first(a_initial.v))
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
  t_ref = Vector{eltype(s)}(undef, length(s)) # Store the reference time at each step (type of the lengths, e.g. TPS for parametric lengths)

  # The maps are tracked with track, which saves them at the rows (the steps step_save) with the
  # reference there. If the GTPSA truncation order is uniform, the identity is tracked once: the
  # maps from the start to the rows, the last one the one-turn map, give a and its value at every
  # row (a composed with them). Else a is computed from the one-turn map, then tracked itself.
  locations = (s, idxs, names, kinds)
  maps = nothing
  r_and_tunes = nothing
  tr = nothing
  if isnothing(a_initial)
    eye = _twiss_make_identity(v0_and_coast[1], init, Val{coast}(), Val{spin}())
    if _check_cachable(GTPSA_descriptor)
      tr = _twiss_track(eye, bl, rf_on, batch_index, zero(s[1]), locations, step_save,
                        in_body_coordinates)
      maps = _twiss_maps(eye, tr)
      m_turn = maps[end]
    else
      tr0 = _twiss_track(eye, bl, rf_on, batch_index, zero(s[1]), (similar(s, 0), Int[], String[],
                         String[]), Int[], in_body_coordinates)
      m_turn = _twiss_map(eye, view(tr0.bunch.v, 1, :), _view_q(tr0.bunch.q))
    end
    a, r, tunes = _a_r_tunes(m_turn)
    if isnothing(maps)
      # With phase space variables below the maximum order, the one-turn map around the closed
      # orbit lacks terms (e.g. x^2) that the fixed point's dependence on δ and on parameters
      # turns into terms of lower order in x, so a is only right to first order in them. Tracking
      # the fixed point itself (a0, a translation) loses nothing: the one-turn map around it
      # gives a again, with the error of the fixed point squared each time (a Newton iteration)
      mo = unsafe_load(GTPSA_descriptor.desc).mo
      for _ in 1:ceil(Int, log2(mo))
        a0 = factorise(a).a0
        tr0 = _twiss_track(a0, bl, rf_on, batch_index, zero(s[1]), (similar(s, 0), Int[], String[],
                           String[]), Int[], in_body_coordinates)
        m0 = inv(a0) ∘ _twiss_map(a0, view(tr0.bunch.v, 1, :), _view_q(tr0.bunch.q))
        b, r, tunes = _a_r_tunes(m0)
        a = a0 ∘ b
      end
    end
    a_initial, r_and_tunes = a, (r, tunes)
  end

  if isnothing(damping)
    damping = norm(NNF.checksymp(NNF.jacobian(a_initial))) > symplectic_tol
  end

  # Determine canonization level
  canonise, phase, damp = canonise_phase_damp(GTPSA_descriptor, coast, damping)

  a_initial = factorise(a_initial; canonise=canonise, damping=isnothing(damp) ? false : true).a

  # a at every row: a composed with the maps to the rows, or tracked
  if isnothing(maps)
    tr = _twiss_track(a_initial, bl, rf_on, batch_index, zero(s[1]), locations, step_save,
                      in_body_coordinates)
    fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3 = _twiss_push_a(tr, a_initial, canonise,
                                                                       phase, damp, coast)
  else
    fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3 = _twiss_push_a(maps, a_initial, canonise,
                                                                       phase, damp, coast)
  end
  _twiss_reference!(beta_gamma_ref, t_ref, tr)

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

function co_and_coast(bl, v0, rf_on, coast_tol, batch_index)
  co_sol = find_closed_orbit(bl; v0=copy(v0), batch=Val{false}(), rf_on, batch_start=batch_index, coast_tol) # find_closed_orbit mutates v0
  if co_sol.sol.retcode != RETCODE_SUCCESS
    error("Closed orbit finder did not converge.")
  end
  return (co_sol.v0, co_sol.coasting_beam)
end

_twiss_assemble_locations(bl::Beamline, ::Colon) = _twiss_assemble_locations(bl, [(0., Inf)])

function _twiss_assemble_locations(bl::Beamline, at::AbstractVector)
  at_idxs = filter(x->x isa Integer, at)
  at_eles = filter(x->x isa LatElement, at)
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
  first_found = false # steps of the first element inside an s-range (not only at s = 0)
  for ele in bl.line
    idx = ((ele.BeamlineParams)::BeamlineParams).beamline_index
    up = (ele.UniversalParams)::UniversalParams
    if up.L isa DefExpr # evaluate it (and n_steps from ds_step) with the Beamline's context, as in tracking
      up = Beamlines.deval(up, bl.context)
    end
    name = up.name
    kind = up.kind
    L = up.L
    # Number of steps with a tracking callback: one per integrator step of each slice (tracking
    # errors for elements tracked in fewer steps, e.g. a patch with n_steps > 1)
    n_steps = up.n_slices*up.n_steps
    ds_step = L/n_steps

    # Check which steps are inside any of the ranges
    found = false
    s_ele = scur # element entrance
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
    if idx == 1
      first_found = found
    end

    # If not in an s-range, check if explicitly provided
    # Always include first element
    if !found && (idx == 1 || (any(x -> x == idx, at_idxs) || any(at_eles) do x
          x == ele || (haskey(getfield(ele, :pdict), InheritParams) ? x == (getfield(ele, :pdict)[InheritParams].parent) : false)
        end
        ))
        push!(stmp, s_ele)
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

  include_start = first_found || any(x->x[1]<=0<=x[2], at_ranges) || any(x->x==1, at_idxs) || any(at_eles) do x
    x == bl.line[1] || (haskey(getfield(bl.line[1], :pdict), InheritParams) ? x == (getfield(bl.line[1], :pdict)[InheritParams].parent) : false)
  end
  # The end row is the exit of the last element, so only an s-range containing it includes it 
  # (a selected element gives the row at its entrance)
  include_end = any(x->x[1]<=scur<=x[2], at_ranges)

  return s, names, kinds, idxs, step_save, include_start, include_end
end

function _check_cachable(GTPSA_descriptor)
  # check if we can cache_and_concat: the maps from the start composed with a lose no term if
  # all phase space variables have the maximum order (a has no constant part, so a term of total
  # order n of the composition only needs terms of the maps up to order n). Parameters may have
  # lower orders
  desc = unsafe_load(GTPSA_descriptor.desc)
  mo = desc.mo
  if all(x->x == mo, unsafe_wrap(Vector{UInt8}, desc.no, desc.nv))
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

# The map with the coordinates v (and quaternion q, or nothing) of a map like template, e.g.
# saved by track
function _twiss_map(template, v, q)
  map = zero(template)
  NNF.setray!(map.v, v=v)
  if !isnothing(map.q)
    NNF.setquat!(map.q, q=q)
  end
  return map
end
_view_q(q, j) = isnothing(q) ? nothing : view(q, 1, :, j, 2)
# The maps saved by track at the rows (with m: from the start)
_twiss_maps(m, tr) = [_twiss_map(m, view(tr.v, 1, :, j, 2), _view_q(tr.q, j)) for j in axes(tr.v, 3)]

# The reference at the rows, saved by track
function _twiss_reference!(beta_gamma_ref, t_ref, tr)
  beta_gamma_ref .= view(tr.beta_gamma_ref, :, 2)
  t_ref .= view(tr.t_ref, :, 2)
  return nothing
end

# The functions at every row from a at the start: composed with the maps to the rows, or a itself
# tracked (saved at the rows by track)
function _twiss_push_a(maps_or_tr, a_initial, canonise, phase, damp, coast)
  tracked = maps_or_tr isa TrackingResult
  n = tracked ? size(maps_or_tr.v, 3) : length(maps_or_tr)
  fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3 = _twiss_make_base_columns(n, a_initial,
                                                                               phase, damp)
  for j in 1:n
    a = tracked ? _twiss_map(a_initial, view(maps_or_tr.v, 1, :, j, 2), _view_q(maps_or_tr.q, j)) :
                  maps_or_tr[j] ∘ a_initial
    _store_twiss!(fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3, a, canonise, phase, damp, j,
                  coast)
  end
  return fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3
end
_view_q(q) = isnothing(q) ? nothing : view(q, 1, :)

# Tracks the map m (its coordinates: the orbit plus the map) through bl with track, saving it at the
# rows (locations, at the steps step_save) with the reference there. (Always with the reference,
# so that all of twiss's tracking compiles the same kernels)
function _twiss_track(m, bl, rf_on, batch_index, t_ref0, locations, step_save, in_body_coordinates)
  if NNF.nvars(m) == 5
    v = reshape([(i < 5 ? m.v0[i]+copy(m.v[i]) : copy(m.v[i])) for i in 1:6], 1, 6)
  else
    v = reshape([m.v0[i]+copy(m.v[i]) for i in 1:6], 1, 6)
  end
  q = isnothing(m.q) ? nothing : [copy(m.q[1]) copy(m.q[2]) copy(m.q[3]) copy(m.q[4])]
  # t_ref must hold the type of the lengths
  p_over_q_ref = bl.p_over_q_ref isa TimeDependentParam ? bl.p_over_q_ref(0) : bl.p_over_q_ref
  b0 = Bunch(v=v, q=q, t_ref=t_ref0, species=bl.species_ref, p_over_q_ref=p_over_q_ref)
  config = TrackingConfig(; in_body_coordinates, rf_on, batch_start=batch_index,
                          save_reference=true, use_KA=false)
  n_steps_turn = isempty(step_save) ? 0 : last(step_save)
  return _track(bl, b0, config, nothing, 1, locations, step_save, n_steps_turn, Val(true))
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

# The functions at row j from a, the map from the start: the phases (and damping) are those from
# the start, the oscillation phases (all but the slip of a coasting beam) mod 1, so unwrapped
# against the previous row (whose phase differs by less than half a turn)
function _store_twiss!(fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3, a, canonise, phase, damp, j,
                       coast)
  damping = !isnothing(damp)
  _clear!(phase)
  damping && _clear!(damp)
  facj = factorise(a; canonise=canonise, phase=phase, damp=damp, damping=damping)
  fac[j] = facj
  for (k, phi) in enumerate((phi1, phi2, phi3_or_slip))
    phi[j] = j != 1 ? copy(phase[k]) : zero(phase[k])
    if j != 1 && (k <= 2 || !coast)
      phi[j] += round(TI.scalar(phi[j-1]) - TI.scalar(phi[j]))
    end
  end
  if damping
    damp1[j] = j != 1 ? copy(damp[1]) : zero(damp[1])
    damp2[j] = j != 1 ? copy(damp[2]) : zero(damp[2])
    damp3[j] = j != 1 ? copy(damp[3]) : zero(damp[3])
  end
  return
end

# (in place: the TPS of an MVector can't be replaced)
function _clear!(x)
  for k in eachindex(x)
    if isbits(x[k])
      x[k] = zero(x[k])
    else
      TI.clear!(x[k])
    end
  end
  return x
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
  alphac = -_z_slip(j, twi, cache, Val{true}()) / twi.s[end] +
            _velocity_slip(j, twi, cache, Val{true}())
  if coast && TI.is_tps_type(typeof(etac)) isa TI.IsTPSType
    etac = TI.deriv(etac, 6)
  end
  if coast && TI.is_tps_type(typeof(alphac)) isa TI.IsTPSType
    alphac = TI.deriv(alphac, 6)
  end
  summ[:etac] = oper(etac)
  summ[:alphac] = oper(alphac)

  if !isnothing(twi.r_and_tunes) && length(twi.r_and_tunes[2]) == 4
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
