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

function twiss(
  bl::Beamline; 

  at::Union{Colon, AbstractVector}  = :,
  in_body_coordinates::Bool = false, 

  # GTPSA truncation order sets:
  chrom::Integer = 0,
  order::Integer = 1,
  GTPSA_descriptor::Union{Descriptor,Nothing} = nothing,

  # Initial input, CO guess if periodic, initial orbit if not periodic
  delta0::Number = 0.,
  v0::Matrix     = [0. 0. 0. 0. 0. delta0], 

  start::Union{Integer,LineElement,Nothing} = nothing, # TODO: Nothing means compute periodic  
  a_initial::Union{Nothing,DAMap}   = nothing, # TODO
  damping::Union{Nothing,Bool} = nothing, # nothing = auto-detect from a_initial

  # The lattice functions to compute
  spin::Bool      = isnothing(a_initial) ? false : !isnothing(a_initial.q),
  base_cols       = SciBmad.base_cols,
  cols            = nothing, # (de_moivre ? DE_MOIVRE : TENG_EDWARDS)..., (spin ? SPIN : Function[])...],

  symplectic_tol = 1e-8, # Tolerance below which to include damping
  )
  if isnothing(start)
    v0_and_coast = co_and_coast(bl, v0)
  else
    v0_and_coast = (v0, false) # Always do 6D if open
  end

  if isnothing(cols)
    if v0_and_coast[2]
      cols = TENG_EDWARDS_COAST
    else
      cols = TENG_EDWARDS
    end
  end

  if spin
    spincols = ["n0x", "n0y", "n0z", "dnx_1", "dny_1", "dnz_1"]
    for spincol in spincols
      spincol in cols || push!(cols, spincol)
    end
  end

  if isnothing(GTPSA_descriptor)
    storedesc = GTPSA.desc_current
    GTPSA_descriptor = Descriptor([order, order, order, order, order, order+chrom], order+chrom)
    GTPSA.desc_current = storedesc # Don't reset the global
  elseif chrom != 0 || order != 1
    @info "`GTPSA_descriptor` has been explicitly provided: ignoring `order`/`chrom` inputs"
  end

  if !v0_and_coast[2] && chrom != 0
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
  if TI.ndiffs(init) > 6
    parametric = Val{true}()
  else
    parametric = Val{false}()
  end

  # Assemble locations. Note that start and end of the Beamline are ALWAYS included
  s, names, kinds, idxs, step_save = _twiss_assemble_locations(bl, at)
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
      a_initial, r_and_tunes, maps = _compute_periodic_a_and_cache!(bl, v0, init, Val{v0_and_coast[2]}(), Val{spin}(), step_save, beta_gamma_ref, t_ref, in_body_coordinates)
    else
      a_initial, r_and_tunes = _compute_periodic_a(bl, v0, init, Val{v0_and_coast[2]}(), Val{spin}())
    end
  end

  if isnothing(damping)
    damping = norm(NNF.checksymp(NNF.jacobian(a_initial))) > symplectic_tol
  end

  # Determine canonization level
  canonise, phase, damp = canonise_phase_damp(GTPSA_descriptor, v0_and_coast[2], damping)

  a_initial = factorise(a_initial; canonise=canonise, damping=isnothing(damp) ? false : true).a

  # Now we push 
  if isnothing(maps)
    fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3 = _twiss_push_a!(bl, step_save, a_initial, canonise, phase, damp, beta_gamma_ref, t_ref, in_body_coordinates)
  else
    fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3 = _twiss_push_a_with_cache(maps, step_save, a_initial, canonise, phase, damp)
  end

  twi = TwissInternal(s, names, kinds, idxs, beta_gamma_ref, t_ref, fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3, r_and_tunes)

  # Finally, construct the summary and the dataframe (with provided columns)
  # And post-process with the provided columns
  # Need to do one row first then can construct the DataFrame
  df = _twiss_df(vcat(base_cols, cols), twi)

  return Twiss(Dict{Symbol,Nothing}(), df)#, r_and_tunes[2]
end

function co_and_coast(bl, v0)
  co_sol = find_closed_orbit(bl; v0=v0, batch=Val{false}())
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
        #step_cur += 1
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

  return s, names, kinds, idxs, step_save
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

function _twiss_track!(eye, cbs, bl)
  if NNF.nvars(eye) == 5
    v = reshape([(i < 5 ? eye.v0[i]+copy(eye.v[i]) : copy(eye.v[i])) for i in 1:6], 1, 6)
  else
    v = reshape([eye.v0[i]+copy(eye.v[i]) for i in 1:6], 1, 6)
  end
  q = isnothing(eye.q) ? nothing : [copy(eye.q[1]) copy(eye.q[2]) copy(eye.q[3]) copy(eye.q[4])]
  b0 = Bunch(v=v, q=q, callbacks=cbs)
  BTBL.check_bl_bunch!(b0, bl, false) # Do not notify
  track!(b0, bl)
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

function _compute_periodic_a(bl::Beamline, v0, init, ::Val{coast}, ::Val{spin}, cbs=()) where {coast, spin}
  eye = _twiss_make_identity(v0, init, Val{coast}(), Val{spin}())
  b0 = _twiss_track!(eye, (), bl)
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

function _compute_periodic_a_and_cache!(bl::Beamline, v0, init, ::Val{coast}, ::Val{spin}, step_save, beta_gamma_ref, t_ref, in_body_coordinates) where {coast, spin}
  eye = _twiss_make_identity(v0, init, Val{coast}(), Val{spin}())
  maps = _twiss_cache_preallocate(step_save, eye)
  cb = _twiss_cache_make_callback(step_save, beta_gamma_ref, t_ref, in_body_coordinates, maps)
  _twiss_track!(eye, (cb,), bl)
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
  phi1[j] = copy(phase[1])
  phi2[j] = copy(phase[2])
  phi3_or_slip[j] = copy(phase[3])
  if damping
    damp1[j] = copy(damp[1])
    damp2[j] = copy(damp[2])
    damp3[j] = copy(damp[3])
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

function _twiss_make_base_columns(n, ::T, phase, damp) where {T}
  fac = Vector{@NamedTuple{a0::T, a1::T, a2::T, a::T, r::T}}(undef, n)
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

function _twiss_push_a!(bl, step_save, a_initial, canonise, phase, damp, beta_gamma_ref, t_ref, in_body_coordinates)
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
  _twiss_track!(a_initial, (cb,), bl)
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
