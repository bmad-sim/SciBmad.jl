#=

The `twiss` function works as follows:

1. The user provides `at`, which is be a `Vector` that can contain:
    a. `Tuple{Float64, Float64}`s as a range of s-positions to save `twiss`
    b. `LineElement`s to save `twiss` at the BEGINNING of the `LineElement`
    c. `Integer`s specifying `beamline_index`es at which to save `twiss` at the BEGINNING

    `at` may alternatively be a `Colon` `(:)`, which means save EVERY step at ALL elements 

  A pass is made over the `Beamline` to:
    a. Resolve the type of the `s` column
    b. Resolve the length and contents of the `s`, `names`, and `beamline_index` columns
    c. Construct an array `step_save` the contains integer values of the TOTAL integration 
        step count to save

** IMPORTANT **
Callbacks are applied AFTER the step. Therefore, 

2. An array `maps` is preallocated to store the map between each saved step

3. A closure is made containing `maps` and `step_save`, as well as over a `Ref` variable that 
    counts the total number of steps up to the given point. This is the callback that will be
    passed to the bunch. If the step is to be saved, then the acculumated map is written to 
    `maps` and reinitialized to the identity.

4. Now we have everything we need to process the data. The one turn map can be obtained by 
    concatenating all the maps in `maps` (or if `Descriptor(6, 1)`, in theory could get from 
    closed orbit finder Jacobian). Now just concatenate, canonize, and continue.
=#

# ========== STEP 1 ==================================
# Resolve number of steps, step indicies, type of s, base columns, etc.

function _twiss_1(bl::Beamline, at::Vector)
  at_idxs = filter(x->x isa Integer, at)
  at_eles = filter(x->x isa LineElement, at)
  at_ranges = filter(x->x isa Tuple, at)
  
  stmp = Vector{Any}(undef, 0)
  names = Vector{String}(undef, 0)
  idxs = Vector{Int}(undef, 0)
  step_save = Vector{Int}(undef, 0)

  # As a guess assume length equal to number of beamline elements + 1
  # This makes the typical Twiss case hopefully faster
  n_ele = length(bl.line)
  sizehint!(stmp, n_ele+1)
  sizehint!(names, n_ele+1)
  sizehint!(idxs, n_ele+1)
  sizehint!(step_save, n_ele)

  scur = 0f0
  step_cur = 0
  for ele in bl.line
    idx = ((ele.BeamlineParams)::BeamlineParams).beamline_index
    up = (ele.UniversalParams)::UniversalParams
    name = up.name
    tm = up.tracking_method
    L = up.L
    n_steps, ds_step = find_steps(tm, L)

    # Check which steps are inside any of the ranges
    found = false
    for _ in 1:n_steps
      if any(x -> x[1] <= scur < x[2], at_ranges)
        push!(stmp, scur)
        push!(names, name)
        push!(idxs, idx)
        push!(step_save, step_cur)
        found = true
      end
      step_cur += 1
      scur += ds_step
    end
    
    # If not in an s-range, check if explicitly provided (BUT ONLY AT BEGINNING!)
    # therefore need to be done at the PREVIOUS element LAST step!
    # First element must be handled specially.
    if !found && ((any(x -> x == idx, at_idxs) || any(at_eles) do x
          x == ele || (haskey(getfield(ele, :pdict), InheritParams) ? x == (getfield(ele, :pdict)[InheritParams].parent) : false)
        end
        ))
        push!(stmp, scur - ds_step*n_steps)
        push!(names, name)
        push!(idxs, idx)
        push!(step_save, step_cur-n_steps)
        #step_cur += 1
    end
  end

  # Now check if any went beyond the length of the line, in which 
  # case also save at the end of the last element.
  if any(x -> x[1] <= scur < x[2], at_ranges)
    push!(stmp, scur)
    push!(names, "END OF BEAMLINE")
    push!(idxs, -1)
    push!(step_save, step_cur)
  end

  # Now resolve type of s:
  s = typeof(scur).(stmp)

  return s, names, idxs, step_save
end

# Colon means save everywhere:
_twiss_1(bl::Beamline, ::Colon) = _twiss_1(bl, [(0., Inf)])

# ========== UTILITIES: STEP 1 =======================
function find_steps(tm::BeamTracking.AbstractYoshida, L) 
  if L == 0
    return (1, L)
  end
  ds_step = tm.ds_step
  n_steps = tm.n_steps
  if ds_step < 0
    ds_step = L / n_steps
    return (n_steps, ds_step)
  else
    return (ceil(Int, L / ds_step), ds_step)
  end
end
find_steps(::SciBmadStandard, L) = (1, L)
find_steps(::Any, L) = (1, L)

# ========== STEP 2 ==================================
# Preallocate maps array and resolve types of everything

function _twiss_2(step_save, v0_and_coast, GTPSA_descriptor, ::Val{spin}, ::Val{RDTs}) where {spin, RDTs}
  v0 = v0_and_coast[1]
  coasting_beam = v0_and_coast[2]

  if isnothing(GTPSA_descriptor)
    storedesc = GTPSA.desc_current
    GTPSA_descriptor = Descriptor(6,1)
    GTPSA.desc_current = storedesc # Don't reset the global
  end

  nn = GTPSA.numnn(GTPSA_descriptor)
  if nn < 6
    error("GTPSA Descriptor must have at least 6 variables for the 6D phase space coordinates")
  end

  numtype = eltype(v0)
  init = TI.InitGTPSA{GTPSA.Dynamic,Descriptor}(; dynamic_descriptor=GTPSA_descriptor)
  mo = TI.maxord(init)
  nn = TI.ndiffs(init)
  nv = 6
  np = nn-nv
  if coasting_beam
    nv -= 1
    np += 1
  end

  # Type of the LATTICE FUNCTIONS
  if mo > 1
    zero_LF = TI.init_tps(numtype, init)
  else
    zero_LF = zero(numtype)
  end

  # Type of the PHASES
  if mo > 1 || coasting_beam
    zero_phase = TI.init_tps(numtype, init)
  else
    zero_phase = zero(numtype)
  end
  
  # Type of the ORBIT
  if coasting_beam || nn > 6
    zero_orbit = TI.init_tps(numtype, init)
  else
    zero_orbit = zero(numtype)
  end

  # Value type of the RDT dict
  if RDTs
    if mo == 1
      error("
        RDTs cannot be computed using a GTPSA_descriptor with max order 1.
        Please specify a higher order GTPSA_descriptor.
      ")
    end
    if np > 0
      zero_h = TI.init_tps(numtype, init)
    else
      zero_h = zero(numtype)
    end
  else
    zero_h = nothing # Don't compute it
  end

  eye = DAMap(init=init, nv=nv, np=np, v0=view(v0, :, 1:nv), v_matrix=I, q=(spin ? I : nothing))
  maps = @noinline _twiss_2_preallocate(step_save, eye)
  return eye, maps, zero_LF, zero_phase, zero_orbit, zero_h
end

function _twiss_2_preallocate(step_save, map::T) where {T<:DAMap}
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

# ========== STEP 3 ==================================
# Construct the closure

function _twiss_3(_step_save, _maps)
  # Note: need to handle the first element differently
  if length(_step_save) > 0 && first(_step_save) == 0
    _cur_step_save_idx = 2
  else
    _cur_step_save_idx = 1
  end
  let step_save=_step_save, maps=_maps, curstep=Ref{Int}(0), cur_step_save_idx=Ref{Int}(_cur_step_save_idx)
    return (coords, ds_step, g) -> begin
      curstep[] += 1
      if cur_step_save_idx[] <= length(step_save) && curstep[] == step_save[cur_step_save_idx[]] # Store the current map
        map = maps[cur_step_save_idx[]]
        _twiss_setmap!(map, coords)
        cur_step_save_idx[] += 1
      end
    end
  end
end

# ========== UTILITIES: STEP 3 =======================

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

# ========== STEP 4 ==================================
# Track a bunch and fill the `maps` array

function _twiss_4(eye, cb, bl)
  if NNF.nvars(eye) == 5
    v = reshape([(i < 5 ? eye.v0[i]+copy(eye.v[i]) : copy(eye.v[i])) for i in 1:6], 1, 6)
  else
    v = reshape([eye.v0[i]+copy(eye.v[i]) for i in 1:6], 1, 6)
  end
  q = isnothing(eye.q) ? nothing : [copy(eye.q[1]) copy(eye.q[2]) copy(eye.q[3]) copy(eye.q[4])]
  b0 = Bunch(v=v, q=q, callbacks=(cb,))
  BTBL.check_bl_bunch!(b0, bl, false) # Do not notify
  track!(b0, bl)
  return b0
end

# ========== STEP 5 ==================================

function _twiss_5!(eye, b0, maps)
  # Now we just concatenate the maps
  m_turn = eye
  for map in maps
    m_turn = map ∘ m_turn
  end
  # Have to do one more now
  _twiss_setmap!(eye, b0.coords)
  if length(maps) > 0
    m_turn = eye ∘ m_turn
  end
  return m_turn
end

# ========== STEP 6 ==================================
# Tunes and a

function _twiss_6(m::DAMap)
  mo = NNF.maxord(m)
  a = normal(m)
  c = c_map(m) # Transform to phasor basis
  r = inv(c) ∘ inv(a) ∘ m ∘ a ∘ c
  # Need to cut highest order
  Q_x = cutord(real(-log(SciBmad.NNF.factor_out(r.v[1], 1))/(2*pi*im)), mo)
  Q_y = cutord(real(-log(SciBmad.NNF.factor_out(r.v[3], 3))/(2*pi*im)), mo)
  if NNF.nvars(m) == 5
    Q_s = real(r.v[5])
    TI.seti!(Q_s, 0, 5) # subtract time identity
  else
    Q_s = cutord(real(-log(SciBmad.NNF.factor_out(r.v[5], 5))/(2*pi*im)), mo)
  end
  if isnothing(m.q)
    return SA[Q_x, Q_y, Q_s], a
  else
    Q_spin = -atan(real(r.q.q2), real(r.q.q0))/pi # not two pi bc quaternion
    return SA[Q_x, Q_y, Q_s, Q_spin], a
  end
end

# ========== STEP 6 ==================================
# Go around the ring

function _twiss_7(
    a::DAMap{<:Any,<:Any,Q}, 
    m,
    maps, 
    s, 
    names, 
    idxs, 
    symplectic_tol, 
    zero_LF::T, 
    zero_phase::V, 
    zero_orbit::U, 
    zero_h::H, 
    ::Val{de_moivre},
    ::Val{normalizing_map},
  ) where {Q, T, V, U, H, de_moivre, normalizing_map}
  # These checks should all be static ================================
  # if compiler isn't a dumb dumb, which it definitely can be
  COMPUTE_TWISS = de_moivre ? compute_de_moivre : compute_sagan_rubin
  LF = !de_moivre ? twiss_tuple : de_moivre_tuple 
  LF_TABLE = !de_moivre ? twiss_table : de_moivre_table
  SCALAR_LF = TI.is_tps_type(T) isa TI.IsTPSType ? Val{false}() : Val{true}()
  SCALAR_PHASE = TI.is_tps_type(V) isa TI.IsTPSType ? Val{false}() : Val{true}()
  SCALAR_ORBIT = TI.is_tps_type(U) isa TI.IsTPSType ? Val{false}() : Val{true}()
  INCLUDE_A = normalizing_map ? at -> at : at -> nothing
  BENGTSSON = H == Nothing ? (a0, a1, m)->nothing : compute_bengtsson

  if Q == Nothing
    PROCESS_SPIN = at -> nothing
  else
    i2 = zero(at)
    NNF.setray!(i2.v; v_matrix=I)
    TI.seti!(i2.q.q2, 1, 0)
    let i2=i2
      PROCESS_SPIN = at -> begin
        n = at ∘ i2 ∘ inv(at)
        SA[n.q.q1, n.q.q2, n.q.q3]
      end
    end
  end
  # Note:
  # Descriptor(6,1) with coasting beam gives SCALAR_LF = true 
  # but SCALAR_ORBIT = false
  # In general we will canonize using SCALAR_ORBIT, and compute 
  # lattice functions using SCALAR_LF. 
  # Finally we have the phases. The phases are done during 
  # canonization, and so should have the same type as the orbit.
  if SCALAR_ORBIT isa Val{false}
    PROCESS_ORBIT = v -> begin
      StaticArrays.sacollect(SVector{6,U}, begin 
      vi = zero(v[i])
      TI.copy_tps!(vi, v[i])
      if i < 6
        TI.seti!(vi, 0, i)
      end
      vi
      end for i in 1:6)
    end
  else
    PROCESS_ORBIT = v -> StaticArrays.sacollect(SVector{6,U}, TI.scalar(v[i]) for i in 1:6)
  end
  # =================================================================
  damping = norm(NNF.checksymp(NNF.jacobian(m))) > symplectic_tol
  a = maps[1] ∘ a
  r = canonize(a, SCALAR_PHASE; damping=damping)
  a = a ∘ r
  fc = factorize(a)
  NNF_tuple = COMPUTE_TWISS(fc.a1, SCALAR_LF)
  m = H == Nothing ? m : (maps[1] ∘ m)
  lf1 = LF(
    s[1],
    idxs[1], 
    names[1], 
    SA[
      zero(zero_phase),
      zero(zero_phase),
      zero(zero_phase)
    ], 
    NNF_tuple, 
    PROCESS_ORBIT(fc.a0.v), 
    PROCESS_SPIN(a), 
    INCLUDE_A(a),
    BENGTSSON(fc.a0, fc.a1, m)
  )
  lf_table = LF_TABLE(lf1, length(maps))
  lf_table[1] = lf1
  phase = MVector{3}(zero(zero_phase),zero(zero_phase),zero(zero_phase))
  len = length(maps)
  for i in 2:len
    m = H == Nothing ? m : (maps[i] ∘ m ∘ inv(maps[i]))
    a = maps[i] ∘ a
    r = canonize(a, SCALAR_PHASE; phase=phase, damping=damping)
    a = a ∘ r
    fc = factorize(a)
    lfi = LF(
      s[i],
      idxs[i], 
      names[i], 
      SA[
        copy(phase[1]),
        copy(phase[2]),
        copy(phase[3])
      ], 
      COMPUTE_TWISS(fc.a1, SCALAR_LF), 
      PROCESS_ORBIT(fc.a0.v), 
      PROCESS_SPIN(a), 
      INCLUDE_A(a),
      BENGTSSON(fc.a0, fc.a1, m)
    )
    lf_table[i] = lfi
  end
  return lf_table
end

function _twiss_type_stable(
  bl,
  eye, 
  maps, 
  s, 
  names, 
  idxs, 
  step_save, 
  symplectic_tol,
  zero_LF, 
  zero_phase, 
  zero_orbit, 
  zero_h, 
  ::Val{de_moivre},
  ::Val{normalizing_map},
  ::Val{table}
  ) where{de_moivre, normalizing_map,table}
  cb = _twiss_3(step_save, maps)
  b0 = _twiss_4(eye, cb, bl)
  m = _twiss_5!(eye, b0, maps)
  tunes, a = _twiss_6(m)
  if table
    lf_table = _twiss_7(a, m, maps, s, names, idxs, symplectic_tol,  zero_LF, zero_phase, zero_orbit, zero_h, Val{de_moivre}(),Val{normalizing_map}())
    return Twiss(NNF.nvars(m) == 5, tunes, lf_table)
  else
    return Twiss(NNF.nvars(m) == 5, tunes, nothing)
  end
end

function twiss(
  bl::Beamline; 

  # High level customizer kwargs
  GTPSA_descriptor::Union{Descriptor,Nothing} = nothing, 
  spin::Bool                                  = false,
  de_moivre::Bool                             = false,
  normalizing_map::Bool                       = false,
  RDTs::Bool                                  = false,
  at::Union{Colon, Vector}                    = :,

  # Initial input:
  v0::Matrix                        = zeros(1,6),
  v0_and_coast::Tuple{Matrix, Bool} = co_and_coast(bl, v0),
  #a_initial::Union{Nothing,DAMap}   = nothing, # TODO

  symplectic_tol=1e-8, # Tolerance below which to include damping
  )
  
  # Type unstable steps:
  s, names, idxs, step_save = _twiss_1(bl, at)
  eye, maps, zero_LF, zero_phase, zero_orbit, zero_h = _twiss_2(step_save, v0_and_coast, GTPSA_descriptor, Val{spin}(), Val{RDTs}())
  table = length(step_save) == 0 ? false : true
  # Type stable steps:
  return _twiss_type_stable(bl, eye, maps, s, names, idxs, step_save, symplectic_tol, zero_LF, zero_phase, zero_orbit, zero_h, Val{de_moivre}(), Val{normalizing_map}(), Val{table}())
end

struct Twiss{S,T}
  coasting_beam::Bool
  tunes::S
  table::T
end

function Base.show(io::IO, tw::Twiss)
  println(io, "Twiss:")
  width = length(" coasting_beam")
  println(io, rpad(" coasting_beam", width), " = ", tw.coasting_beam)
  spin = length(tw.tunes) == 4

  print(io, rpad(" tunes[1:$(length(tw.tunes))]", width), " = [Qx, Qy")
  if tw.coasting_beam
    print(io, ", slip")
  else
    print(io, ", Qz")
  end
  if spin
    print(io, ", Qspin")
  end

  print(io, "]\n")

  if !isnothing(tw.table)
    print(io, rpad(" table", width), " has columns: ") 
    cols = keys(getfield(tw.table, :data))
    for col in cols
      print(io, String(col))
      if col != last(cols)
        print(io, ", ")
      end
    end
  end
  return
end

function co_and_coast(bl, v0)
  co_sol = find_closed_orbit(bl; v0=v0, batch=Val{false}())
  if co_sol.sol.retcode != RETCODE_SUCCESS
    error("Closed orbit finder did not converge.")
  end
  return (co_sol.v0, co_sol.coasting_beam)
end

function twiss_tuple(s, beamline_index, name, phi, NNF_tuple::TT, orbit, n, a, h) where {TT}
  outt = (;
    s = s,
    beamline_index = beamline_index,
    name = name,
    phi_1 = phi[1],
    beta_1 = NNF_tuple.beta[1],
    alpha_1 = NNF_tuple.alpha[1],
    phi_2 = phi[2],
    beta_2 = NNF_tuple.beta[2],
    alpha_2 = NNF_tuple.alpha[2],
    phi_3 = phi[3],
    gamma_c = NNF_tuple.gamma_c,
    c11 = NNF_tuple.C[1,1],
    c12 = NNF_tuple.C[1,2],
    c21 = NNF_tuple.C[2,1],
    c22 = NNF_tuple.C[2,2],
    orbit_x  = orbit[1],
    orbit_px = orbit[2],
    orbit_y  = orbit[3],
    orbit_py = orbit[4],
    orbit_z  = orbit[5],
    orbit_pz = orbit[6],
  )

  if !isnothing(h)
    outt = merge(outt, (; h = h)) # Bengtsson polynomial dict
  end

  # static check
  if hasfield(TT, :eta) # NOT coasting
    # eta, zeta, and slip are APPROXIMATIONS
    # In coasting case all quantities are exact and in a0
    outt = merge(outt, 
      (; 
        eta_1 = NNF_tuple.eta[1],
        etap_1 = NNF_tuple.eta[2],
        eta_2 = NNF_tuple.eta[3],
        etap_2 = NNF_tuple.eta[4],
        zeta_1 = NNF_tuple.zeta[1],
        zetap_1 = NNF_tuple.zeta[2],
        zeta_2 = NNF_tuple.zeta[3],
        zetap_2 = NNF_tuple.zeta[4],
        slip = NNF_tuple.approx_slip*sin(phi[3]*2*pi), # Approximation from EBB)
      )
    )
  end

  # static check
  if !isnothing(n)
    outt = merge(outt, (; n_x = n[1], n_y = n[2], n_z = n[3],))
  end
  
  # static check
  if !isnothing(a)
    outt = merge(outt, (; a=a))
  end

  return outt
end

function twiss_table(tt::TT, N_ele) where {TT}
  S = typeof(tt.s)
  V = typeof(tt.phi_1)
  T = typeof(tt.beta_1)
  U = typeof(tt.orbit_x)

  cols = (;
    s = Vector{S}(undef, N_ele),
    beamline_index = Vector{Int}(undef, N_ele),
    name = Vector{String}(undef, N_ele),
    phi_1 = Vector{V}(undef, N_ele),
    beta_1 = Vector{T}(undef, N_ele),
    alpha_1 = Vector{T}(undef, N_ele),
    phi_2 = Vector{V}(undef, N_ele),
    beta_2 = Vector{T}(undef, N_ele),
    alpha_2 = Vector{T}(undef, N_ele),
    phi_3 = Vector{V}(undef, N_ele),
    gamma_c = Vector{T}(undef, N_ele),
    c11 = Vector{T}(undef, N_ele),
    c12 = Vector{T}(undef, N_ele),
    c21 = Vector{T}(undef, N_ele),
    c22 = Vector{T}(undef, N_ele),
    orbit_x = Vector{U}(undef, N_ele),
    orbit_px = Vector{U}(undef, N_ele),
    orbit_y = Vector{U}(undef, N_ele),
    orbit_py = Vector{U}(undef, N_ele),
    orbit_z = Vector{U}(undef, N_ele),
    orbit_pz = Vector{U}(undef, N_ele),
  )

  if hasfield(TT, :h)
    H = typeof(tt.h)
    cols = merge(cols, (; h = Vector{H}(undef, N_ele)))
  end

  # static check
  if hasfield(TT, :eta_1)
    cols = merge(cols, 
      (;
        eta_1   = Vector{T}(undef, N_ele),
        etap_1  = Vector{T}(undef, N_ele),
        eta_2   = Vector{T}(undef, N_ele),
        etap_2  = Vector{T}(undef, N_ele),
        zeta_1  = Vector{T}(undef, N_ele),
        zetap_1 = Vector{T}(undef, N_ele),
        zeta_2  = Vector{T}(undef, N_ele),
        zetap_2 = Vector{T}(undef, N_ele),
        slip    = Vector{T}(undef, N_ele),
      )
    )
  end

  # static check
  if hasfield(TT, :n_x)
    W = typeof(tt.n_x)
    cols = merge(cols, 
      (;
        n_x = Vector{W}(undef, N_ele),
        n_y = Vector{W}(undef, N_ele),
        n_z = Vector{W}(undef, N_ele),
      )
    )
  end

  if hasfield(TT, :a)
    A = typeof(tt.a)
    cols = merge(cols, (; a = Vector{A}(undef, N_ele)))
  end

  return Table(cols)
end

function de_moivre_tuple(s, beamline_index, name, phi, NNF_tuple, orbit, n, a, h)
  outt = (;
    s = s,
    beamline_index = beamline_index,
    name = name,
    phi_1 = phi[1],
    phi_2 = phi[2],
    phi_3 = phi[3],
    H = NNF_tuple.H,
    B = NNF_tuple.B,
    E = NNF_tuple.E,
    K = NNF_tuple.K,
    orbit_x  = orbit[1],
    orbit_px = orbit[2],
    orbit_y  = orbit[3],
    orbit_py = orbit[4],
    orbit_z  = orbit[5],
    orbit_pz = orbit[6],
  )

  if !isnothing(h)
    outt = merge(outt, (; h = h)) # Bengtsson polynomial dict
  end

  if !isnothing(n)
    outt = merge(outt, (; n_x = n[1], n_y = n[2], n_z = n[3],))
  end

  if !isnothing(a)
    outt = merge(outt, (; a=a))
  end

  return outt
end

function de_moivre_table(dt::DT, N_ele) where {DT}
  S = typeof(dt.s)
  V = typeof(dt.phi_1)
  T = typeof(dt.H)
  U = typeof(dt.orbit_x)

  cols = (;
    s = Vector{S}(undef, N_ele),
    beamline_index = Vector{Int}(undef, N_ele),
    name = Vector{String}(undef, N_ele),
    phi_1 = Vector{V}(undef, N_ele),
    phi_2 = Vector{V}(undef, N_ele),
    phi_3 = Vector{V}(undef, N_ele),
    H = Vector{T}(undef, N_ele),
    B = Vector{T}(undef, N_ele),
    E = Vector{T}(undef, N_ele),
    K = Vector{T}(undef, N_ele),
    orbit_x = Vector{U}(undef, N_ele),
    orbit_px = Vector{U}(undef, N_ele),
    orbit_y = Vector{U}(undef, N_ele),
    orbit_py = Vector{U}(undef, N_ele),
    orbit_z = Vector{U}(undef, N_ele),
    orbit_pz = Vector{U}(undef, N_ele),
  )

  if hasfield(DT, :h)
    H = typeof(dt.h)
    cols = merge(cols, (; h = Vector{H}(undef, N_ele)))
  end

  if hasfield(DT, :n_x)
    W = typeof(dt.n_x)
    cols = merge(cols, 
      (;
        n_x = Vector{W}(undef, N_ele),
        n_y = Vector{W}(undef, N_ele),
        n_z = Vector{W}(undef, N_ele),
      )
    )
  end

  if hasfield(DT, :a)
    A = typeof(dt.a)
    cols = merge(cols, (; a = Vector{A}(undef, N_ele)))
  end

  return Table(cols)
end