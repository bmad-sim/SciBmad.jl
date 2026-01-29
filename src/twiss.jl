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

# Returns a Table of the Twiss parameters
# See Eq. 4.28 in EBB
function twiss(
  bl::Beamline; 
  GTPSA_descriptor=nothing, #Descriptor(6, 1),
  spin=false,
  de_moivre=false,
  co_info=find_closed_orbit(bl),
  symplectic_tol=1e-8, # Tolerance below which to include damping
  at::Union{Colon,AbstractArray}=:, # Colon means all elements, nothing means no elements
)

  if isnothing(GTPSA_descriptor)
    storedesc = GTPSA.desc_current
    GTPSA_descriptor = Descriptor(6,1)
    GTPSA.desc_current = storedesc # Don't reset the global
  end

  # First get closed orbit:
  # This will get the map and tell us if coasting, etc etc
  nn = GTPSA.numnn(GTPSA_descriptor)
  if nn < 6
    error("GTPSA Descriptor must have at least 6 variables for the 6D phase space coordinates")
  end

  v0 = co_info[1]
  coast = co_info[2]

  # Track once through and construct a DAMap
  Δv = vars(GTPSA_descriptor)[1:6]
  for (Δvi, v0i) in zip(Δv, v0)
    Δvi[0] = v0i # expand around closed orbit
  end
  if spin
    q = [one(first(Δv)) zero(first(Δv)) zero(first(Δv)) zero(first(Δv))]
    b0 = Bunch(reshape(Δv, (1,6)), q)
  else
    b0 = Bunch(reshape(Δv, (1,6)))
  end
  BTBL.check_bl_bunch!(bl, b0, false) # Do not notify

  # type of the LATTICE FUNCTIONS
  # linear_LF = true -> floats, linear_LF = false -> TPSs
  numtype = TI.numtype(b0.coords.v[1])
  init = TI.getinit(b0.coords.v[1])
  mo = TI.maxord(init)
  nn = TI.ndiffs(init)
  nv = 6
  np = nn-nv
  if coast
    nv -= 1
    np += 1
  end
  track!(b0, bl)

  m = DAMap(nv=nv, np=np, v=view(dropdims(b0.coords.v; dims=1), 1:nv), q=b0.coords.q)

  # twiss will ALWAYS compute the (amplitude-dependent) tunes, even when `at` is empty
  # function barrier:
  tunes, a = _tunes_and_a(m, mo, coast)

  if !coast && mo == 1
    tunes = TI.scalar.(tunes)
  end

  if at isa Colon || (at isa AbstractArray && length(at) > 0)
    if !(at isa Colon)
      if eltype(at) != LineElement
        error("Keyword argument `at` must be either a Colon (:) to specify all elements, 
                an empty array (which can have Any type) to specify no elements, or an 
                array with element type LineElement.")
      else
        at = sort(at, by=x->x.beamline_index)
      end
    end

    # Check if symplectic or not
    damping = norm(NNF.checksymp(NNF.jacobian(m))) > symplectic_tol

    # Type of the LATTICE FUNCTIONS
    if mo > 1 && (coast || nn > 6)
      zero_LF = TI.init_tps(numtype, init)
    else
      zero_LF = zero(numtype)
    end

    # Type of the PHASES
    if coast || mo > 1 && nn > 6
      zero_phase = TI.init_tps(numtype, init)
    else
      tunes = TI.scalar.(tunes)
      zero_phase = zero(numtype)
    end
    
    # Type of the ORBIT
    if coast || nn > 6
      zero_orbit = TI.init_tps(numtype, init)
    else
      zero_orbit = zero(numtype)
    end

    N_ele = at isa Colon ? length(bl.line)+1 : length(at)
    # Fill the s array now for each at
    # as well as names and beamline_idxs
    stmp = Vector{Any}(undef, N_ele)
    names = Vector{String}(undef, N_ele)
    idxs = Vector{Int}(undef, N_ele)
    scur = 0f0
    idx = 1
    for ele in bl.line
      if at isa Colon || ele in at
        stmp[idx] = scur
        idxs[idx] = ((ele.BeamlineParams)::BeamlineParams).beamline_index
        names[idx] = ((ele.UniversalParams)::UniversalParams).name
        idx += 1
      end
      scur += ele.L
    end
    if at isa Colon
      stmp[end] = scur
      idxs[end] = -1
      names[end] = "END OF BEAMLINE"
    end
    s = typeof(scur).(stmp)
    lf_table = _twiss(a, b0, bl, idxs, names, s, Val{de_moivre}(), damping, zero_LF, zero_phase, zero_orbit, at)
  else
    lf_table = nothing
  end

  return Twiss(coast, tunes, lf_table)
end

function _tunes_and_a(m::DAMap, mo, coast)
  a = normal(m)
  c = c_map(m) # Transform to phasor basis
  r = inv(c) ∘ inv(a) ∘ m ∘ a ∘ c
  # Need to cut highest order
  Q_x = cutord(real(-log(SciBmad.NNF.factor_out(r.v[1], 1))/(2*pi*im)), mo)
  Q_y = cutord(real(-log(SciBmad.NNF.factor_out(r.v[3], 3))/(2*pi*im)), mo)
  if coast
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
  
function _twiss(
  a::DAMap{<:Any,<:Any,Q}, 
  b0::Bunch, 
  bl::Beamline, 
  idxs, 
  names,
  s,
  ::Val{de_moivre}, 
  damping,
  zero_LF::T, 
  zero_phase::V,
  zero_orbit::U, 
  at::C,
) where {de_moivre, Q, T, V, U, C}
  # Ripken-Wolski-Forest de Moivre coupling formalism
  
  # These checks should all be static ================================
  # if compiler isn't a dumb dumb, which it definitely can be
  COMPUTE_TWISS = de_moivre ? compute_de_moivre : compute_sagan_rubin
  LF = !de_moivre ? twiss_tuple : de_moivre_tuple 
  LF_TABLE = !de_moivre ? twiss_table : de_moivre_table
  SCALAR_LF = TI.is_tps_type(T) isa TI.IsTPSType ? Val{false}() : Val{true}()
  SCALAR_PHASE = TI.is_tps_type(V) isa TI.IsTPSType ? Val{false}() : Val{true}()
  SCALAR_ORBIT = TI.is_tps_type(U) isa TI.IsTPSType ? Val{false}() : Val{true}()
  if Q == Nothing
    PROCESS_SPIN = a -> nothing
  else
    PROCESS_SPIN = at -> begin
      i2 = zero(at)
      NNF.setray!(i2.v; v_matrix=I)
      TI.seti!(i2.q.q2, 1, 0)
      n = at ∘ i2 ∘ inv(at)
      SA[n.q.q1, n.q.q2, n.q.q3]
    end
  end
  # Note:
  # Descriptor(6,1) with coasting beam gives SCALAR_LF = true 
  # but SCALAR_ORBIT = false
  # In general we will canonize using SCALAR_ORBIT, and compute 
  # lattice functions using SCALAR_LF. 
  # Finally we have the phases. The phases are done during 
  # canonization, and so should have the same type as the orbit.

  # For some horrendous reason, processing the orbit is impacting the future lattice
  # function calc
  # no clue why
  # need to investigate futher
  if SCALAR_ORBIT isa Val{false}
    PROCESS_ORBIT = v -> begin
      StaticArrays.sacollect(SVector{6,U}, begin 
      if i < 6
        TI.seti!(v[i], 0, i)
      end
      v[i]
      end for i in 1:6)
    end
  else
    PROCESS_ORBIT = v -> StaticArrays.sacollect(SVector{6,U}, TI.scalar(v[i]) for i in 1:6)
  end
  # =================================================================
  if C == Colon
    N_ele = length(bl.line)+1
  else
    N_ele = length(at)
  end

  # a = normal(m)
  # This should not be necessary anymore with fixing of normal:
  # NNF.setray!(a.v, scalar=m.v)
  r = canonize(a, SCALAR_PHASE; damping=damping)
  a = a ∘ r
  fc = factorize(a)
  NNF_tuple = COMPUTE_TWISS(fc.a1, SCALAR_LF)
  lf1 = LF(idxs[1], names[1], zero(first(s)), SA[zero(zero_phase),zero(zero_phase),zero(zero_phase)], NNF_tuple, PROCESS_ORBIT(fc.a0.v), PROCESS_SPIN(a))
  lf_table = LF_TABLE(lf1, N_ele)

  idx = 1
  if C == Colon || bl.line[1] in at
    lf_table[1] = lf1
    idx += 1
    if idx > N_ele
      return lf_table
    end
  end

  phase = MVector{3}(zero(zero_phase),zero(zero_phase),zero(zero_phase))
  len = length(bl.line)
  for i in 1:len
    b0.coords.v .= view(a.v, 1:6)'
    if Q != Nothing
      b0.coords.q[1] = a.q.q0
      b0.coords.q[2] = a.q.q1
      b0.coords.q[3] = a.q.q2
      b0.coords.q[4] = a.q.q3
    end
    track!(b0, bl.line[i])
    NNF.setray!(a.v; v=view(b0.coords.v, 1:6))
    if Q != Nothing
      NNF.setquat!(a.q; q=Quaternion(b0.coords.q[1], b0.coords.q[2], b0.coords.q[3], b0.coords.q[4]))
    end
    #s = lf_table.s[i] + S(bl.line[i].L)::S
    r = canonize(a, SCALAR_PHASE; phase=phase, damping=damping)
    a = a ∘ r
    if C == Colon || (i != 1 && bl.line[i] in at)
      fc = factorize(a)
      lfi = LF(idxs[idx], names[idx], s[idx], SA[copy(phase[1]), copy(phase[2]), copy(phase[3])], COMPUTE_TWISS(fc.a1, SCALAR_LF), PROCESS_ORBIT(fc.a0.v), PROCESS_SPIN(a))
      lf_table[idx] = lfi
      idx += 1
      if idx > N_ele
        break
      end
    end
  end
  return lf_table
end

# a
# b
# c


function twiss_tuple(beamline_index, name, s, phi, NNF_tuple::T, orbit, n::Nothing) where {T}
  if haskey(NNF_tuple, :eta) # NOT coasting
    # eta, zeta, and slip are APPROXIMATIONS
    # In coasting case all quantities are exact and in a0
    return (;
      beamline_index = beamline_index,
      name = name,
      s = s,
      phi_1 = phi[1],
      beta_1 = NNF_tuple.beta[1],
      alpha_1 = NNF_tuple.alpha[1],
      phi_2 = phi[2],
      beta_2 = NNF_tuple.beta[2],
      alpha_2 = NNF_tuple.alpha[2],
      phi_3 = phi[3],
      eta_1 = NNF_tuple.eta[1],
      etap_1 = NNF_tuple.eta[2],
      eta_2 = NNF_tuple.eta[3],
      etap_2 = NNF_tuple.eta[4],
      zeta_1 = NNF_tuple.zeta[1],
      zetap_1 = NNF_tuple.zeta[2],
      zeta_2 = NNF_tuple.zeta[3],
      zetap_2 = NNF_tuple.zeta[4],
      slip = NNF_tuple.approx_slip*sin(phi[3]*2*pi), # Approximation from EBB
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
  else
    return (;
      beamline_index = beamline_index,
      name = name,
      s = s,
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
  end
end

function twiss_tuple(beamline_index, name, s, phi, NNF_tuple::T, orbit, n) where {T}
  if haskey(NNF_tuple, :eta) # NOT coasting
    # eta, zeta, and slip are APPROXIMATIONS
    # In coasting case all quantities are exact and in a0
    return (;
      beamline_index = beamline_index,
      name = name,
      s = s,
      phi_1 = phi[1],
      beta_1 = NNF_tuple.beta[1],
      alpha_1 = NNF_tuple.alpha[1],
      phi_2 = phi[2],
      beta_2 = NNF_tuple.beta[2],
      alpha_2 = NNF_tuple.alpha[2],
      phi_3 = phi[3],
      eta_1 = NNF_tuple.eta[1],
      etap_1 = NNF_tuple.eta[2],
      eta_2 = NNF_tuple.eta[3],
      etap_2 = NNF_tuple.eta[4],
      zeta_1 = NNF_tuple.zeta[1],
      zetap_1 = NNF_tuple.zeta[2],
      zeta_2 = NNF_tuple.zeta[3],
      zetap_2 = NNF_tuple.zeta[4],
      slip = NNF_tuple.approx_slip*sin(phi[3]*2*pi), # Approximation from EBB
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
      n_x = n[1],
      n_y = n[2],
      n_z = n[3],
    )
  else
    return (;
      beamline_index = beamline_index,
      name = name,
      s = s,
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
      n_x = n[1],
      n_y = n[2],
      n_z = n[3],
    )
  end
end

function twiss_table(tt, N_ele)
  S = typeof(tt.s)
  V = typeof(tt.phi_1)
  T = typeof(tt.beta_1)
  U = typeof(tt.orbit_x)
  if !haskey(tt, :n_x)
    if haskey(tt, :eta_1)
      t = Table(
        beamline_index = Vector{Int}(undef, N_ele),
        name = Vector{String}(undef, N_ele),
        s = Vector{S}(undef, N_ele),
        phi_1 = Vector{V}(undef, N_ele),
        beta_1 = Vector{T}(undef, N_ele),
        alpha_1 = Vector{T}(undef, N_ele),
        phi_2 = Vector{V}(undef, N_ele),
        beta_2 = Vector{T}(undef, N_ele),
        alpha_2 = Vector{T}(undef, N_ele),
        phi_3 = Vector{V}(undef, N_ele),
        eta_1   = Vector{T}(undef, N_ele),
        etap_1  = Vector{T}(undef, N_ele),
        eta_2   = Vector{T}(undef, N_ele),
        etap_2  = Vector{T}(undef, N_ele),
        zeta_1  = Vector{T}(undef, N_ele),
        zetap_1 = Vector{T}(undef, N_ele),
        zeta_2  = Vector{T}(undef, N_ele),
        zetap_2 = Vector{T}(undef, N_ele),
        slip    = Vector{T}(undef, N_ele),
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
      return t
    else
      t = Table(
        beamline_index = Vector{Int}(undef, N_ele),
        name = Vector{String}(undef, N_ele),
        s = Vector{S}(undef, N_ele),
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
      return t
    end
  else
    W = typeof(tt.n_x)
    if haskey(tt, :eta_1)
      t = Table(
        beamline_index = Vector{Int}(undef, N_ele),
        name = Vector{String}(undef, N_ele),
        s = Vector{S}(undef, N_ele),
        phi_1 = Vector{V}(undef, N_ele),
        beta_1 = Vector{T}(undef, N_ele),
        alpha_1 = Vector{T}(undef, N_ele),
        phi_2 = Vector{V}(undef, N_ele),
        beta_2 = Vector{T}(undef, N_ele),
        alpha_2 = Vector{T}(undef, N_ele),
        phi_3 = Vector{V}(undef, N_ele),
        eta_1   = Vector{T}(undef, N_ele),
        etap_1  = Vector{T}(undef, N_ele),
        eta_2   = Vector{T}(undef, N_ele),
        etap_2  = Vector{T}(undef, N_ele),
        zeta_1  = Vector{T}(undef, N_ele),
        zetap_1 = Vector{T}(undef, N_ele),
        zeta_2  = Vector{T}(undef, N_ele),
        zetap_2 = Vector{T}(undef, N_ele),
        slip    = Vector{T}(undef, N_ele),
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
        n_x = Vector{W}(undef, N_ele),
        n_y = Vector{W}(undef, N_ele),
        n_z = Vector{W}(undef, N_ele),
      )
      return t
    else
      t = Table(
        beamline_index = Vector{Int}(undef, N_ele),
        name = Vector{String}(undef, N_ele),
        s = Vector{S}(undef, N_ele),
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
        n_x = Vector{W}(undef, N_ele),
        n_y = Vector{W}(undef, N_ele),
        n_z = Vector{W}(undef, N_ele),
      )
      return t
    end
  end
end

function de_moivre_tuple(beamline_index, name, s, phi, NNF_tuple, orbit, n::Nothing)
  return (;
    beamline_index = beamline_index,
    name = name,
    s = s,
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
end

function de_moivre_tuple(beamline_index, name, s, phi, NNF_tuple, orbit, n)
  return (;
    beamline_index = beamline_index,
    name = name,
    s = s,
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
    n_x = n[1],
    n_y = n[2],
    n_z = n[3],
  )
end

function de_moivre_table(dt, N_ele)
  S = typeof(dt.s)
  V = typeof(dt.phi_1)
  T = typeof(dt.H)
  U = typeof(dt.orbit_x)
  if !haskey(dt, :n_x)
    t = Table(
      beamline_index = Vector{Int}(undef, N_ele),
      name = Vector{String}(undef, N_ele),
      s = Vector{S}(undef, N_ele),
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
    return t
  else
    W = typeof(dt.n_x)
    t = Table(
      beamline_index = Vector{Int}(undef, N_ele),
      name = Vector{String}(undef, N_ele),
      s = Vector{S}(undef, N_ele),
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
      n_x = Vector{W}(undef, N_ele),
      n_y = Vector{W}(undef, N_ele),
      n_z = Vector{W}(undef, N_ele),
    )
    return t
  end
end
