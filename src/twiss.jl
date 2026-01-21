# Returns a Table of the Twiss parameters
# See Eq. 4.28 in EBB
function twiss(
  bl::Beamline; 
  GTPSA_descriptor=nothing, #Descriptor(6, 1),
  de_moivre=false,
  co_info=find_closed_orbit(bl),
  symplectic_tol=1e-8, # Tolerance below which to include damping
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

  # If it's greater than 6 variables, assume a parameter is
  # set in lattice and have to use AutoGTPSA. Else use faster ForwardDiff
  #=
  if isnothing(co_info)
    if GTPSA.numnn(GTPSA_descriptor) == 6
      co_info = find_closed_orbit(bl)
    else
      old_desc = GTPSA.desc_current
      GTPSA.desc_current = GTPSA_descriptor
      co_info = find_closed_orbit(bl; backend=DI.AutoGTPSA(GTPSA_descriptor), prep=nothing, prep_coast=nothing)
      GTPSA.desc_current = old_desc
    end
  end
=#
  v0 = co_info[1]
  coast = co_info[2]

  # Track once through and construct a DAMap
  Δv = vars(GTPSA_descriptor)[1:6]
  for (Δvi, v0i) in zip(Δv, v0)
    Δvi[0] = v0i
  end
  b0 = Bunch(reshape(Δv, (1,6)))
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
  m = DAMap(nv=nv, np=np, v=view(dropdims(b0.coords.v; dims=1), 1:nv))

  # Check if symplectic or not
  damping = norm(NNF.checksymp(NNF.jacobian(m))) > symplectic_tol

  if mo > 1 && (coast || nn > 6)
    zero_LF = TI.init_tps(numtype, init)
  else
    zero_LF = zero(numtype)
  end

  # Type of the PHASES
  if coast || mo > 1 && nn > 6
    zero_phase = TI.init_tps(numtype, init)
  else
    zero_phase = zero(numtype)
  end
  
  # Type of the ORBIT
  if coast || nn > 6
    zero_orbit = TI.init_tps(numtype, init)
  else
    zero_orbit = zero(numtype)
  end

  # Type of the s coordinate
  zero_s = zero(bl.line[end].s)

  # function barrier
  return _twiss(m, b0, bl, Val{de_moivre}(), damping, zero_LF, zero_phase, zero_orbit, zero_s)
end
  
function _twiss(
  m::DAMap, 
  b0::Bunch, 
  bl::Beamline, 
  ::Val{de_moivre}, 
  damping,
  zero_LF::T, 
  zero_phase::V,
  zero_orbit::U, 
  zero_s::S
) where {de_moivre, T, V, U, S}
  # Ripken-Wolski-Forest de Moivre coupling formalism
  # If linear is true, then we do not need to do any factorization
  # Else we must factorize at every element
  
  # These checks should all be static ================================
  # if compiler isn't a dumb dumb, which it definitely can be
  COMPUTE_TWISS = de_moivre ? compute_de_moivre : compute_sagan_rubin
  LF = !de_moivre ? twiss_tuple : de_moivre_tuple 
  LF_TABLE = !de_moivre ? twiss_table : de_moivre_table
  SCALAR_LF = TI.is_tps_type(T) isa TI.IsTPSType ? Val{false}() : Val{true}()
  SCALAR_PHASE = TI.is_tps_type(V) isa TI.IsTPSType ? Val{false}() : Val{true}()
  SCALAR_ORBIT = TI.is_tps_type(U) isa TI.IsTPSType ? Val{false}() : Val{true}()

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

  s::S = zero_s
  N_ele = length(bl.line)
  a = normal(m)
  NNF.setray!(a.v, scalar=m.v)
  r = canonize(a, SCALAR_PHASE; damping=damping)
  a = a ∘ r
  a0, a1, a2 = factorize(a)
  NNF_tuple = COMPUTE_TWISS(a1, SCALAR_LF)
  lf1 = LF(S(s), SA[zero(zero_phase),zero(zero_phase),zero(zero_phase)], NNF_tuple, PROCESS_ORBIT(a0.v))
  lf_table = LF_TABLE(lf1, N_ele)
  phase = MVector{3}(zero(zero_phase),zero(zero_phase),zero(zero_phase))
  for i in 1:N_ele
    if SCALAR_PHASE isa Val{true}
      phase .= 0
    else
      TI.clear!(phase[1]); TI.clear!(phase[2]); TI.clear!(phase[3]);
    end 
    b0.coords.v .= view(a.v, 1:6)'
    track!(b0, bl.line[i])
    NNF.setray!(a.v; v=view(b0.coords.v, 1:6))
    s = lf_table.s[i] + S(bl.line[i].L)::S
    r = canonize(a, SCALAR_PHASE; phase=phase, damping=damping)
    a = a ∘ r
    a0, a1, a2 = factorize(a)
    old_phases = SA[lf_table.phi_1[i], lf_table.phi_2[i], lf_table.phi_3[i]]
    lfi = LF(s, old_phases+phase, COMPUTE_TWISS(a1, SCALAR_LF), PROCESS_ORBIT(a0.v))
    lf_table[i+1] = lfi
  end
  return lf_table
end


function twiss_tuple(s, phi, NNF_tuple::T, orbit) where {T}
  if haskey(NNF_tuple, :eta) # NOT coasting
    # eta, zeta, and slip are APPROXIMATIONS
    # In coasting case all quantities are exact and in a0
    return (;
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

function twiss_table(tt, N_ele)
  S = typeof(tt.s)
  V = typeof(tt.phi_1)
  T = typeof(tt.beta_1)
  U = typeof(tt.orbit_x)
  if haskey(tt, :eta_1)
    t = Table(
      s = Vector{S}(undef, N_ele+1),
      phi_1 = Vector{V}(undef, N_ele+1),
      beta_1 = Vector{T}(undef, N_ele+1),
      alpha_1 = Vector{T}(undef, N_ele+1),
      phi_2 = Vector{V}(undef, N_ele+1),
      beta_2 = Vector{T}(undef, N_ele+1),
      alpha_2 = Vector{T}(undef, N_ele+1),
      phi_3 = Vector{V}(undef, N_ele+1),
      eta_1   = Vector{T}(undef, N_ele+1),
      etap_1  = Vector{T}(undef, N_ele+1),
      eta_2   = Vector{T}(undef, N_ele+1),
      etap_2  = Vector{T}(undef, N_ele+1),
      zeta_1  = Vector{T}(undef, N_ele+1),
      zetap_1 = Vector{T}(undef, N_ele+1),
      zeta_2  = Vector{T}(undef, N_ele+1),
      zetap_2 = Vector{T}(undef, N_ele+1),
      slip    = Vector{T}(undef, N_ele+1),
      gamma_c = Vector{T}(undef, N_ele+1),
      c11 = Vector{T}(undef, N_ele+1),
      c12 = Vector{T}(undef, N_ele+1),
      c21 = Vector{T}(undef, N_ele+1),
      c22 = Vector{T}(undef, N_ele+1),
      orbit_x = Vector{U}(undef, N_ele+1),
      orbit_px = Vector{U}(undef, N_ele+1),
      orbit_y = Vector{U}(undef, N_ele+1),
      orbit_py = Vector{U}(undef, N_ele+1),
      orbit_z = Vector{U}(undef, N_ele+1),
      orbit_pz = Vector{U}(undef, N_ele+1),
    )
    t[1] = tt
    return t
  else
    t = Table(
      s = Vector{S}(undef, N_ele+1),
      phi_1 = Vector{V}(undef, N_ele+1),
      beta_1 = Vector{T}(undef, N_ele+1),
      alpha_1 = Vector{T}(undef, N_ele+1),
      phi_2 = Vector{V}(undef, N_ele+1),
      beta_2 = Vector{T}(undef, N_ele+1),
      alpha_2 = Vector{T}(undef, N_ele+1),
      phi_3 = Vector{V}(undef, N_ele+1),
      gamma_c = Vector{T}(undef, N_ele+1),
      c11 = Vector{T}(undef, N_ele+1),
      c12 = Vector{T}(undef, N_ele+1),
      c21 = Vector{T}(undef, N_ele+1),
      c22 = Vector{T}(undef, N_ele+1),
      orbit_x = Vector{U}(undef, N_ele+1),
      orbit_px = Vector{U}(undef, N_ele+1),
      orbit_y = Vector{U}(undef, N_ele+1),
      orbit_py = Vector{U}(undef, N_ele+1),
      orbit_z = Vector{U}(undef, N_ele+1),
      orbit_pz = Vector{U}(undef, N_ele+1),
    )
    t[1] = tt
    return t
  end
end

function de_moivre_tuple(s, phi, NNF_tuple, orbit)
  return (;
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

function de_moivre_table(dt, N_ele)
  S = typeof(dt.s)
  V = typeof(dt.phi_1)
  T = typeof(dt.H)
  U = typeof(dt.orbit_x)
  t = Table(
    s = Vector{S}(undef, N_ele+1),
    phi_1 = Vector{V}(undef, N_ele+1),
    phi_2 = Vector{V}(undef, N_ele+1),
    phi_3 = Vector{V}(undef, N_ele+1),
    H = Vector{T}(undef, N_ele+1),
    B = Vector{T}(undef, N_ele+1),
    E = Vector{T}(undef, N_ele+1),
    K = Vector{T}(undef, N_ele+1),
    orbit_x = Vector{U}(undef, N_ele+1),
    orbit_px = Vector{U}(undef, N_ele+1),
    orbit_y = Vector{U}(undef, N_ele+1),
    orbit_py = Vector{U}(undef, N_ele+1),
    orbit_z = Vector{U}(undef, N_ele+1),
    orbit_pz = Vector{U}(undef, N_ele+1),
  )
  t[1] = dt
  return t
end
