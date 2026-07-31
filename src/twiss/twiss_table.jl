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
    phi1 = Vector{V}(undef, N_ele),
    beta1 = Vector{T}(undef, N_ele),
    alpha1 = Vector{T}(undef, N_ele),
    phi2 = Vector{V}(undef, N_ele),
    beta2 = Vector{T}(undef, N_ele),
    alpha2 = Vector{T}(undef, N_ele),
    phi3 = Vector{V}(undef, N_ele),
    gamma_c = Vector{T}(undef, N_ele),
    c11 = Vector{T}(undef, N_ele),
    c12 = Vector{T}(undef, N_ele),
    c21 = Vector{T}(undef, N_ele),
    c22 = Vector{T}(undef, N_ele),
    x = Vector{U}(undef, N_ele),
    px = Vector{U}(undef, N_ele),
    y = Vector{U}(undef, N_ele),
    py = Vector{U}(undef, N_ele),
    z = Vector{U}(undef, N_ele),
    pz = Vector{U}(undef, N_ele),
    dpx_2
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