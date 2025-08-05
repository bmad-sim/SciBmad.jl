module SciBmad

using LinearAlgebra,
      TypedTables,
      Reexport,
      StaticArrays,
      NLsolve,
      Optimization,
      ForwardDiff,
      RecursiveArrayTools

@reexport using Beamlines
@reexport using BeamTracking
@reexport using NonlinearNormalForm
@reexport using GTPSA

using NonlinearNormalForm: NonlinearNormalForm as NNF
using DifferentiationInterface: DifferentiationInterface as DI

# Put AtomicAndPhysicalConstants in a box for now for safety
include("Constants.jl")
using .Constants: Constants, Species, massof, chargeof, C_LIGHT, isnullspecies
export Species

# BeamTrackingBeamlinesExt
const BTBL = Base.get_extension(BeamTracking, :BeamTrackingBeamlinesExt)

export twiss, find_closed_orbit

function find_closed_orbit(bl::Beamline; ftol=1e-13, autodiff=:forward, method=:newton, options...)
  function track_a_particle!(coords, coords0=coords; track_options...)
    coords .= coords0
    b0 = Bunch(reshape(coords, (1,6)))
    BTBL.check_bl_bunch!(bl, b0, false) # Do not notify
    track!(b0, bl; track_options...)
    return coords
  end
  
  # First check if coasting
  grad = DI.gradient(t->track_a_particle!(t)[6], AutoForwardDiff(), zeros(6))
  if all(view(grad, 1:5) .≈ 0) && grad[6] ≈ 1
    coast = true
  else
    coast = false
  end

  if coast
    sol = fixedpoint(zeros(4); ftol=ftol, autodiff=autodiff, method=method, options...) do c, c0
      track_a_particle!(ArrayPartition(c,eltype(c)[0,0]), SA[c0[1], c0[2], c0[3], c0[4], 0, 0], use_KA=false, use_explicit_SIMD=false)
    end
    #optf = OptimizationFunction((t,p)->sum(abs2, view(track_a_particle!([t[1],t[2],t[3],t[4],0,0]), 1:4) .- t), AD_backend)
    #optprob = OptimizationProblem(optf, zeros(4))
    #sol = solve(optprob, Optimization.LBFGS(); options...)
  else
    sol = fixedpoint(track_a_particle!, zeros(6); ftol=ftol, autodiff=autodiff, method=method, options...)
    #optf = OptimizationFunction((t,p)->sum(abs2, track_a_particle!(collect(t)) .- t), AD_backend)
    #optprob = OptimizationProblem(optf, zeros(6))
    #sol = solve(optprob, Optimization.LBFGS(); options...)
  end

  return sol
end


# Returns a Table of the Twiss parameters
# See Eq. 4.28 in EBB
function twiss(
  bl::Beamline; 
  GTPSA_descriptor=GTPSA.desc_current, 
  de_moivre=false,
  closed_orbit=find_closed_orbit(bl).zero)
  # First get closed orbit:
  if length(closed_orbit) == 4
    coast = true
    v0 = similar(closed_orbit, 6)
    v0[1:4] .= closed_orbit
    v0[5:6] .= 0
  else
    coast = false
    v0 = closed_orbit
  end
  # This will get the map and tell us if coasting, etc etc
  if GTPSA_descriptor.desc == C_NULL
    GTPSA_descriptor = Descriptor(6, 1)
  end
  Δv = @vars(GTPSA_descriptor)[1:6]
  v = reshape(v0 + Δv, (1, 6))
  b0 = Bunch(v)
  BTBL.check_bl_bunch!(bl, b0, false) # Do not notify
  linear = NNF.maxord(b0.coords.v[1]) == 1 ? true : false
  track!(b0, bl)
  m = DAMap(v=dropdims(b0.coords.v; dims=1))
  # function barrier
  S = typeof(bl.line[end].s)
  #return m, b0, bl, S, Val{linear}(), Val{de_moivre}()
  return _twiss(m, b0, bl, S, Val{linear}(), Val{de_moivre}())
end

function twiss_tuple(s, phi, NNF_tuple, orbit)
  return (;
    s = s,
    phi_a = phi[1],
    beta_a = NNF_tuple.beta[1],
    alpha_a = NNF_tuple.alpha[1],
    phi_b = phi[2],
    beta_b = NNF_tuple.beta[2],
    alpha_b = NNF_tuple.alpha[2],
    phi_c = phi[3],
    eta_a   = NNF_tuple.eta[1],
    eta_b   = NNF_tuple.eta[2],
    etap_a  = NNF_tuple.etap[1],
    etap_b  = NNF_tuple.etap[2],
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

function twiss_table(tt, N_ele)
  S = typeof(tt.s)
  T = typeof(tt.phi_a)
  t = Table(
    s = Vector{S}(undef, N_ele+1),
    phi_a = Vector{T}(undef, N_ele+1),
    beta_a = Vector{T}(undef, N_ele+1),
    alpha_a = Vector{T}(undef, N_ele+1),
    phi_b = Vector{T}(undef, N_ele+1),
    beta_b = Vector{T}(undef, N_ele+1),
    alpha_b = Vector{T}(undef, N_ele+1),
    phi_c = Vector{T}(undef, N_ele+1),
    eta_a = Vector{T}(undef, N_ele+1),
    eta_b = Vector{T}(undef, N_ele+1),
    etap_a = Vector{T}(undef, N_ele+1),
    etap_b = Vector{T}(undef, N_ele+1),
    gamma_c = Vector{T}(undef, N_ele+1),
    c11 = Vector{T}(undef, N_ele+1),
    c12 = Vector{T}(undef, N_ele+1),
    c21 = Vector{T}(undef, N_ele+1),
    c22 = Vector{T}(undef, N_ele+1),
    orbit_x = Vector{T}(undef, N_ele+1),
    orbit_px = Vector{T}(undef, N_ele+1),
    orbit_y = Vector{T}(undef, N_ele+1),
    orbit_py = Vector{T}(undef, N_ele+1),
    orbit_z = Vector{T}(undef, N_ele+1),
    orbit_pz = Vector{T}(undef, N_ele+1),
  )
  t[1] = tt
  return t
end

function de_moivre_tuple(s, phi, NNF_tuple, orbit)
  return (;
    s = s,
    phi_a = phi[1],
    phi_b = phi[2],
    phi_c = phi[3],
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
  T = typeof(dt.phi_a)
  U = typeof(dt.H)
  t = Table(
    s = Vector{S}(undef, N_ele+1),
    phi_a = Vector{T}(undef, N_ele+1),
    phi_b = Vector{T}(undef, N_ele+1),
    phi_c = Vector{T}(undef, N_ele+1),
    H = Vector{U}(undef, N_ele+1),
    B = Vector{U}(undef, N_ele+1),
    E = Vector{U}(undef, N_ele+1),
    K = Vector{U}(undef, N_ele+1),
    orbit_x = Vector{T}(undef, N_ele+1),
    orbit_px = Vector{T}(undef, N_ele+1),
    orbit_y = Vector{T}(undef, N_ele+1),
    orbit_py = Vector{T}(undef, N_ele+1),
    orbit_z = Vector{T}(undef, N_ele+1),
    orbit_pz = Vector{T}(undef, N_ele+1),
  )
  t[1] = dt
  return t
end

function _twiss(m::DAMap, b0::Bunch, bl::Beamline, S::Type, ::Val{linear}, ::Val{de_moivre}) where {linear, de_moivre}
  # Ripken-Wolski-Forest de Moivre coupling formalism
  # If linear is true, then we do not need to do any factorization
  # Else we must factorize at every element
  
  COMPUTE_TWISS = de_moivre ? compute_de_moivre : compute_sagan_rubin
  LF = !de_moivre ? twiss_tuple : de_moivre_tuple 
  LF_TABLE = !de_moivre ? twiss_table : de_moivre_table

  s::S = 0

  if linear
    N_ele = length(bl.line)
    a = normal(m)
    NNF.setray!(a.v, scalar=m.v)
    r = fast_canonize(a)
    a = a ∘ r
    NNF_tuple = COMPUTE_TWISS(a, Val{linear}())
    T = eltype(eltype(typeof(getproperty(NNF_tuple, first(propertynames(NNF_tuple))))))
    lf1 = LF(S(s), T.(SA[0,0,0]), COMPUTE_TWISS(a, Val{linear}()), scalar.(view(a.v, 1:6)))
    lf_table = LF_TABLE(lf1, N_ele)
    phase = MVector{3,T}(0,0,0)
    for i in 1:N_ele
      phase .= 0
      b0.coords.v .= view(a.v, 1:6)'
      track!(b0, bl.line[i])
      s = lf_table.s[i] + S(bl.line[i].L)::S
      NNF.setray!(a.v; v=view(b0.coords.v, 1:6))
      r = fast_canonize(a; phase=phase)
      a = a ∘ r
      old_phases = SA[lf_table.phi_a[i], lf_table.phi_b[i], lf_table.phi_c[i]]
      lfi = LF(s, old_phases+phase, COMPUTE_TWISS(a, Val{linear}()), scalar.(view(a.v, 1:6)))
      lf_table[i+1] = lfi
    end
  else
    # In the nonlinear case, we need to track the FULL a, and 
    # factorize each element. In this case, if there is coasting, 
    # H will NOT contain the dispersion because we include that 
    # in the parameter-dependent transformation
    
    # The lattice functions must be evaluated for each a1 specifically
    error("Nonlinear twiss calculation currently being developed")
  end

  return lf_table
end

  #=
  t = Table(s=s, phi_x=phase[1,:], phi_y=phase[2,:], phi_z=phase[3,:],
            beta_xx=map(t->t.E[1][1,1], lf),
            alpha_xx=map(t->t.E[1][1,2], lf),
            beta_yy=map(t->t.E[2][3,3], lf),
            alpha_yy=map(t->t.E[2][3,4], lf),
            eta_x=map(t->t.H[3][1,6], lf),
            eta_y=map(t->t.H[3][3,6], lf),
  )
  =#
end
