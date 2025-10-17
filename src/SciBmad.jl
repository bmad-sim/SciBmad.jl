module SciBmad

using PrecompileTools: @setup_workload, @compile_workload, @recompile_invalidations
using Reexport

@recompile_invalidations begin
  using NonlinearNormalForm: NonlinearNormalForm as NNF
  using DifferentiationInterface: DifferentiationInterface as DI
  using LinearAlgebra,
      TypedTables,
      StaticArrays,
      ForwardDiff,
      RecursiveArrayTools
  using BeamTracking
  @reexport using Beamlines
  @reexport using NonlinearNormalForm
  @reexport using GTPSA
  @reexport using AtomicAndPhysicalConstants
end

const BTBL = Base.get_extension(BeamTracking, :BeamTrackingBeamlinesExt)

export twiss, find_closed_orbit, track!, track, Time

function fast_coast_check(bl; use_KA=false, use_explicit_SIMD=false)
  # Just check if dpz/dz == 0
  coords = @MVector [ForwardDiff.Dual(0.,0.),
                     ForwardDiff.Dual(0.,0.),
                     ForwardDiff.Dual(0.,0.),
                     ForwardDiff.Dual(0.,0.),
                     ForwardDiff.Dual(0.,1.),
                     ForwardDiff.Dual(0.,0.)]
  b0 = Bunch(coords)
  BTBL.check_bl_bunch!(bl, b0, false) # Do not notify
  track!(b0, bl; use_KA=use_KA, use_explicit_SIMD=use_explicit_SIMD)
  return b0.coords.v[1,6].partials[1] == 0 # true if coasting, false if not
end

function track_a_particle!(coords, coords0, bl; use_KA=false, use_explicit_SIMD=false)
  coords .= coords0
  b0 = Bunch(reshape(coords, (1,6)))
  BTBL.check_bl_bunch!(bl, b0, false) # Do not notify
  track!(b0, bl; use_KA=use_KA, use_explicit_SIMD=use_explicit_SIMD)
  return coords
end


function _co_res!(y, x, bl)
  track_a_particle!(y, x, bl)
  return y .= y .- x
end

function _co_res_coast!(y, x, bl)
  track_a_particle!(
    ArrayPartition(y, MVector{2,eltype(y)}(0,0)), 
    SA[x[1], x[2], x[3], x[4], 0, 0],
    bl
  )
  return y .= y .- x
end

const CLOSED_ORBIT_FORWARDDIFF_PREP = (
  x = @MVector zeros(6);
  y = @MVector zeros(6);
  bl = Beamline([LineElement()]);
  DI.prepare_jacobian(_co_res!, y, DI.AutoForwardDiff(), x, DI.Constant(bl))
)

const CLOSED_ORBIT_FORWARDDIFF_PREP_COAST = (
  x = view(@MVector(zeros(6)), 1:4);
  y = view(@MVector(zeros(6)), 1:4);
  bl = Beamline([LineElement()]);
  DI.prepare_jacobian(_co_res_coast!, y, DI.AutoForwardDiff(), x, DI.Constant(bl))
)

function find_closed_orbit(
  bl::Beamline; 
  v0=zero(MVector{6,Float64}), 
  abstol=1e-11, 
  max_iter=100, 
  #backend=DI.AutoForwardDiff()
)
  # First check if coasting, for this push a particle starting at 0 and see if
  # delta is a parameter
  v = zero(v0)
  coast = fast_coast_check(bl)
  if coast
    newton!(_co_res_coast!, view(v, 1:4), view(v0, 1:4), bl; prep=CLOSED_ORBIT_FORWARDDIFF_PREP_COAST)
  else
    newton!(_co_res!, v, v0, bl; prep=CLOSED_ORBIT_FORWARDDIFF_PREP)
  end
  return v0, coast
end

# Returns a Table of the Twiss parameters
# See Eq. 4.28 in EBB
function twiss(
  bl::Beamline; 
  GTPSA_descriptor=Descriptor(6, 1), # GTPSA.desc_current, 
  de_moivre=false,
  co_info=find_closed_orbit(bl)
)
  v0 = co_info[1]
  coast = co_info[2]
  # First get closed orbit:
  # This will get the map and tell us if coasting, etc etc
  if GTPSA_descriptor.desc == C_NULL
    GTPSA_descriptor = Descriptor(6, 1)
  end
  Δv = vars(GTPSA_descriptor)[1:6]
  for (Δvi, v0i) in zip(Δv, v0)
    Δvi[0] = v0i
  end
  b0 = Bunch(reshape(Δv, (1,6)))
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
    phi_1 = phi[1],
    beta_1 = NNF_tuple.beta[1],
    alpha_1 = NNF_tuple.alpha[1],
    phi_2 = phi[2],
    beta_2 = NNF_tuple.beta[2],
    alpha_2 = NNF_tuple.alpha[2],
    phi_3 = phi[3],
    eta_1   = NNF_tuple.eta[1],
    eta_2   = NNF_tuple.eta[2],
    etap_1  = NNF_tuple.etap[1],
    etap_2  = NNF_tuple.etap[2],
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
  T = typeof(tt.phi_1)
  t = Table(
    s = Vector{S}(undef, N_ele+1),
    phi_1 = Vector{T}(undef, N_ele+1),
    beta_1 = Vector{T}(undef, N_ele+1),
    alpha_1 = Vector{T}(undef, N_ele+1),
    phi_2 = Vector{T}(undef, N_ele+1),
    beta_2 = Vector{T}(undef, N_ele+1),
    alpha_2 = Vector{T}(undef, N_ele+1),
    phi_3 = Vector{T}(undef, N_ele+1),
    eta_1 = Vector{T}(undef, N_ele+1),
    eta_2 = Vector{T}(undef, N_ele+1),
    etap_1 = Vector{T}(undef, N_ele+1),
    etap_2 = Vector{T}(undef, N_ele+1),
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
  T = typeof(dt.phi_1)
  U = typeof(dt.H)
  t = Table(
    s = Vector{S}(undef, N_ele+1),
    phi_1 = Vector{T}(undef, N_ele+1),
    phi_2 = Vector{T}(undef, N_ele+1),
    phi_3 = Vector{T}(undef, N_ele+1),
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
      old_phases = SA[lf_table.phi_1[i], lf_table.phi_2[i], lf_table.phi_3[i]]
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

include("track.jl")
include("newton.jl")

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


@setup_workload begin
  
  @compile_workload begin   
    # We want to compile drift-kick-drift, matrix-kick-matrix
    # and solenoid kick for different numbers of multipoles
    # Bend too, but that is not implemented yet.
    qf = Quadrupole(Kn1=0.36, L=0.5); # Matrix kick, 1 multipole
    sf = Sextupole(Kn2=0.1, L=0.2);   # Drift kick, 1 multipole
    d1 = Drift(L=0.3, Kn3=1e-4, Kn4=1e-5); # Drift kick, 2 multipoles
    d2 = Drift(L=0.3, Ksol=1e-6); # Solenoid kick, 1 multipole
    b  = SBend(L=6.0, angle=pi/132); # Bend
    qd = Quadrupole(Kn1=-0.36, Ks20=1e-3,L=0.5); # matrix kick, 2 multipoles
    sd = Sextupole(Kn2=-0.1, Ksol=1e-6, L=0.2); # solenoid-kick, 2 multipoles
    rf = RFCavity(L=1e-2, voltage=1e-3, rf_frequency=1e6);
    thin = Multipole(Kn1L=1e-9); # Thin quad
    d3 = Drift(L=0.3);
    marker = Marker(); # nothing
    fodo_line = [qf, sf, d1, b, d2, qd, sd, d1, b, d2, rf, thin, marker, d3];
    fodo = Beamline(fodo_line, species_ref=Species("electron"), E_ref=18e9);
    # Track scalars/.=    b0s = Bunch(rand(4,6));
    BTBL.check_bl_bunch!(fodo, b0s, false); # Do not notify
    track!(b0s, fodo);
    # twiss + normal
    co = find_closed_orbit(fodo);
    d2 = Descriptor(6, 2);
    b0 = Bunch(vars(d2));
    BTBL.check_bl_bunch!(fodo, b0, false); # Do not notify
    track!(b0, fodo);
    m = DAMap(v=b0.coords.v);
    a = normal(m);
    # Coast:
    rf.voltage = 0;
    co = find_closed_orbit(fodo);
    b0.coords.v .= vars(d2)';
    track!(b0, fodo);
    m = DAMap(v=b0.coords.v);
    a = normal(m);
    t = twiss(fodo);
  end
end

end
