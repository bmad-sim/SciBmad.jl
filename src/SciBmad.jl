module SciBmad

using PrecompileTools: @setup_workload, @compile_workload, @recompile_invalidations
using Reexport

@recompile_invalidations begin
  using NonlinearNormalForm: NonlinearNormalForm as NNF
  using TPSAInterface: TPSAInterface as TI
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

export twiss, find_closed_orbit, track!, track, Time, Yoshida, MatrixKick, BendKick, 
        SolenoidKick, DriftKick, Exact, Bunch

function track_a_particle!(coords, coords0, bl; use_KA=false, use_explicit_SIMD=false, scalar_params=true)
  coords .= coords0
  b0 = Bunch(reshape(coords, (1,6)))
  BTBL.check_bl_bunch!(bl, b0, false) # Do not notify
  track!(b0, bl; use_KA=use_KA, use_explicit_SIMD=use_explicit_SIMD, scalar_params=scalar_params)
  if b0.coords.state[1] != 0x1
    @warn "Particle lost in tracking"
  end
  return coords
end

function coast_check(bl, backend=DI.AutoForwardDiff())
  x0 = zeros(6)
  y = zeros(6)
  jac = zeros(6, 6)
  DI.value_and_jacobian!(track_a_particle!, y, jac, backend, x0, DI.Constant(bl))
  return view(jac, 6, :) ≈ SA[0, 0, 0, 0, 0, 1]
end

#=
# Hacky temporary solution for precompile
function coast_check(bl, backend::DI.AutoGTPSA{Nothing})
  coords = view(vars(GTPSA.desc_current), 1:6)
  b0 = Bunch(reshape(coords, (1,6)))
  BTBL.check_bl_bunch!(bl, b0, false) 
  track!(b0, bl; use_KA=false, use_explicit_SIMD=false)
  for i in 1:5 
    if b0.coords.v[1,6][i] != 0
      return false
    end
  end
  if b0.coords.v[1,6][6] ≈ 1
    return true
  else
    return false
  end
end
=#

function _co_res!(y, x, bl)
  track_a_particle!(y, x, bl)
  return y .= y .- x
end

function _co_res_coast!(y, x, bl)
  track_a_particle!(
    ArrayPartition(y, eltype(y)[0,0]), 
    SA[x[1], x[2], x[3], x[4], 0, 0],
    bl
  )
  return y .= y .- x
end

const CLOSED_ORBIT_FORWARDDIFF_PREP = (
  x = zeros(6);
  y = zeros(6);
  bl = Beamline([LineElement()]);
  DI.prepare_jacobian(_co_res!, y, DI.AutoForwardDiff(), x, DI.Constant(bl))
)

const CLOSED_ORBIT_FORWARDDIFF_PREP_COAST = (
  x = view(zeros(6), 1:4);
  y = view(zeros(6), 1:4);
  bl = Beamline([LineElement()]);
  DI.prepare_jacobian(_co_res_coast!, y, DI.AutoForwardDiff(), x, DI.Constant(bl))
)

function find_closed_orbit(
  bl::Beamline; 
  v0=zeros(6), 
  reltol=1e-13,
  abstol=1e-13, 
  max_iter=100, 
  backend=DI.AutoForwardDiff(),
  prep=CLOSED_ORBIT_FORWARDDIFF_PREP,
  prep_coast=CLOSED_ORBIT_FORWARDDIFF_PREP_COAST,
  lambda=1,
  coast::Val{C}=Val{Nothing}(),
) where {C}
  # First check if coasting, for this push a particle starting at 0 and see if
  # delta is a parameter
  v = zero(v0)
  if C == Nothing
    _coast = coast_check(bl, backend)
  else
    _coast = C
  end
  if _coast
    if !newton!(_co_res_coast!, view(v, 1:4), view(v0, 1:4), bl; reltol=reltol, abstol=abstol, max_iter=max_iter, backend=backend, prep=prep_coast, lambda=lambda).converged
      error("Closed orbit finder not converging")
    end
  else
    if !newton!(_co_res!, v, v0, bl; reltol=reltol, abstol=abstol, max_iter=max_iter, backend=backend, prep=prep, lambda=lambda).converged
      error("Closed orbit finder not converging")
    end
  end
  return v0, _coast
end


include("track.jl")
include("newton.jl")
include("twiss.jl")

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
    kicker = Sextupole(Kn0=1e-5, L=0.01)
    rf = RFCavity(L=1e-2, voltage=1e6, rf_frequency=1e6);
    thin = Multipole(Kn1L=1e-9); # Thin quad
    d3 = Drift(L=0.3);
    marker = Marker(); # nothing
    fodo_line = [qf, sf, d1, b, d2, qd, sd, d1, b, d2, rf, thin, marker, d3, kicker];
    fodo = Beamline(fodo_line, species_ref=Species("electron"), E_ref=18e9);
    # Track scalars
    b0s = Bunch(rand(4,6));
    BTBL.check_bl_bunch!(fodo, b0s, false); # Do not notify
    track!(b0s, fodo);
    # twiss
    # first order and second order
    co = find_closed_orbit(fodo);
    desc1 = Descriptor(6, 1);
    t = twiss(fodo);
    desc2 = Descriptor(6, 2);
    t = twiss(fodo, GTPSA_descriptor=desc2);
    # Coast, first order and second order
    rf.voltage = 0;
    co = find_closed_orbit(fodo);
    t = twiss(fodo; GTPSA_descriptor=desc1);
    t = twiss(fodo; GTPSA_descriptor=desc2);
    # Parameters, coast and no coast:
    descp = Descriptor(7, 1);
    qf.Kn1 = qf.Kn1 + vars(descp)[7];
    t = twiss(fodo; GTPSA_descriptor=descp);
    rf.voltage = 1e6;
    t = twiss(fodo; GTPSA_descriptor=descp);
  end
end

end
