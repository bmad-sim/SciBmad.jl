__precompile__(false)
module SciBmad

using PrecompileTools: @setup_workload, @compile_workload, @recompile_invalidations
using Reexport

@recompile_invalidations begin
  using KernelAbstractions: KernelAbstractions as KA
  using KernelAbstractions: @index, @kernel
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
  using DelimitedFiles
end

const BTBL = Base.get_extension(BeamTracking, :BeamTrackingBeamlinesExt)

export twiss, find_closed_orbit, track!, Time, Yoshida, MatrixKick, BendKick, BatchParam,
        SolenoidKick, DriftKick, Exact, Bunch, dynamic_aperture, track, rotate_spins, rotate_spins!

include("closed_orbit.jl") 
include("utils.jl")
include("track.jl")
include("newton.jl")
include("twiss.jl")
include("dynamic_aperture.jl")

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
    b0s = Bunch(rand(4,6), [1. 0. 0. 0.; 1. 0. 0. 0; 1. 0. 0. 0.; 1. 0. 0. 0.]);
    BTBL.check_bl_bunch!(fodo, b0s, false); # Do not notify
    track!(b0s, fodo);
    # twiss
    # first order and second order
    co = find_closed_orbit(fodo);
    desc1 = Descriptor(6, 1);
    t = twiss(fodo);
    desc2 = Descriptor(6, 2);
    t = twiss(fodo, GTPSA_descriptor=desc2);
    t = twiss(fodo, GTPSA_descriptor=desc2, spin=true);
    # Coast, first order and second order
    rf.voltage = 0;
    co = find_closed_orbit(fodo);
    t = twiss(fodo; GTPSA_descriptor=desc1);
    t = twiss(fodo; GTPSA_descriptor=desc2);
    t = twiss(fodo; GTPSA_descriptor=desc1, spin=true);
    t = twiss(fodo; GTPSA_descriptor=desc2, spin=true);
    # Parameters, coast and no coast:
    descp = Descriptor(7, 1);
    qf.Kn1 = qf.Kn1 + vars(descp)[7];
    t = twiss(fodo; GTPSA_descriptor=descp);
    t = twiss(fodo; GTPSA_descriptor=descp, spin=true);
    rf.voltage = 1e6;
    t = twiss(fodo; GTPSA_descriptor=descp);
    t = twiss(fodo; GTPSA_descriptor=descp, spin=true);
  end
end

end
