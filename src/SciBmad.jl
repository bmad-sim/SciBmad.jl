module SciBmad
using PrecompileTools: @setup_workload, @compile_workload, @recompile_invalidations
using Reexport

@recompile_invalidations begin
  using BatchSolve: Constant, Cache, ConstantOrCache, AutoBatch, 
    newton, newton!, brent, brent!, RETCODE_SUCCESS, RETCODE_FAILURE, 
    RETCODE_MAXITER, newton, newton!, BatchSolve
  using KernelAbstractions: KernelAbstractions as KA
  using KernelAbstractions: @index, @kernel, @Const
  using NonlinearNormalForm: NonlinearNormalForm as NNF
  using TPSAInterface: TPSAInterface as TI
  using DifferentiationInterface: DifferentiationInterface as DI
  using LinearAlgebra,
        DataFrames,
        OrderedCollections,
        StaticArrays,
        ForwardDiff,
        DelimitedFiles,
        PrettyTables
  @reexport using ADTypes      
  @reexport using BeamTracking
  @reexport using Beamlines
  @reexport using NonlinearNormalForm
  @reexport using GTPSA
  @reexport using AtomicAndPhysicalConstants
  @reexport using FundamentalFrequencies
end

const BTBL = Base.get_extension(BeamTracking, :BeamTrackingBeamlinesExt)

export  twiss, 
        find_closed_orbit, 
        track!,
        dynamic_aperture,
        track,
        track_spin,
        track_spin!,
        TrackingResult,
        TrackingConfig,
        Twiss,
        TwissSummary,
        getterm,
        AmplitudeDependentValue,
        grad,
        val,
        jac
        
include("closed_orbit.jl") 
include("track.jl")
include("adv.jl")
include("twiss/twiss_types.jl")
include("twiss/twiss_operators.jl")
include("twiss/twiss_df.jl")
include("twiss/twiss.jl")
include("twiss/twiss_map.jl")
include("dynamic_aperture.jl")
include("experimental/Experimental.jl")
include("utils.jl")

@setup_workload begin
  
  @compile_workload begin   
    # We want to compile drift-kick-drift, matrix-kick-matrix
    # and solenoid kick for different numbers of multipoles
    # Bend too, but that is not implemented yet.
    qf = Quadrupole(Kn1=0.36, L=0.25); # Matrix kick, 1 multipole
    qf1 = Quadrupole(Kn1=0.36, Kn20=1e-3, L=0.25) # Matrix kick, 2 multipoles
    sf = Sextupole(Kn2=0.1, L=0.2);   # Drift kick, 1 multipole
    d1 = Drift(L=0.3, Kn3=1e-4, Kn4=1e-5); # Drift kick, 2 multipoles
    d2 = Drift(L=0.3, Ksol=1e-6); # Solenoid kick, 1 multipole
    b1  = SBend(L=6.0/2, angle=pi/132/2); # Bend
    b2  = SBend(L=6.0/2, angle=pi/132/2, e1=1e-5, e2=1e-4); # Bend
    qd1 = Quadrupole(Kn1=-0.36, L=0.25); # matrix kick, 1 multipoles w/ rotation
    qd = Quadrupole(Kn1=-0.36, Ks20=1e-3,L=0.25); # matrix kick w/ rotation, 2 multipoles
    sd = Sextupole(Kn2=-0.1, Ksol=1e-6, L=0.2); # solenoid-kick, 2 multipoles
    kicker = Sextupole(Kn0=1e-5, L=0.01)
    rf = RFCavity(L=1e-2, voltage=1e6, rf_frequency=1e6, zero_phase=PhaseRef.AboveTransition);
    thin = Multipole(Kn1L=1e-9); # Thin quad
    d3 = Drift(L=0.3);
    p = Patch(dx=1e-9, dy_rot=-1e-9, dz_rot=1e-9)
    marker = Marker(); # nothing
    fodo_line = [qf, qf1, sf, d1, b1, b2, d2, qd1, qd, sd, d1, b1, b2, d2, rf, thin, marker, d3, kicker, p];
    fodo = Beamline(fodo_line, species_ref=Species("electron"), E_ref=18e9);
    # track
    res = track(fodo, v0=rand(4, 6) .* 1e-5)
    res = track(fodo, v0=rand(4, 6) .* 1e-5, spin=true)
    # twiss
    # first order and second order
    tw = twiss(fodo)
    tw = twiss(fodo, rf_on=false)
    tw = twiss(fodo, order=2)
    tw = twiss(fodo, order=2, rf_on=false)
    tw = twiss(fodo, chrom=2, rf_on=false)
    tw = twiss(fodo, spin=true)
    tw = twiss(fodo, rf_on=false, spin=true)
    tw = twiss(fodo, order=2, spin=true)
    tw = twiss(fodo, order=2, rf_on=false, spin=true)
    tw = twiss(fodo, chrom=2, rf_on=false, spin=true)

    # Parametric normal form example
    @elements begin
      qf = Quadrupole(Kn1=DefExpr(c -> c.kqf), L=0.5)
      sf = Sextupole(Kn2=DefExpr(c-> c.ksf), L=0.2)
      d = Drift(L=0.1)
      b = SBend(L=1.2, angle=pi/132)
      qd = Quadrupole(Kn1=DefExpr(c -> c.kqd), L=0.5)
      sd = Sextupole(Kn2=DefExpr(c -> c.ksd), L=0.2)
    end

    fodo = Beamline([qf, sf, b, d, qd, sd, b, d], 
            species_ref=Species("electron"), pc_ref=18e9)

    fodo.context.kqf = 0.36
    fodo.context.kqd = -0.36
    fodo.context.ksf = 1.2
    fodo.context.ksd = -1.2
    dnf = Descriptor([1, 1, 1, 1, 1, 1], 2, [1, 1], 1)
    dk = params(dnf)
    fodo.context.kqf += dk[1]
    fodo.context.kqd += dk[2]
    tw = twiss(fodo, GTPSA_descriptor=dnf)

    grad(tw.q1) # [dq1/dkqf, dq1/dkqd]
    grad(tw.q2) # [dq2/dkqf, dq2/dkqd]
    grad(tw.beta1[1]) # [dbeta1/dkf, dbeta1/dkd]
    grad(tw.beta2[1]) # [dbeta2/dkf, dbeta2/dkd]

    scalarize!(fodo)
    dnf2 = Descriptor([1, 1, 1, 1, 1, 2], 3, [1, 1], 1)
    dk = params(dnf2)
    fodo.context.ksf += dk[1]
    fodo.context.ksd += dk[2]

    tw = twiss(fodo, GTPSA_descriptor=dnf2)
    chromx = getterm(tw.q1, delta=1, as_taylor_series=true)
    chromy = getterm(tw.q2, delta=1, as_taylor_series=true)

    scalarize!(fodo)

    dnf3 = Descriptor([2, 2, 2, 2, 2, 2], 3, [1], 1)
    dk = params(dnf3)
    fodo.context.ksf += dk[1]

    tw = twiss(fodo, GTPSA_descriptor=dnf3, cols=["dh2000"])
    grad(tw.dh2000[1])
  end
end

end
