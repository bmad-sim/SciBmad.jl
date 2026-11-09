using SciBmad
using GTPSA
using DelimitedFiles
using Test

@testset "SciBmad.jl" begin
    # See SciBmad's subpackages for all of the unit tests.
    # Those are where most of the functionalities are, hence
    # where most of the tests are.

    # Here we have regression tests for SciBmad-level functionality.
    # Reference Twiss data was generated with Tao (Bmad) using:
    #   show lat -pipe -beginning * -at beta_a@e24.16 -at beta_b@e24.16 -at alpha_a@e24.16 -at alpha_b@e24.16
    # With -beginning, Tao reports Twiss at the BEGINNING of each element, matching
    # SciBmad's twiss().table convention. The BEGINNING pseudo-element row (index 0)
    # is skipped; Tao rows 1..N correspond to SciBmad's twiss().table[1..N].
    # Note: SciBmad's twiss().table has N+1 entries — entries 1..N for element beginnings
    # plus one "END OF BEAMLINE" entry (ring closure). The reference covers entries 1..N;
    # ring closure is verified separately below.
    #
    # Tao binary: /Users/matthewsignorelli/miniconda3/envs/condamatt/bin/tao
    # Bmad lattice: /Users/matthewsignorelli/Documents/ESR/v6.3.1/08-21-24/yunhai's soln/esr.bmad
    # Pipe format uses semicolon delimiters; @e24.16 gives 24-char 16-decimal scientific notation.
    # Orbit attributes: orbit_x, orbit_px, orbit_y, orbit_py, orbit_z, orbit_pz (NOT orbit.vec.N)
    # Kicker: set element cv01_7 kick = <value>   (VKicker in Bmad ↔ Kicker.Ks0L in SciBmad)

  @testset "Bengtsson h columns of thin multipoles" begin
      # Thin octupole: h2200 = K3L β²/16. Thin sextupole: h3000 = K2L β^(3/2)/(12√2), h2100 = 3 h3000.
    qf = Multipole(Kn1L=0.8); qd = Multipole(Kn1L=-0.7); d = Drift(L=1.0)
    cell(e...) = Beamline([e..., qf, d, qd, d]; species_ref=Species("electron"), pc_ref=1e9)
    h(bl, col, order, tps) = scalar(getproperty(twiss(bl; rf_on=false, order=order, at=[1],
        cols=[col], as_taylor_series=tps), Symbol(col))[1])
    β = twiss(cell(); rf_on=false, at=[1], cols=["beta1"]).beta1[1]
    oct = cell(Multipole(Kn3L=1000.0)); sx = cell(Multipole(Kn2L=10.0))
    for tps in (false, true)
        # The bare cell's exact drifts also contribute to h2200
      @test h(oct, "h2200", 3, tps) - h(cell(), "h2200", 3, tps) ≈ 1000 * β^2 / 16
      @test h(sx, "h3000", 2, tps) ≈ 10 * β^1.5 / (12 * sqrt(2))
      @test h(sx, "h2100", 2, tps) ≈ 10 * β^1.5 / (4 * sqrt(2))
    end
  end

  @testset "ESR (18 GeV EIC-ESR) Twiss vs Tao/Bmad" begin
    ref = readdlm(
        joinpath(@__DIR__, "reference", "esr-twiss-tao.csv"),
        ',', Float64; skipstart=1
    )
    # Columns: beta_a (→beta_1), beta_b (→beta_2), alpha_a (→alpha_1), alpha_b (→alpha_2)

    ring = include(joinpath(@__DIR__, "lattices", "esr.jl"))

    tw = twiss(ring)

    # N+1 entries: N element beginnings + END OF BEAMLINE
    N = size(ref, 1)
    @test length(tw.beta1) == N + 1

    # Beta functions: element-wise relative tolerance (beta ranges 0.59–702 m)
    @test all(isapprox.(tw.beta1[1:N], ref[:,1]; rtol=1e-5, atol=0))
    @test all(isapprox.(tw.beta2[1:N], ref[:,2]; rtol=1e-5, atol=0))

    # Alpha functions: atol covers elements where alpha ≈ 0 at the periodic solution
    @test all(isapprox.(tw.alpha1[1:N], ref[:,3]; rtol=1e-5, atol=1e-8))
    @test all(isapprox.(tw.alpha2[1:N], ref[:,4]; rtol=1e-5, atol=1e-8))

    # Ring closure: END OF BEAMLINE entry equals the start of the ring.
    # Alpha uses atol since it is ≈0 at the periodic solution (pure relative tol is meaningless).
    @test tw.beta1[end]  ≈ tw.beta1[1]
    @test tw.beta2[end]  ≈ tw.beta2[1]
    @test isapprox(tw.alpha1[end], tw.alpha1[1]; atol=1e-8)
    @test isapprox(tw.alpha2[end], tw.alpha2[1]; atol=1e-8)
  end

  @testset "ESR Coupling Matrix vs Tao/Bmad" begin
      # Generated with: show lat -pipe -beginning * -at cmat_11@e24.16 -at cmat_12@e24.16 -at cmat_21@e24.16 -at cmat_22@e24.16
    ref = readdlm(
        joinpath(@__DIR__, "reference", "esr-coupling-tao.csv"),
        ',', Float64; skipstart=1
    )
    # Columns: cmat_11 (→c11), cmat_12 (→c12), cmat_21 (→c21), cmat_22 (→c22)

    ring = include(joinpath(@__DIR__, "lattices", "esr.jl"))
    tw = twiss(ring)

    N = size(ref, 1)

    # atol covers elements where cmat ≈ 0 (ring has very weak coupling overall)
    @test all(isapprox.(tw.c11[1:N], ref[:,1]; rtol=1e-5, atol=1e-8))
    @test all(isapprox.(tw.c12[1:N], ref[:,2]; rtol=1e-5, atol=1e-8))
    @test all(isapprox.(tw.c21[1:N], ref[:,3]; rtol=1e-5, atol=1e-8))
    @test all(isapprox.(tw.c22[1:N], ref[:,4]; rtol=1e-5, atol=1e-8))
  end

  @testset "Scalar chromatic derivatives match as_taylor_series" begin
    qf = Multipole(Kn1L=0.8); qd = Multipole(Kn1L=-0.7); d = Drift(L=1.0)
    bl = Beamline([qf, d, qd, d]; species_ref=Species("electron"), pc_ref=1e9)
    cols = ["dbeta1_2", "dbeta1_3"]
    ts = twiss(bl; rf_on=false, chrom=4, cols=cols)
    tt = twiss(bl; rf_on=false, chrom=4, cols=cols, as_taylor_series=true)
    @test ts.dbeta1_2 ≈ scalar.(tt.dbeta1_2)
    @test ts.dbeta1_3 ≈ scalar.(tt.dbeta1_3)
  end

  @testset "Twiss at every step of sliced elements" begin
      # twiss stores the maps at the tracking callbacks, which are called at every integrator step 
      # of every slice (n_slices*n_steps per element, also for Exact elements with steps). An element 
      # with n_slices = 2 tracks bitwise like two elements of half the length with the fringes at the 
      # outer ends, so the Twiss rows must be equal
    cell(q...) = Beamline([q..., Sextupole(L=0.2, Kn2=2.0, n_steps=2), Drift(L=1.2, tracking_method=Exact(), n_steps=3),
        Quadrupole(L=0.5, Kn1=-0.36), Drift(L=1.2)]; species_ref=Species("electron"), pc_ref=1e9)
    t1 = twiss(cell(Quadrupole(L=0.5, Kn1=0.36, n_slices=2, n_steps=2)))
    t2 = twiss(cell(Quadrupole(L=0.25, Kn1=0.36, n_steps=2, fringe_at=Fringe.EntranceEnd),
        Quadrupole(L=0.25, Kn1=0.36, n_steps=2, fringe_at=Fringe.ExitEnd)))
    @test length(t1.s) == 12
    for c in (:s, :beta1, :beta2, :alpha1, :alpha2, :phi1, :phi2, :dx)
      @test getproperty(t1, c) == getproperty(t2, c)
    end
  end

  @testset "Phase trombone" begin
      # A phase trombone changes the tunes by phi1 and phi2 (in units of 2π, also when set later
      # through transport_map_params) and leaves the Twiss parameters unchanged
    tr = Marker()
    fodo = Beamline([Quadrupole(Kn1=0.36, L=0.5), Drift(L=1.2), Quadrupole(Kn1=-0.36, L=0.5), Drift(L=1.2), tr];
        species_ref=Species("electron"), pc_ref=18e9)
    t0 = twiss(fodo)
    t = fodo[tr][1]
    phase_trombone!(t; phi1=0.1, phi2=-0.05)
    @test t.transport_map_params[1:3] == (0.1, -0.05, 0)
    t1 = twiss(fodo)
    @test t1.q1 ≈ t0.q1 + 0.1
    @test t1.q2 ≈ t0.q2 - 0.05
    for c in (:beta1, :beta2, :alpha1, :alpha2, :dx)
      @test all(isapprox.(getproperty(t1, c), getproperty(t0, c); rtol=1e-10, atol=1e-12))
    end
    t.transport_map_params = Base.setindex(t.transport_map_params, 0.2, 1)
    t2 = twiss(fodo)
    @test t2.q1 ≈ t0.q1 + 0.2
    @test t2.q2 ≈ t0.q2 - 0.05
  end

  @testset "Momentum compaction and slip factor" begin
      # At low energy they differ by 1/γ²: Bmad gives Alpha_p = 2.54269E-02, Eta_p = 2.28225E-02 for
      # this ring (coasting). The δ dependence (to δ³) agrees with the derivatives of a polynomial fit
      # of one-turn tracking on the δ-dependent closed orbits: ΔT/T = -Δz/(βcT) and 
      # ΔL/L = -Δz/L + β/β0 - 1 (the differences are the accuracy of the fit, which changes with its
      # range and degree)
    e = Species("electron")
    ring = Beamline([Quadrupole(Kn1=0.36, L=0.5), SBend(L=1.2, angle=pi/132), Drift(L=0.1),
        Quadrupole(Kn1=-0.36, L=0.5), SBend(L=1.2, angle=pi/132), Drift(L=0.1)]; species_ref=e, pc_ref=10e6)
    t1 = twiss(ring)
    @test t1.alphac ≈ 2.54269e-2 rtol=1e-5
    @test t1.etac ≈ 2.28225e-2 rtol=1e-5
    tilde_m = massof(e)/10e6
    @test t1.alphac - t1.etac ≈ tilde_m^2/(1 + tilde_m^2) rtol=1e-8
    t4 = twiss(ring; order=4, chrom=4)
    @test t4.alphac[delta=0] ≈ t1.alphac rtol=1e-12
    @test t4.etac[delta=0] ≈ t1.etac rtol=1e-12
    L = 3.6
    beta(δ) = (1 + δ)/sqrt((1 + δ)^2 + tilde_m^2)
    function dz(δ)
      co = find_closed_orbit(ring; v0=[0.0 0 0 0 0 δ]).v0
      return track(ring; v0=copy(co)).v[1,5,end] - co[1,5]
    end
    rel_dt(δ) = -dz(δ)/beta(δ)*beta(0.0)/L
    rel_dl(δ) = -dz(δ)/L + beta(δ)/beta(0.0) - 1
    h, deg = 2e-2, 14
    δs = range(-h, h, length=25)
    V = [(δ/h)^k for δ in δs, k in 0:deg]
    for (f, tps) in ((rel_dt, t4.etac), (rel_dl, t4.alphac))
      c = (V \ f.(δs)) ./ h .^ (0:deg) # c[k+1] is the coefficient of δ^k of the fit
      for (k, rtol) in zip(0:3, (1e-9, 1e-7, 1e-5, 1e-3))
        @test tps[delta=k] ≈ (k + 1)*c[k+2] rtol=rtol
      end
    end
  end

  @testset "Spin normal form with n0 exactly vertical" begin
      # In a flat ring without anything tilting the spin, the closed-orbit spin axis n0 is 
      # exactly -yhat, where the rotation of n0 to yhat in NonlinearNormalForm's normal had a 
      # 0/0 axis (NaN). The result must be the limit of a ring with n0 tilted by a tiny rotation 
      # (the differences scale linearly with the tilt)
    cell() = [Quadrupole(L=0.5, Kn1=0.36), Sextupole(L=0.2, Kn2=3.0), Drift(L=1.2), Quadrupole(L=0.5, Kn1=-0.33),
        Sextupole(L=0.2, Kn2=-3.0), Drift(L=1.0), SBend(L=1.0, angle=0.05)]
    run(bl) = twiss(bl; spin=true, order=2, cols=["nx", "ny", "nz"], as_taylor_series=true, at=[1])
    flat = run(Beamline(cell(); species_ref=Species("electron"), pc_ref=1e9))
    tilted = run(Beamline([cell()..., Patch(dz_rot=1e-10)]; species_ref=Species("electron"), pc_ref=1e9))
    coefs(t) = [t[i] for i in 0:GTPSA.numcoefs(t)-1]
    @test all(isfinite, coefs(flat.qspin.taylor_series))
    @test maximum(abs.(coefs(flat.qspin.taylor_series) .- coefs(tilted.qspin.taylor_series))) < 1e-8
    for c in (:nx, :ny, :nz)
      @test maximum(abs.(coefs(getproperty(flat, c)[1]) .- coefs(getproperty(tilted, c)[1]))) < 1e-5
    end
  end

  @testset "Regressions" begin
    e = Species("electron")
    cell(L=1.2) = Beamline([Quadrupole(L=0.5, Kn1=0.36, n_steps=2), Drift(L=L, n_steps=3), Quadrupole(L=0.5, Kn1=-0.33),
        Drift(L=1.0), SBend(L=1.0, angle=0.05), Multipole(Kn0L=1e-4)]; species_ref=e, pc_ref=1e9, context=Context(Ld=1.2))
    # A DefExpr element length is evaluated with the Beamline's context, as in tracking
    cols = ["beta1", "alpha1", "phi1", "dx"]
    t1 = twiss(cell(DefExpr(c -> c.Ld)); cols=cols)
    t2 = twiss(cell(); cols=cols)
    @test all(getproperty(t1, c) == getproperty(t2, c) for c in (:s, :beta1, :alpha1, :phi1, :dx))
    # Selecting the last element gives only its entrance row, at the same s as in the full table
    n = length(cell().line)
    t3 = twiss(cell(); at=[n], cols=["beta1"])
    @test t3.index == [n] && t3.s[1] == t2.s[findfirst(==(n), t2.index)]
    # twiss does not mutate the user's initial guess
    g = [1e-3 0 0 0 0 0.0]; g0 = copy(g)
    twiss(cell(); v0=g, at=[])
    @test g == g0
    # track accepts a weight vector
    @test track(cell(); v0=zeros(3, 6), weight=ones(3)) isa TrackingResult
    # getterm with all orders 0 keeps the constant term
    @elements begin
      qf = Quadrupole(Kn1=DefExpr(c -> c.kqf), L=0.5); dd = Drift(L=1.2); qd = Quadrupole(Kn1=-0.36, L=0.5)
    end
    p = Beamline([qf, dd, qd, dd]; species_ref=e, pc_ref=1e9)
    dnf = Descriptor([1, 1, 1, 1, 1, 1], 2, [1], 1)
    p.context.kqf = 0.36 + params(dnf)[1]
    tw = twiss(p; GTPSA_descriptor=dnf, at=[])
    @test scalar(getterm(tw.q1, as_taylor_series=true)) == scalar(tw.q1.taylor_series)
    # Parametric element length: derivative of the tune and beta functions vs finite differences
    mk(Ld) = Beamline([Quadrupole(Kn1=0.36, L=0.5), Drift(L=Ld, n_steps=2), Quadrupole(Kn1=-0.36, L=0.5),
        Drift(L=1.2), SBend(L=1.0, angle=0.05)]; species_ref=e, pc_ref=1e9)
    twp = twiss(mk(1.2 + params(dnf)[1]); GTPSA_descriptor=dnf, cols=["beta1"])
    h = 1e-5
    tp = twiss(mk(1.2 + h); cols=["beta1"]); tm = twiss(mk(1.2 - h); cols=["beta1"])
    @test grad(twp.q1)[1] ≈ (tp.q1 - tm.q1)/(2h) rtol=1e-9
    @test [grad(b)[1] for b in twp.beta1] ≈ (tp.beta1 .- tm.beta1)./(2h) rtol=1e-8
    @test scalar.(twp.s) == twiss(mk(1.2); cols=["beta1"]).s
  end

  @testset "ESR Chromatic Functions vs Tao/Bmad" begin
      # Generated with: show lat -pipe -beginning * -at dbeta_dpz_a@e24.16 -at dbeta_dpz_b@e24.16 -at dalpha_dpz_a@e24.16 -at dalpha_dpz_b@e24.16
      # TODO: Tao's dbeta_dpz_a shows a non-physical jump between elements 1 and 2
      # (both at s=0) and a ~3x discrepancy vs SciBmad at most elements.
      # SciBmad gives smooth, continuous values. Run test/investigate_chromatic.jl
      # to compare side-by-side.
    ref = readdlm(
        joinpath(@__DIR__, "reference", "esr-chromatic-tao.csv"),
        ',', Float64; skipstart=1
    )
    # Columns: dbeta_dpz_a (→∂β₁/∂pz), dbeta_dpz_b, dalpha_dpz_a, dalpha_dpz_b

    ring = include(joinpath(@__DIR__, "lattices", "esr.jl"))

    # Descriptor(6,2): 6 phase-space variables, order 2 → enables chromatic derivatives
    tw = twiss(ring; chrom=2, cols=["dbeta1", "dbeta2", "dalpha1", "dalpha2"])

    N = size(ref, 1)
    # Chromatic functions range up to ~5000 m; atol covers near-zero cases
    @test_broken all(isapprox.(tw.dbeta1,  ref[:,1]; rtol=1e-4, atol=1e-5))
    @test_broken all(isapprox.(tw.dbeta2,  ref[:,2]; rtol=1e-4, atol=1e-5))
    @test_broken all(isapprox.(tw.dalpha1, ref[:,3]; rtol=1e-4, atol=1e-5))
    @test_broken all(isapprox.(tw.dalpha2, ref[:,4]; rtol=1e-4, atol=1e-5))
  end

  @testset "ESR Closed Orbit (cv01_7 kicker) vs Tao/Bmad" begin
      # Tao reference: set element cv01_7 kick = 1e-5
      # then: show lat -pipe 1 -at orbit_x@e24.16 -at orbit_px@e24.16 -at orbit_y@e24.16 ...
      # cv01_7 is VKicker in Bmad (Kicker in SciBmad); Ks0L ↔ kick for vertical kick
    ring = include(joinpath(@__DIR__, "lattices", "esr.jl"))
    cv01_7.Ks0L = 1e-5

    co = find_closed_orbit(ring)

    # rtol=1e-3: SciBmad Kicker and Bmad VKicker have slightly different element models.
    # atol=1e-9: x, px are tiny (coupling response to vertical kick); absolute diff ~7e-10.
    @test isapprox(co.v0[1,1], 1.786150922152766e-7;  rtol=1e-3, atol=1e-9)
    @test isapprox(co.v0[1,2], -4.334497550054360e-7; rtol=1e-3, atol=1e-9)
    @test isapprox(co.v0[1,3], 2.677821948409130e-6;  rtol=1e-3, atol=1e-9)
    @test isapprox(co.v0[1,4], 2.649765658093324e-4;  rtol=1e-3, atol=1e-9)
    @test isapprox(co.v0[1,5], 0.0; atol=1e-12)
    @test isapprox(co.v0[1,6], 0.0; atol=1e-12)
  end
end
