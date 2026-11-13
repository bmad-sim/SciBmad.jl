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

  @testset "track at locations" begin
    # Reference: each particle tracked alone with track!, with a callback that counts the steps of
    # the turn and saves the particle in lab coordinates at every step (and restores it)
    qf = Quadrupole(L=0.5, Kn1=0.36, n_steps=4)
    qd = Quadrupole(L=0.5, Kn1=-0.36, x_offset=1e-4, y_rot=1e-4, tilt=0.1, n_steps=3,
        x1_limit=-1.5e-3, x2_limit=1.5e-3, y1_limit=-1.5e-3, y2_limit=1.5e-3)
    b = SBend(L=1.0, angle=0.1, n_slices=2, n_steps=2, x_rot=1e-4, y_offset=2e-4)
    rf = RFCavity(L=0.1, voltage=1e5, rf_frequency=4e8, zero_phase=PhaseRef.BelowTransition)
    bl = Beamline([qf, Drift(L=1.0), b, qd, Drift(L=1.0), rf]; species_ref=Species("electron"),
        E_ref=1e9)
    n_turns = 4
    v0 = [2e-4*sin(7.0*i + k) for i in 1:11, k in 1:6] # some particles are lost
    n_steps = sum(e -> e.n_slices*e.n_steps, bl.line)
    # A turn ends at the end of the beamline, the first one begins at the element start (its
    # beginning is step0 steps from the beginning of the beamline, the step of a location)
    function reference(start)
      ref = Dict{Tuple{Int,Int,Int},Any}() # (particle, turn, step) => (v, q, state)
      step0 = sum(e -> e.n_slices*e.n_steps, bl.line[1:start-1]; init=0)
      for p in 1:size(v0, 1)
        cnt = Ref(0); turn = Ref(1)
        cb = (i, coords, s, t_ref, step, bg, ds, g, out!, in!) -> begin
          cnt[] += 1
          vs = coords.v[i,:]; qs = coords.q[i,:]
          out!(i, coords)
          ref[(p, turn[], cnt[])] = (coords.v[i,:], coords.q[i,:], coords.state[i])
          coords.v[i,:] .= vs; coords.q[i,:] .= qs
        end
        bun = Bunch(v=v0[p:p,:], q=[1.0 0 0 0], callbacks=(cb,), species=bl.species_ref,
            p_over_q_ref=bl.p_over_q_ref)
        ref[(p, 1, step0)] = (bun.v[1,:], bun.q[1,:], bun.state[1]) # the start in the first turn
        cnt[] = step0
        for ele in bl.line[start:end] # element by element from the start
          BeamTracking.track!(bun, ele; notify=false)
        end
        for t in 2:n_turns
          ref[(p, t, 0)] = (bun.v[1,:], bun.q[1,:], bun.state[1]) # the beginning of the beamline
          cnt[] = 0; turn[] = t
          BeamTracking.track!(bun, bl; notify=false)
        end
      end
      return ref, step0
    end
    # steps of the rows: step of the location, the end of the beamline n_steps
    function matches(r, ref, step0, steps; m=1)
      for p in 1:size(v0, 1), (l, step) in enumerate(steps), t in 1:m:n_turns
        j = div(t-1, m) + 1
        if t == 1 && step < step0 # not reached yet
          all(isnan, r.v[p,:,l,j]) && r.state[p,l,j] == BeamTracking.STATE_PREBORN || return false
        else
          isequal((r.v[p,:,l,j], r.q[p,:,l,j], r.state[p,l,j]), ref[(p, t, step)]) || return false
        end
      end
      return true
    end
    all_steps = [0:n_steps-1; n_steps]
    key(res, k) = (res.index[k], res.s[k])
    for start in (1, 4)
      ref, step0 = reference(start)
      plain = track(bl; v0, n_turns, spin=true, start)
      @test plain.v0 == v0 && plain.v[:,:,1,end] == plain.bunch.v && plain.index == [-1]
      @test matches(plain, ref, step0, [n_steps])
      @test 0 < count(!=(BeamTracking.STATE_ALIVE), plain.bunch.state) < size(v0, 1)
      for kw in ((;), (; use_KA=true, use_explicit_SIMD=false), (; use_explicit_SIMD=false))
        r = track(bl; v0, n_turns, spin=true, at=:, start, kw...)
        @test size(r.v, 3) == n_steps + 1 && r.index[end] == -1
        @test matches(r, ref, step0, all_steps)
        # saving does not change the tracking
        @test isequal(r.bunch.v, plain.bunch.v) && isequal(r.bunch.q, plain.bunch.q)
        # Other locations are the same rows of at=:
        for at in ([qf, qd], [2, 5], [(0.2, 2.6)], [qd, (2.4, 2.6), 1], [(3.0, Inf)]), m in (1, 2)
          rs = track(bl; v0, n_turns, spin=true, at, start, save_every_n_turns=m, kw...)
          rows = [findfirst(k -> key(r, k) == key(rs, l), eachindex(r.s)) for l in eachindex(rs.s)]
          @test matches(rs, ref, step0, all_steps[rows]; m)
        end
      end
      # in_body_coordinates: the misaligned elements differ from lab coordinates
      rb = track(bl; v0, n_turns, spin=true, at=:, start, in_body_coordinates=true)
      @test !matches(rb, ref, step0, all_steps)
    end
    @test_throws ErrorException track(bl; v0, n_turns=0)
    # twiss: an s-range starting inside the first element keeps all of its steps in the range
    cell = Beamline([qf, Drift(L=1.0), Quadrupole(L=0.5, Kn1=-0.36), Drift(L=1.0)];
        species_ref=Species("electron"), E_ref=1e9)
    tw = twiss(cell; at=[(0.2, 0.4)], cols=["beta1"])
    tall = twiss(cell; cols=["beta1"])
    # (the one-turn map is composed from the maps between the rows, so it rounds differently)
    @test tw.s == [0.25, 0.375]
    @test tw.beta1 ≈ tall.beta1[3:4] rtol=1e-14
  end

  @testset "Open-lattice Twiss (a_initial)" begin
      # Starting from the periodic a reproduces the periodic Twiss (coasting and 6D, with and
      # without spin, orders 1 and 2). An unmatched start (a thin kick in x, y scaled) propagates
      # as Courant-Snyder through thin quadrupoles and drifts, and differs from the periodic Twiss
    ring = Beamline([Quadrupole(Kn1=0.37, L=0.5), Sextupole(Kn2=1.2, L=0.2),
        SBend(L=1.2, angle=pi/132), Drift(L=0.1), Quadrupole(Kn1=-0.37, L=0.5, Ks1=1e-3),
        Sextupole(Kn2=-1.2, L=0.2), SBend(L=1.2, angle=pi/132), Drift(L=0.1),
        RFCavity(L=1e-2, voltage=1e7, rf_frequency=1e9, zero_phase=PhaseRef.AboveTransition)];
        species_ref=Species("electron"), E_ref=18e9)
    for (rf_on, order, spin) in ((false, 1, false), (false, 1, true), (true, 1, false),
                                 (true, 1, true), (false, 2, false), (true, 2, true))
      tw = twiss(ring; rf_on, order, spin)
      a = twiss(ring; rf_on, order, spin, cols=["a"], as_taylor_series=true, at=[1]).a[1]
      to = twiss(ring; rf_on, a_initial=a)
      @test isodd(SciBmad.NonlinearNormalForm.nvars(a)) == !rf_on
      @test isnothing(a.q) == !spin
      for c in propertynames(tw)
        c == :qspin && continue # The spin tune is a one-turn quantity
        x = getproperty(tw, c)
        x isa AbstractVector{<:Real} || continue
        @test getproperty(to, c) ≈ x atol=1e-11
      end
    end

    qf = Multipole(Kn1L=0.8); qd = Multipole(Kn1L=-0.7); d = Drift(L=1.0)
    bl = Beamline([qf, d, qd, d]; species_ref=Species("electron"), pc_ref=1e9)
    tw = twiss(bl)
    a = twiss(bl; cols=["a"], as_taylor_series=true, at=[1]).a[1]
    K, r = 0.3, 1.5
    M = [1 0 0 0 0; -K 1 0 0 0; 0 0 r 0 0; 0 0 0 1/r 0; 0 0 0 0 1.0]
    m = DAMap(v_matrix=M, init=SciBmad.NonlinearNormalForm.getinit(a), nv=5, np=1)
    to = twiss(bl; a_initial=m ∘ a)
    quad = (k -> [1 0; -k 1.0], k -> [1 0; k 1.0]) # Thin quadrupole in x and y
    for (p, β0, α0, β, α, ϕ, βp) in (
        (1, tw.beta1[1], tw.alpha1[1] + K*tw.beta1[1], to.beta1, to.alpha1, to.phi1, tw.beta1),
        (2, r^2*tw.beta2[1], tw.alpha2[1], to.beta2, to.alpha2, to.phi2, tw.beta2))
      γ0 = (1 + α0^2)/β0
      T = [1 0; 0 1.0]
      for (i, m) in enumerate([quad[p](0.8), [1 1; 0 1.0], quad[p](-0.7), [1 1; 0 1.0]])
        T = m*T
        @test β[i+1] ≈ T[1,1]^2*β0 - 2T[1,1]*T[1,2]*α0 + T[1,2]^2*γ0 atol=1e-13
        @test α[i+1] ≈ -T[1,1]*T[2,1]*β0 + (T[1,1]*T[2,2] + T[1,2]*T[2,1])*α0 -
            T[1,2]*T[2,2]*γ0 atol=1e-13
        @test ϕ[i+1] ≈ atan(T[1,2], T[1,1]*β0 - T[1,2]*α0)/(2pi) atol=1e-14
      end
      @test abs(β[end] - βp[end]) > 1 # Unmatched: not the periodic Twiss
    end
  end

  @testset "Parametric normal form vs finite differences" begin
      # The examples of docs/src/parametric-nf.md: derivatives of the tunes, of beta1, of the 
      # chromaticities and of dh2000 in magnet strengths (GTPSA parameters, changing Descriptors 
      # with scalarize!) agree with central finite differences of ordinary twiss calls
    @elements begin
      qf = Quadrupole(Kn1=DefExpr(c -> c.kqf), L=0.5)
      sf = Sextupole(Kn2=DefExpr(c -> c.ksf), L=0.2)
      d = Drift(L=0.1)
      b = SBend(L=1.2, angle=pi/132)
      qd = Quadrupole(Kn1=DefExpr(c -> c.kqd), L=0.5)
      sd = Sextupole(Kn2=DefExpr(c -> c.ksd), L=0.2)
    end
    fodo = Beamline([qf, sf, b, d, qd, sd, b, d]; species_ref=Species("electron"), pc_ref=18e9)
    k0 = (kqf=0.36, kqd=-0.36, ksf=1.2, ksd=-1.2)
    for k in keys(k0)
      setproperty!(fodo.context, k, k0[k])
    end
    function fd(f, key, h) # central difference in one context value, the others at k0
      setproperty!(fodo.context, key, k0[key] + h)
      fp = f()
      setproperty!(fodo.context, key, k0[key] - h)
      fm = f()
      setproperty!(fodo.context, key, k0[key])
      return (fp .- fm) ./ (2h)
    end

    lin() = (tt = twiss(fodo); (tt.q1, tt.q2, tt.beta1[1]))
    fd_lin = [fd(lin, :kqf, 1e-6), fd(lin, :kqd, 1e-6)]
    chrom() = (tt = twiss(fodo; chrom=2); (tt.q1[delta=1], tt.q2[delta=1]))
    fd_chrom = [fd(chrom, :ksf, 1e-5), fd(chrom, :ksd, 1e-5)]
    fd_dh2000 = fd(() -> twiss(fodo; order=2, cols=["dh2000"]).dh2000[1], :ksf, 1e-5)

    dnf = Descriptor([1, 1, 1, 1, 1, 1], 2, [1, 1], 1)
    dk = params(dnf)
    fodo.context.kqf += dk[1]
    fodo.context.kqd += dk[2]
    tw = twiss(fodo; GTPSA_descriptor=dnf)
    for (i, v) in enumerate((tw.q1, tw.q2, tw.beta1[1]))
      @test grad(v) ≈ [fd_lin[1][i], fd_lin[2][i]] rtol=1e-6
    end

    scalarize!(fodo)
    dnf2 = Descriptor([1, 1, 1, 1, 1, 2], 3, [1, 1], 1)
    dk = params(dnf2)
    fodo.context.ksf += dk[1]
    fodo.context.ksd += dk[2]
    tw = twiss(fodo; GTPSA_descriptor=dnf2)
    for (i, q) in enumerate((tw.q1, tw.q2))
      chrom_q = getterm(q, delta=1, as_taylor_series=true)
      @test grad(chrom_q) ≈ [fd_chrom[1][i], fd_chrom[2][i]] rtol=1e-6
    end

    scalarize!(fodo)
    dnf3 = Descriptor([2, 2, 2, 2, 2, 2], 3, [1], 1)
    dk = params(dnf3)
    fodo.context.ksf += dk[1]
    tw = twiss(fodo; GTPSA_descriptor=dnf3, cols=["dh2000"])
    @test only(grad(tw.dh2000[1])) ≈ fd_dh2000 rtol=1e-6
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
      return track(ring; v0=copy(co)).v[1,5,1,end] - co[1,5]
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
    # With a parameter (the quadrupole strength): at order 1 the same values, at order 2 also 
    # their derivatives in it, as finite differences (in separate Beamlines: each Descriptor needs 
    # its own lattice)
    ring_k(K) = Beamline([Quadrupole(Kn1=K, L=0.5), SBend(L=1.2, angle=pi/132), Drift(L=0.1),
        Quadrupole(Kn1=-0.36, L=0.5), SBend(L=1.2, angle=pi/132), Drift(L=0.1)]; species_ref=e, pc_ref=10e6)
    D1 = Descriptor([1, 1, 1, 1, 1, 1], 1, [1], 1)
    tk1 = twiss(ring_k(0.36 + params(D1)[1]); GTPSA_descriptor=D1)
    @test tk1.etac[] ≈ t1.etac rtol=1e-12
    @test tk1.alphac[] ≈ t1.alphac rtol=1e-12
    D2 = Descriptor([1, 1, 1, 1, 1, 1], 2, [1], 1)
    tk2 = twiss(ring_k(0.36 + params(D2)[1]); GTPSA_descriptor=D2)
    hk = 1e-6
    tp, tm = twiss(ring_k(0.36 + hk)), twiss(ring_k(0.36 - hk))
    @test tk2.etac[] ≈ t1.etac rtol=1e-12
    @test tk2.alphac[] ≈ t1.alphac rtol=1e-12
    @test GTPSA.geti(tk2.etac.taylor_series, 7) ≈ (tp.etac - tm.etac)/(2hk) rtol=1e-7
    @test GTPSA.geti(tk2.alphac.taylor_series, 7) ≈ (tp.alphac - tm.alphac)/(2hk) rtol=1e-7
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
