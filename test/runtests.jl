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
        cb = (i, coords, pt) -> begin
          cnt[] += 1
          vs = coords.v[i,:]; qs = coords.q[i,:]
          transforms_out!(i, coords, pt)
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
    # steps of the rows: step of the location, the end of the beamline n_steps. The first slot
    # holds the initial particles at every location, then every m-th turn
    function matches(r, ref, step0, steps; m=1)
      for p in 1:size(v0, 1), (l, step) in enumerate(steps)
        isequal((r.v[p,:,l,1], r.q[p,:,l,1], r.state[p,l,1]),
                (v0[p,:], [1.0, 0, 0, 0], BeamTracking.STATE_ALIVE)) || return false
      end
      for p in 1:size(v0, 1), (l, step) in enumerate(steps), t in m:m:n_turns
        j = div(t, m) + 1
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
      @test plain.v[:,:,1,1] == v0 && plain.v[:,:,1,end] == plain.bunch.v && plain.index == [-1]
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

  @testset "Expressions: value_and_jacobian vs finite differences" begin
      # Values of expressions of twiss quantities equal those of ordinary twiss calls, and their
      # derivatives in knobs (element parameters, a context variable, a length) agree with central
      # differences: 6D and coasting, a kicked closed orbit, chromatic and amplitude-dependent
      # terms, deriv (orbit response and its derivative), lazy functions, a chained open lattice
    function fdjac(exprs, knobs; h=1e-6) # central differences of value(exprs) in the knobs
      x0 = [SciBmad._knob_value(k) for k in knobs]
      J = zeros(length(exprs), length(knobs))
      for j in eachindex(knobs)
        x = copy(x0)
        x[j] += h
        setknobs!(knobs, x)
        vp = vec(value(exprs))
        x[j] -= 2h
        setknobs!(knobs, x)
        J[:, j] = (vp .- vec(value(exprs))) ./ 2h
      end
      setknobs!(knobs, x0)
      return J
    end
      # (the differences have noise of ~1e-7)
    close_to(J, Jfd) = all(abs.(J .- Jfd) .<= 2e-6 .* (abs.(Jfd) .+ 1e-3))
    @elements begin
      qf = Quadrupole(Kn1=0.36, L=0.5)
      sf = Sextupole(Kn2=1.2, L=0.2)
      d = Drift(L=0.1)
      b = SBend(L=1.2, angle=pi/132, n_steps=4)
      qd = Quadrupole(Kn1=-0.36, L=0.5)
      sd = Sextupole(Kn2=-1.2, L=0.2)
      hk = HKicker(Kn0L=2e-4)
      rf = RFCavity(L=1e-2, voltage=1e7, rf_frequency=1e9, zero_phase=PhaseRef.AboveTransition)
    end
    ring = Beamline([qf, sf, b, d, qd, sd, b, d, hk, rf]; species_ref=Species("electron"),
                    E_ref=18e9)
    knobs = [@knob(qf.Kn1), Knob(qd, :Kn1), Knob(hk => :Kn0L), @knob(d.L)]
    for rf_on in (true, false)
      tw = TwissExpr(ring; rf_on)
      e = [tw.q1, tw.q2, tw.beta1[qd], tw.beta2[end], tw.x[5], tw.dx[b, :exit]...,
           mean(tw.beta1[3, :]), maximum(tw.beta2[:]), tw.phi1[7] - tw.phi1[2],
           sqrt(sum(abs2(tw.x["q*"]))), tw.s[end], tw.beta1[1]^2 - 5*mean(tw.alpha1[3, :])]
      push!(e, rf_on ? tw.q3 : tw.etac)
      t = twiss(ring; rf_on)
      r3 = [findall(==(3), t.index); findfirst(==(4), t.index)] # the 4 steps of b and its end
      @test length(r3) == 5
      v = value(e)
      @test v[1:4] ≈ [t.q1, t.q2, t.beta1[findfirst(==(5), t.index)], t.beta2[end]] rtol=1e-12
      @test v[8] ≈ sum(t.beta1[r3])/5 rtol=1e-12
      @test v[9] ≈ maximum(t.beta2) rtol=1e-12
      @test v[13] ≈ t.beta1[1]^2 - 5*sum(t.alpha1[r3])/5 rtol=1e-12
      v2, J = value_and_jacobian(e, knobs)
      @test v2 ≈ v rtol=1e-11
      @test close_to(J, fdjac(e, knobs))
      @test !close_to(J .* 1.0001, fdjac(e, knobs)) # (the comparison resolves 1e-4)
        # The lattice is restored
      @test qf.BMultipoleParams isa BMultipoleParams{Float64} && d.L === 0.1
    end
      # A single expression gives its gradient. With the kicked closed orbit, twiss with the
      # variables at order 1 agrees (it did not before the fixed point was refined, see "Chromatic
      # and parametric functions with a parameter dependent closed orbit")
    tw = TwissExpr(ring)
    D1 = Descriptor([1, 1, 1, 1, 1, 1], 2, [1], 1)
    qf.Kn1 = 0.36 + params(D1)[1]
    t1 = twiss(ring; GTPSA_descriptor=D1, at=[])
    qf.Kn1 = 0.36
    scalarize!(qf)
    v, g = value_and_jacobian(tw.q1, [@knob(qf.Kn1)])
    gfd = fdjac([tw.q1], [@knob(qf.Kn1)])[1]
    @test g[1] ≈ gfd rtol=1e-7
    @test grad(t1.q1)[1] ≈ gfd rtol=1e-7

      # Chromatic and amplitude-dependent terms (the orders are chosen from the quantities)
    tc = TwissExpr(ring; rf_on=false)
    e = [tc.dq1, tc.dq2, tc.q1[delta=2], tc.dbeta1[qd], tc.dx_2[1], tc.q1[J1=1]]
    ks = [@knob(sf.Kn2), @knob(sd.Kn2), @knob(qf.Kn1)]
    v, J = value_and_jacobian(e, ks)
    t3 = twiss(ring; rf_on=false, order=3, cols=["dbeta1", "dx_2"], at=[1, 5])
    @test v ≈ [t3.q1[delta=1], t3.q2[delta=1], t3.q1[delta=2], t3.dbeta1[2], t3.dx_2[1],
               t3.q1[J1=1]] rtol=1e-9
    @test close_to(J, fdjac(e, ks; h=1e-5))
    @test v[1] ≈ twiss(ring; rf_on=false, chrom=2, at=[]).q1[delta=1] rtol=1e-9
      # h terms are complex
    th = twiss(ring; rf_on=false, order=2, cols=["h3000", "h2100"], at=[1, 5])
    @test value([abs(tc.h3000[1]), real(tc.h2100[5]), imag(tc.h2100[5]), abs2(tc.h3000[1])]) ≈
          [abs(th.h3000[1]), real(th.h2100[2]), imag(th.h2100[2]), abs2(th.h3000[1])] rtol=1e-10
    @test_throws ErrorException value(tc.h3000[1])

      # deriv: the orbit response to the kicker, and its derivative in a quadrupole (second order)
    theta = Knob(hk, :Kn0L)
    R = [deriv(tw.x[i], theta) for i in (1, 5)]
    @test value(R) ≈ vec(fdjac([tw.x[1], tw.x[5]], [theta]; h=1e-7)) rtol=1e-6
    v, J = value_and_jacobian(R, [@knob(qf.Kn1), theta])
    @test v ≈ value(R) rtol=1e-11
    @test isapprox(J, fdjac(R, [@knob(qf.Kn1), theta]; h=1e-5); rtol=1e-5)

      # A context variable behind a deferred expression, lazy functions, a knob in an expression
    qf.Kn1 = DefExpr(c -> c.k1)
    ring.context.k1 = 0.36
    kc = @knob ring.context.k1
    e = [tw.beta1[5]*kc, lazy(() -> ring[1].Bn1),
         lazy((x, y) -> atan(x, y), tw.alpha1[5], tw.beta1[5]), abs(tw.alpha1[5]),
         max(tw.beta1[5], tw.beta2[5]), sqrt(tw.beta1[5])/kc^2]
    v, J = value_and_jacobian(e, [kc])
    @test v[2] == ring[1].Bn1
    @test close_to(J, fdjac(e, [kc]))
    @test_throws ErrorException value_and_jacobian(tw.q1, [@knob(qf.Kn1)]) # a deferred expression
    @test_throws ErrorException value_and_jacobian(tw.q1, [kc, Knob(ring.context, :k1)]) # twice
    @test ring.context.k1 === 0.36
    qf.Kn1 = 0.36

      # An open line that starts with the periodic functions of a cell, and with a given map
    cell = Beamline([qf, d, b, qd, d, b]; species_ref=Species("electron"), pc_ref=18e9)
    q3 = Quadrupole(Kn1=0.2, L=0.5)
    line = Beamline([q3, d, qd, d]; species_ref=Species("electron"), pc_ref=18e9)
    tcell = TwissExpr(cell)
    tl = TwissExpr(line; a_initial=tcell.a[1])
    e = [tl.beta1[end], tl.alpha1[end], tl.beta2[3], tl.dx[end], tl.phi1[end], tcell.q1]
    ks = [@knob(qf.Kn1), @knob(qd.Kn1), @knob(q3.Kn1)]
    v, J = value_and_jacobian(e, ks)
    a = twiss(cell, cols=["a"], as_taylor_series=true, at=[1]).a[1]
    t = twiss(line, a_initial=a)
    @test v[1:5] ≈ [t.beta1[end], t.alpha1[end], t.beta2[3], t.dx[end], t.phi1[end]] rtol=1e-11
    @test close_to(J, fdjac(e, ks))
    tf = TwissExpr(line; a_initial=a)
    e = [tf.beta1[end], tf.alpha2[end]]
    v, J = value_and_jacobian(e, [@knob(q3.Kn1), @knob(qd.Kn1)])
    @test v ≈ [t.beta1[end], t.alpha2[end]] rtol=1e-11
    @test close_to(J, fdjac(e, [@knob(q3.Kn1), @knob(qd.Kn1)]))

      # Tunes and phases keep their integer parts whatever rows are used (6 cells: twiss with few
      # rows gives them modulo 1)
    qf6 = Quadrupole(Kn1=1.2, L=0.5)
    qd6 = Quadrupole(Kn1=-1.1, L=0.5)
    cells = Beamline(repeat([qf6, d, b, qd6, d, b], 6); species_ref=Species("electron"),
                     pc_ref=18e9)
    t = twiss(cells)
    t6 = TwissExpr(cells)
    e = [t6.q1, t6.q2, t6.phi1[30] - t6.phi1[3]]
    ref = [t.q1, t.q2, t.phi1[findfirst(==(30), t.index)] - t.phi1[findfirst(==(3), t.index)]]
    @test ref[1] > 1 && twiss(cells; at=[]).q1 < 0.5
    @test value(e) ≈ ref rtol=1e-12
    v, J = value_and_jacobian(e, [@knob(qf6.Kn1), @knob(qd6.Kn1)])
    @test v ≈ ref rtol=1e-11
    @test close_to(J, fdjac(e, [@knob(qf6.Kn1), @knob(qd6.Kn1)]))
    res = match!(vary=[@knob(qf6.Kn1), @knob(qd6.Kn1)], targets=[t6.q1 == 1.6, t6.q2 == 1.3])
    t = twiss(cells)
    @test res.converged
    @test t.q1 ≈ 1.6 atol=1e-8
    @test t.q2 ≈ 1.3 atol=1e-8
  end

  @testset "match!" begin
    @elements begin
      qf = Quadrupole(Kn1=0.36, L=0.5)
      sf = Sextupole(Kn2=1.2, L=0.2)
      d = Drift(L=0.1)
      b = SBend(L=1.2, angle=pi/132)
      qd = Quadrupole(Kn1=-0.36, L=0.5)
      sd = Sextupole(Kn2=-1.2, L=0.2)
    end
    cell = Beamline([qf, sf, b, d, qd, sd, b, d]; species_ref=Species("electron"), pc_ref=18e9)
    tw = TwissExpr(cell)
    kf, kd = @knob(qf.Kn1), @knob(qd.Kn1)
      # Tunes, checked with an ordinary twiss; undo!
    res = match!(vary=[kf, kd], targets=[tw.q1 == 0.2, tw.q2 == 0.15])
    @test res.converged && res.status == "all targets met"
    t = twiss(cell; at=[])
    @test t.q1 ≈ 0.2 atol=1e-8
    @test t.q2 ≈ 0.15 atol=1e-8
    @test res.x == [qf.Kn1, qd.Kn1]
    @test occursin("Match converged", sprint(show, res))
    x_tunes = res.x
    undo!(res)
    @test [qf.Kn1, qd.Kn1] == [0.36, -0.36]
      # The same given the Beamline and a function of its TwissExpr, and as a do block
    res = match!(cell; vary=[qf => :Kn1, qd => :Kn1], targets = tw -> [tw.q1 == 0.2, tw.q2 == 0.15])
    @test res.converged && res.x ≈ x_tunes
    undo!(res)
    res = match!(cell; vary=[kf, kd]) do tw
      [tw.q1 == 0.2, tw.q2 == 0.15]
    end
    @test res.converged && res.x ≈ x_tunes
    undo!(res)
      # Chromaticities (coasting), checked with twiss at order 2
    tc = TwissExpr(cell; rf_on=false)
    res = match!(vary=[@knob(sf.Kn2), @knob(sd.Kn2)], targets=[tc.dq1 == 1, tc.dq2 == 1])
    t = twiss(cell; order=2, at=[])
    @test res.converged
    @test t.q1[delta=1] ≈ 1 atol=1e-8
    @test t.q2[delta=1] ≈ 1 atol=1e-8
    undo!(res)
      # A bound stops the match with the knob on it
    res = match!(vary=[Knob(qf, :Kn1; upper=0.5), kd], targets=[tw.q1 == 0.2, tw.q2 == 0.15])
    @test !res.converged && qf.Kn1 == 0.5 && x_tunes[1] > 0.5
    undo!(res)
      # An inequality that holds moves nothing; one that does not is brought to its limit
    res = match!(vary=[kf, kd], targets=[tw.beta1[1] <= 20])
    @test res.converged && res.iterations == 0
    res = match!(vary=[kf, kd], targets=[tw.q1 == 0.2, maximum(tw.beta1[:]) <= 6])
    t = twiss(cell)
    @test res.converged
    @test t.q1 ≈ 0.2 atol=1e-8
    @test maximum(t.beta1) ≈ 6 atol=1e-7
    undo!(res)
      # More knobs than targets, and more targets than knobs (least squares: the gradient of the
      # squared residuals vanishes there, and not at the start)
    res = match!(vary=[kf, kd, @knob(d.L)], targets=[tw.q1 == 0.2, tw.q2 == 0.15])
    @test res.converged && d.L != 0.1
    undo!(res)
    res = match!(vary=[kf], targets=[tw.q1 == 0.2, tw.q2 == 0.15])
    @test !res.converged && res.status == "no further improvement possible"
    r, J = value_and_jacobian(res.problem, res.x)
    r0, J0 = value_and_jacobian(res.problem, res.x0)
    @test sqrt(sum(abs2, J'*r)) < 1e-5*sqrt(sum(abs2, J)*sum(abs2, r))
    @test sqrt(sum(abs2, J0'*r0)) > 0.1*sqrt(sum(abs2, J0)*sum(abs2, r0))
    undo!(res)
      # A step into an unstable lattice is shortened (the first step for 0.49 overshoots)
    res = match!(vary=[kf, kd], targets=[tw.q1 == 0.49, tw.q2 == 0.49])
    @test res.converged && any(diff(res.history) .> -0.9 .* res.history[1:end-1])
    undo!(res)
      # minimize
    res = match!(vary=[kf, kd], minimize=[tw.beta1[1] - 8, tw.beta2[1] - 12])
    t = twiss(cell; at=[1])
    @test t.beta1[1] ≈ 8 atol=1e-6
    @test t.beta2[1] ≈ 12 atol=1e-6
    undo!(res)
      # The functions for other optimizers
    prob = MatchProblem(vary=[kf, kd], targets=[tw.q1 == 0.2, tw.q2 == 0.15])
    x0 = prob.x
    @test x0 == [0.36, -0.36] && prob.lower == [-Inf, -Inf] && prob.upper == [Inf, Inf]
    @test penalty(prob, x0) ≈ sum(abs2, value(prob, x0))
    @test penalty(prob, [50.0, 50.0]) == Inf # unstable
    @test penalty(prob, x_tunes) < 1e-15
    setknobs!(prob, x0)
    @test [qf.Kn1, qd.Kn1] == x0
      # Vectors of targets
    @test (tw.beta1["q*"] <= 30) isa Vector{Target}
    @test length(tw.dx[[1, 5]] == [1.0, 2.0]) == 2
    @test_throws ErrorException match!(vary=[kf], targets=[[tw.q1, tw.q2] == [0.2, 0.1]])
    @test_throws ErrorException match!(vary=[kf, Knob(qf, :Kn1)], targets=[tw.q1 == 0.2])
  end

  @testset "Chromatic and parametric functions with a parameter dependent closed orbit" begin
      # With phase space variables below the maximum order (chrom above order, or GTPSA parameters
      # with the variables at order 1), the one-turn map around the closed orbit lacks the terms
      # (x^2, ...) that the dependence of the fixed point on δ or on a parameter turns into terms of
      # lower order, so the fixed point is refined by tracking it. A coupled cell with sextupoles
      # and dispersion at its start: chrom=n equals order=n, the chromaticities equal differences
      # of the mode tunes of tracked one-turn matrices around the closed orbits at ±δ, and the
      # derivative of a tune in a quadrupole strength with a kicked orbit equals a difference
    qf = Quadrupole(Kn1=0.36, L=0.5)
    hk = HKicker(Kn0L=2e-4)
    bl = Beamline([qf, Sextupole(Kn2=1.2, L=0.2), SBend(L=1.2, angle=pi/132), Drift(L=0.1),
                   Quadrupole(Kn1=-0.36, L=0.5, tilt=0.02), Sextupole(Kn2=-1.2, L=0.2),
                   SBend(L=1.2, angle=pi/132), Drift(L=0.1), hk];
                  species_ref=Species("electron"), pc_ref=18e9)
    cols = ["dbeta1", "dbeta2", "dalpha1", "dc12", "dx_2", "w1", "dx"]
    tc = twiss(bl; chrom=2, cols=cols)
    to = twiss(bl; order=2, cols=cols)
    @test abs(tc.dx[1]) > 1 # dispersion at the start
    for c in cols
      @test getproperty(tc, Symbol(c)) ≈ getproperty(to, Symbol(c)) rtol=1e-9
    end
    @test [tc.q1[delta=1], tc.q2[delta=1]] ≈ [to.q1[delta=1], to.q2[delta=1]] rtol=1e-10
    t3 = twiss(bl; chrom=3, cols=["dbeta1_2", "dx_3"], at=[1, 5])
    o3 = twiss(bl; order=3, cols=["dbeta1_2", "dx_3"], at=[1, 5])
    @test t3.dbeta1_2 ≈ o3.dbeta1_2 rtol=1e-9
    @test t3.dx_3 ≈ o3.dx_3 rtol=1e-9
    @test [t3.q1[delta=2], t3.q2[delta=2]] ≈ [o3.q1[delta=2], o3.q2[delta=2]] rtol=1e-9
    function mode_tunes(delta) # from the one-turn matrix around the closed orbit, by differences
      co = find_closed_orbit(bl; v0=[0. 0. 0. 0. 0. delta]).v0
      M = zeros(4, 4)
      for j in 1:4
        vp = copy(co)
        vp[1,j] += 1e-6
        vm = copy(co)
        vm[1,j] -= 1e-6
        M[:,j] = (track(bl; v0=vp).v[1,1:4,1,end] .- track(bl; v0=vm).v[1,1:4,1,end]) ./ 2e-6
      end
      return sort(filter(>(0), angle.(SciBmad.LinearAlgebra.eigvals(M)) ./ 2pi))
    end
    @test mode_tunes(0.0) ≈ sort([tc.q1[], tc.q2[]]) rtol=1e-8
    chroms = (mode_tunes(1e-4) .- mode_tunes(-1e-4)) ./ 2e-4
    @test chroms ≈ [tc.q2[delta=1], tc.q1[delta=1]] rtol=1e-5 # (q2 is the lower tune)
    @test abs(chroms[1] - chroms[2]) > 0.01
      # A parameter that moves the closed orbit, with the variables at order 1
    f() = (t = twiss(bl; at=[]); [t.q1, t.q2])
    qf.Kn1 = 0.36 + 1e-6
    fp = f()
    qf.Kn1 = 0.36 - 1e-6
    fm = f()
    D = Descriptor([1, 1, 1, 1, 1, 1], 2, [1], 1)
    qf.Kn1 = 0.36 + params(D)[1]
    tp = twiss(bl; GTPSA_descriptor=D, at=[])
    @test [grad(tp.q1)[1], grad(tp.q2)[1]] ≈ (fp .- fm) ./ 2e-6 rtol=1e-7
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
