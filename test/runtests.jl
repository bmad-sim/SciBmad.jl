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
        isequal((r.data.v[p,:,l,1], r.data.q[p,:,l,1], r.data.state[p,l,1]),
                (v0[p,:], [1.0, 0, 0, 0], BeamTracking.STATE_ALIVE)) || return false
      end
      for p in 1:size(v0, 1), (l, step) in enumerate(steps), t in m:m:n_turns
        j = div(t, m) + 1
        if t == 1 && step < step0 # not reached yet
          all(isnan, r.data.v[p,:,l,j]) && r.data.state[p,l,j] == BeamTracking.STATE_PREBORN ||
            return false
        else
          isequal((r.data.v[p,:,l,j], r.data.q[p,:,l,j], r.data.state[p,l,j]), ref[(p, t, step)]) ||
            return false
        end
      end
      return true
    end
    all_steps = [0:n_steps-1; n_steps]
    key(res, k) = (res.index[k], res.s[k])
    for start in (1, 4)
      ref, step0 = reference(start)
      plain = track(bl; v0, n_turns, spin=true, start)
      @test plain.data.v[:,:,1,1] == v0 && plain.data.v[:,:,1,end] == plain.bunch.v &&
            plain.index == [-1]
      @test matches(plain, ref, step0, [n_steps])
      @test 0 < count(!=(BeamTracking.STATE_ALIVE), plain.bunch.state) < size(v0, 1)
      for kw in ((;), (; use_KA=true, use_explicit_SIMD=false), (; use_explicit_SIMD=false))
        r = track(bl; v0, n_turns, spin=true, at=:, start, kw...)
        @test size(r.data.v, 3) == n_steps + 1 && r.index[end] == -1
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
    @test tw.df.s == [0.25, 0.375]
    @test tw.df.beta1 ≈ tall.df.beta1[3:4] rtol=1e-14
  end

  @testset "Tracking results read at locations" begin
      # res.x etc. are read at the locations of twiss (by default the end of the beamline), for a
      # turn (its number) and a particle, from the saved arrays res.data
    qf = Quadrupole(L=0.5, Kn1=0.36, n_steps=2)
    qd = Quadrupole(L=0.5, Kn1=-0.36)
    d = Drift(L=1.0)
    bl = Beamline([qf, d, qd, d]; species_ref=Species("electron"), pc_ref=1e9)
    v0 = [1e-3 0 0 0 0 0; 0 2e-4 1e-3 0 0 1e-4]
    res = track(bl; v0, n_turns=4, save_every_n_turns=2, spin=true, save_reference=true,
                at=[qd, (0.0, 0.6), (3.0, Inf)])
    D = res.data
    loc(i, s) = findfirst(l -> res.index[l] == i && res.s[l] == s, eachindex(res.s))
    lend, lqd, lqf1 = loc(-1, 3.0), loc(3, 1.5), loc(1, 0.25)
    @test res.turns == [0, 2, 4] && size(D.v, 4) == 3
    @test res.x[end] == D.v[:, 1, lend, :]
    @test_throws ErrorException res.x[] # several locations: one must be given
    @test res.pz[end, turn=2] == D.v[:, 6, lend, 2] &&
          res.pz[end, turn=4, particle=2] == D.v[2, 6, lend, 3]
    @test res.y[qd, turn=4] == res.y[3, turn=4] == D.v[:, 3, lqd, 3]
    @test res.x[qf, 1, particle=2] == D.v[2, 1, lqf1, :]
    @test res.px[[d], :exit, turn=2] == D.v[:, 2, [lqd, lend], 2] # d is placed twice
    @test res.y[[qd], turn=4] == D.v[:, 3, [lqd], 3]
    @test_throws ErrorException res.px[d, :exit, turn=2] # (placed twice: one must be chosen)
    @test res.x[qf, :, turn=4, particle=1] == D.v[1, 1, [loc(1, 0.0), lqf1, loc(2, 0.5)], 3]
    @test res.x[(0.0, 0.6), turn=0] == v0[:, 1] .* ones(1, 3)
    @test res.v[qd, particle=1, turn=2] == D.v[1, :, lqd, 2]
    @test res.q[end, turn=4] == D.q[:, :, lend, 3] && res.state[end, turn=4] == D.state[:, lend, 3]
    @test res.t_ref[qd, turn=2] == D.t_ref[lqd, 2] &&
          res.beta_gamma_ref[end] == D.beta_gamma_ref[lend, :]
    @test res.x[end, turn=:, particle=:] == res.x[end]
    @test res.x[end, turn=[4, 0], particle=2:2] == D.v[2:2, 1, lend, [3, 1]]
    @test res.y[qd, turn=0:2:4, particle=1] == D.v[1, 3, lqd, :]
    @test_throws ErrorException res.x[end, turn=3] # not saved
    @test_throws ErrorException res.x[end, turn=0:1:2]
    @test_throws ErrorException res.x[[d], turn=2] # the beginning of d is not saved
    plain = track(bl; v0)
    @test plain.x[turn=1] == plain.bunch.v[:, 1] && plain.x[turn=0] == v0[:, 1]
    @test_throws ErrorException plain.q[]
    @test_throws ErrorException plain.t_ref[]
    one = track(bl; v0, n_turns=2, at=[qd]) # the only saved location need not be given
    @test one.x[turn=1] == one.x[qd, turn=1] == one.data.v[:, 1, 1, 2]
    @test_throws ErrorException track(bl; v0, at=[]).x[]
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
      for c in names(tw.df) # (the summary values, e.g. the spin tune, are one-turn quantities)
        x = tw.df[!, c]
        x isa AbstractVector{<:Real} || continue
        @test to.df[!, c] ≈ x atol=1e-11
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
        (1, tw.beta1[1], tw.alpha1[1] + K*tw.beta1[1], to.df.beta1, to.df.alpha1, to.df.phi1,
         tw.df.beta1),
        (2, r^2*tw.beta2[1], tw.alpha2[1], to.df.beta2, to.df.alpha2, to.df.phi2, tw.df.beta2))
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

  @testset "Open-line Twiss from initial Twiss parameters (initial)" begin
    # Started with the periodic Twiss parameters of row 1 (a coupled ring with dispersion and a
    # kicked orbit, at δ = 1e-3), the open line reproduces the periodic Twiss (with the coupling
    # of the opposite sign or transposed, β1 is off by 1.6 and 0.12)
    @elements begin
      qf = Quadrupole(Kn1=0.36, L=0.5)
      qd = Quadrupole(Kn1=-0.36, L=0.5)
      sq = Quadrupole(Ks1=0.05, L=0.2)
      d = Drift(L=0.6)
      b = SBend(L=1.0, g_ref=0.05, angle=0.05)
      sx = Sextupole(Kn2=2.0, L=0.2)
      kick = Kicker(Kn0L=2e-4, Ks0L=-1e-4)
    end
    ring = Beamline([qf, d, b, sx, d, sq, qd, d, b, kick, d], E_ref=18e9,
                    species_ref=Species("electron"))
    cols = ["beta1", "alpha1", "beta2", "alpha2", "c11", "c12", "c21", "c22", "gammac", "dx",
            "dpx", "dy", "dpy", "x", "px", "y", "py", "pz", "phi1", "phi2"]
    tp = twiss(ring; cols, delta0=1e-3)
    r1 = tp.df[1, :]
    @test abs(r1.c11) > 0.5 && abs(r1.dy) > 0.1 && abs(r1.x) > 1e-3
    tw = twiss(ring; initial=r1, cols) # (a table row: its other columns are ignored)
    for c in cols
      @test tw.df[!, c] ≈ tp.df[!, c] atol=1e-12
    end
    ini = (beta1=r1.beta1, alpha1=r1.alpha1, beta2=r1.beta2, alpha2=r1.alpha2, c11=r1.c11,
           c12=r1.c12, c21=r1.c21, c22=r1.c22, dx=r1.dx, dpx=r1.dpx, dy=r1.dy, dpy=r1.dpy,
           x=r1.x, px=r1.px, y=r1.y, py=r1.py, pz=r1.pz)
    @test twiss(ring; initial=ini, cols).df == tw.df
    # A Dict with String keys (as from a Python dict), δ from delta0
    dct = Dict{String,Any}(String(k) => v for (k, v) in pairs(ini) if k != :pz)
    @test twiss(ring; initial=dct, delta0=r1.pz, cols).df == tw.df
    # A part of the line (start, stop): the rows of the whole open line
    i4 = findfirst(==(4), tp.df.index)
    tpart = twiss(ring; initial=tp.df[i4, :], cols, start=4, stop=9)
    @test tpart.df.index == tp.df.index[i4:i4+5] # (the last row: the beginning of stop)
    for c in setdiff(cols, ["phi1", "phi2"])
      @test tpart.df[!, c] ≈ tp.df[i4:i4+5, c] atol=1e-12
    end

    # Uncoupled thin quadrupoles and drifts: β and α propagate as Courant-Snyder
    qf = Multipole(Kn1L=0.8); qd = Multipole(Kn1L=-0.7); d = Drift(L=1.0)
    bl = Beamline([qf, d, qd, d]; species_ref=Species("electron"), pc_ref=1e9)
    to = twiss(bl; initial=(beta1=3.0, alpha1=-0.4, beta2=5.0, alpha2=1.2))
    quad = (k -> [1 0; -k 1.0], k -> [1 0; k 1.0]) # Thin quadrupole in x and y
    for (p, β0, α0, β, α, ϕ) in ((1, 3.0, -0.4, to.df.beta1, to.df.alpha1, to.df.phi1),
                                  (2, 5.0, 1.2, to.df.beta2, to.df.alpha2, to.df.phi2))
      γ0 = (1 + α0^2)/β0
      T = [1 0; 0 1.0]
      @test β[1] ≈ β0 && α[1] ≈ α0
      for (i, m) in enumerate([quad[p](0.8), [1 1; 0 1.0], quad[p](-0.7), [1 1; 0 1.0]])
        T = m*T
        @test β[i+1] ≈ T[1,1]^2*β0 - 2T[1,1]*T[1,2]*α0 + T[1,2]^2*γ0 atol=1e-13
        @test α[i+1] ≈ -T[1,1]*T[2,1]*β0 + (T[1,1]*T[2,2] + T[1,2]*T[2,1])*α0 -
            T[1,2]*T[2,2]*γ0 atol=1e-13
        @test ϕ[i+1] ≈ atan(T[1,2], T[1,1]*β0 - T[1,2]*α0)/(2pi) atol=1e-14
      end
    end

    # chrom=2: δ derivatives against central differences in δ (the start orbit moving with the
    # dispersion, the start optics fixed)
    tc = twiss(ring; initial=ini, chrom=2, cols=["beta1", "dbeta1", "dc12", "dx_2"])
    @test tc.df.beta1 ≈ tw.df.beta1 atol=1e-13
    h = 1e-6
    mv(s) = merge(ini, (x=ini.x + s*h*ini.dx, px=ini.px + s*h*ini.dpx, y=ini.y + s*h*ini.dy,
                        py=ini.py + s*h*ini.dpy, pz=ini.pz + s*h))
    tpl = twiss(ring; initial=mv(1), cols=["beta1", "c12", "dx"])
    tmi = twiss(ring; initial=mv(-1), cols=["beta1", "c12", "dx"])
    for (c, cd) in (("beta1", "dbeta1"), ("c12", "dc12"), ("dx", "dx_2"))
      @test tc.df[!, cd] ≈ (tpl.df[!, c] .- tmi.df[!, c])/2h atol=1e-7
    end

    # Values that are Taylor series in GTPSA parameters, against central differences
    D = Descriptor([1, 1, 1, 1, 1, 1], 2, [1, 1], 1)
    p = @params(D)
    inip = merge(ini, (c12=ini.c12 + p[1], x=ini.x + p[2]))
    tt = twiss(ring; initial=inip, cols=["beta1", "c12", "y"], GTPSA_descriptor=D)
    @test [t[0] for t in tt.df.beta1] ≈ tw.df.beta1 atol=1e-13
    fdmax = Dict{Tuple{Symbol,String},Float64}()
    for (k, name) in enumerate((:c12, :x))
      tpp = twiss(ring; initial=merge(ini, NamedTuple{(name,)}((ini[name] + h,))),
                  cols=["beta1", "c12", "y"])
      tmm = twiss(ring; initial=merge(ini, NamedTuple{(name,)}((ini[name] - h,))),
                  cols=["beta1", "c12", "y"])
      for c in ("beta1", "c12", "y")
        fd = (tpp.df[!, c] .- tmm.df[!, c])/2h
        @test [GTPSA.deriv(t, 6 + k)[0] for t in tt.df[!, c]] ≈ fd atol=5e-8
        fdmax[(name, c)] = maximum(abs, fd)
      end
    end
    # (not trivially 0; the orbit does not depend on the coupling)
    @test fdmax[(:c12, "beta1")] > 0.05 && fdmax[(:x, "beta1")] > 1 && fdmax[(:x, "y")] > 0.005
    @test fdmax[(:c12, "y")] == 0

    # Errors
    @test_throws "beta2" twiss(ring; initial=(beta1=1.0,))
    @test_throws "beta_2" twiss(ring; initial=(beta1=1.0, beta2=1.0, beta_2=3))
    @test_throws "delta" twiss(ring; initial=(beta1=1.0, beta2=1.0, delta=1e-3))
    @test_throws "delta0" twiss(ring; initial=(beta1=1.0, beta2=1.0, pz=1e-3), delta0=1e-3)
    @test_throws "v0" twiss(ring; initial=(beta1=1.0, beta2=1.0), v0=zeros(1, 6))
    @test_throws "a_initial" twiss(ring; initial=(beta1=1.0, beta2=1.0),
                                   a_initial=twiss(ring; cols=["a"], as_taylor_series=true).a[1])
    @test_throws "spin" twiss(ring; initial=(beta1=1.0, beta2=1.0), spin=true)
    @test_throws "determinant" twiss(ring; initial=(beta1=1.0, beta2=1.0, c11=1.0, c22=1.0))
    # (a map without synchrotron motion would lose the energy change of a cavity)
    cav = RFCavity(L=0.5, voltage=5e6, rf_frequency=5e8, zero_phase=PhaseRef.BelowTransition)
    linac = Beamline([Drift(L=1.0), cav, Drift(L=1.0)]; E_ref=5e9, species_ref=Species("electron"))
    @test_throws "RF cavities" twiss(linac; initial=(beta1=1.0, beta2=1.0))
    @test twiss(linac; initial=(beta1=1.0, beta2=1.0), rf_on=false).beta1[end] ≈ 1 + 2.5^2
    # the same for a coasting a_initial; a cavity without voltage is fine
    acoast = twiss(Beamline([Quadrupole(Kn1=0.36, L=0.5), Drift(L=1.0), Quadrupole(Kn1=-0.36,
                   L=0.5), Drift(L=1.0)]; E_ref=5e9, species_ref=Species("electron"));
                   cols=["a"], as_taylor_series=true, at=[1]).a[1]
    @test_throws "RF cavities" twiss(linac; a_initial=acoast)
    cav0 = RFCavity(L=0.5, voltage=0.0, rf_frequency=5e8, zero_phase=PhaseRef.BelowTransition)
    linac0 = Beamline([Drift(L=1.0), cav0, Drift(L=1.0)]; E_ref=5e9,
                      species_ref=Species("electron"))
    @test twiss(linac0; initial=(beta1=1.0, beta2=1.0)).beta1[end] ≈ 1 + 2.5^2
    @test twiss(linac0; a_initial=acoast).beta1[end] ≈ twiss(linac0; a_initial=acoast,
                                                            rf_on=false).beta1[end]

    # TwissExpr with a Context as initial: its variables are knobs
    ctx = Context(beta1=8.0, alpha1=-0.5, beta2=6.0, alpha2=0.3, c12=0.02, dx=0.3, x=1e-3)
    nt() = (beta1=ctx.beta1, alpha1=ctx.alpha1, beta2=ctx.beta2, alpha2=ctx.alpha2, c12=ctx.c12,
            dx=ctx.dx, x=ctx.x)
    q1 = Quadrupole(Kn1=0.3, L=0.4); q2 = Quadrupole(Kn1=-0.35, L=0.4); dl = Drift(L=1.0)
    line = Beamline([dl, q1, dl, SBend(L=1.0, g_ref=0.04, angle=0.04), Quadrupole(Ks1=0.04, L=0.2),
                     q2, dl, q1, dl], E_ref=5e9, species_ref=Species("electron"))
    te = TwissExpr(line; initial=ctx)
    exprs = [te.beta1[end], te.alpha1[end], te.c12[end], te.y[end]]
    cs = ["beta1", "alpha1", "c12", "y"]
    vals(t) = [t.beta1[end], t.alpha1[end], t.c12[end], t.y[end]]
    @test value(exprs) == vals(twiss(line; initial=nt(), cols=cs))
    knobs = [@knob(ctx.beta1), @knob(ctx.alpha2), @knob(ctx.c12), @knob(ctx.x), @knob(q2.Kn1)]
    val, J = value_and_jacobian(exprs, knobs)
    @test val ≈ vals(twiss(line; initial=nt(), cols=cs)) atol=1e-13
    @test ctx.beta1 === 8.0 && q2.Kn1 === -0.35 # (numbers again)
    for (k, kn) in enumerate(knobs)
      x0 = SciBmad._getknob(kn)
      SciBmad._setknob!(kn, x0 + h); vp = vals(twiss(line; initial=nt(), cols=cs))
      SciBmad._setknob!(kn, x0 - h); vm = vals(twiss(line; initial=nt(), cols=cs))
      SciBmad._setknob!(kn, x0)
      @test J[:, k] ≈ (vp .- vm)/2h atol=2e-8
    end
    res = match!(vary=[@knob(ctx.beta1), @knob(ctx.alpha1)],
                 targets=[te.beta1[end] == 12.0, te.alpha1[end] == 0.0])
    t1 = twiss(line; initial=nt(), cols=cs)
    @test t1.beta1[end] ≈ 12.0 atol=1e-8
    @test t1.alpha1[end] ≈ 0.0 atol=1e-8
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
      # Central differences of value(exprs) in the knobs (a function in a testset shares the
      # testset's variables: its own have names not used below)
    function fdjac(exprs, knobs; h=1e-6)
      fd_x0 = [SciBmad._knob_value(k) for k in knobs]
      fd_J = zeros(length(exprs), length(knobs))
      for fd_j in eachindex(knobs)
        fd_x = copy(fd_x0)
        fd_x[fd_j] += h
        setknobs!(knobs, fd_x)
        fd_vp = vec(value(exprs))
        fd_x[fd_j] -= 2h
        setknobs!(knobs, fd_x)
        fd_J[:, fd_j] = (fd_vp .- vec(value(exprs))) ./ 2h
      end
      setknobs!(knobs, fd_x0)
      return fd_J
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
      e = [tw.q1, tw.q2, tw.beta1[qd], tw.beta2[end], tw.x[5],
           tw.dx[[b], :exit]..., mean(tw.beta1[3, :]), maximum(tw.beta2[:]),
           tw.phi1[7] - tw.phi1[2],
           sqrt(sum(abs2(tw.x["q*"]))), tw.s[end], tw.beta1[1]^2 - 5*mean(tw.alpha1[3, :])]
      push!(e, tw.etac, tw.alphac) # (with RF and parameters they lacked the factor sin(2π q3))
      rf_on && push!(e, tw.q3)
      t = twiss(ring; rf_on)
      r3 = [findall(==(3), t.df.index); findfirst(==(4), t.df.index)] # the 4 steps of b, its end
      @test length(r3) == 5
      v = value(e)
      @test v[1:4] ≈ [t.q1, t.q2, t.df.beta1[findfirst(==(5), t.df.index)], t.df.beta2[end]] rtol=1e-12
      @test v[8] ≈ sum(t.df.beta1[r3])/5 rtol=1e-12
      @test v[9] ≈ maximum(t.df.beta2) rtol=1e-12
      @test v[13] ≈ t.df.beta1[1]^2 - 5*sum(t.df.alpha1[r3])/5 rtol=1e-12
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
    @test v ≈ [t3.q1[delta=1], t3.q2[delta=1], t3.q1[delta=2], t3.df.dbeta1[2], t3.df.dx_2[1],
               t3.q1[J1=1]] rtol=1e-9
    @test close_to(J, fdjac(e, ks; h=1e-5))
    @test v[1] ≈ twiss(ring; rf_on=false, chrom=2, at=[]).q1[delta=1] rtol=1e-9
      # h terms are complex
    th = twiss(ring; rf_on=false, order=2, cols=["h3000", "h2100"], at=[1, 5])
    @test value([abs(tc.h3000[1]), real(tc.h2100[5]), imag(tc.h2100[5]), abs2(tc.h3000[1])]) ≈
          [abs(th.df.h3000[1]), real(th.df.h2100[2]), imag(th.df.h2100[2]), abs2(th.df.h3000[1])] rtol=1e-10
      # Their values and derivatives are complex, but not as targets
    @test value(tc.h3000[1]) ≈ th.df.h3000[1] rtol=1e-10
    v, J = value_and_jacobian(tc.h3000[1], [@knob(sf.Kn2), @knob(qf.Kn1)])
    vr, Jr = value_and_jacobian([real(tc.h3000[1]), imag(tc.h3000[1])],
                                [@knob(sf.Kn2), @knob(qf.Kn1)])
    @test v ≈ vr[1] + im*vr[2] && J ≈ Jr[1, :] .+ im .* Jr[2, :] && abs(imag(v)) > 1e-3
    @test_throws ErrorException match!(vary=[@knob(sf.Kn2)], targets=[tc.h3000[1] == 0])
      # Their derivatives, also in knobs that change the linear optics (the quadrupole) and the
      # closed orbit (the kicker): the Courant-Snyder map of h depends on the parameters
    e = [real(tc.h3000[1]), imag(tc.h2100[5]), abs(tc.h2200[1]), real(tc.dh2000[1])]
    ks = [@knob(sf.Kn2), @knob(qf.Kn1), @knob(d.L)]
    v, J = value_and_jacobian(e, ks)
    Jfd = fdjac(e, ks; h=1e-5)
    @test all(abs.(J .- Jfd) .<= 1e-5 .* (abs.(Jfd) .+ 1e-3))
    @test abs(J[1,2]) > 1 # (the derivative in the quadrupole was wrong, not small)
    v, J = value_and_jacobian(e, [Knob(hk, :Kn0L)])
    @test all(abs.(J .- fdjac(e, [Knob(hk, :Kn0L)]; h=3e-7)) .<= 1e-5 .* abs.(J))
    t6 = TwissExpr(ring)
    e = [real(t6.h300000[1]), imag(t6.h210000[5]), real(t6.h200010[1])]
    v, J = value_and_jacobian(e, ks)
    @test all(abs.(J .- fdjac(e, ks; h=1e-5)) .<= 1e-5 .* (abs.(J) .+ 1e-3))

      # deriv: the orbit response to the kicker, and its derivative in a quadrupole (second order)
    theta = Knob(hk, :Kn0L)
    R = [deriv(tw.x[i], theta) for i in (1, 5)]
    @test value(R) ≈ vec(fdjac([tw.x[1], tw.x[5]], [theta]; h=1e-7)) rtol=1e-6
    v, J = value_and_jacobian(R, [@knob(qf.Kn1), theta])
    @test v ≈ value(R) rtol=1e-11
    @test isapprox(J, fdjac(R, [@knob(qf.Kn1), theta]; h=1e-5); rtol=1e-5)
      # A matrix of expressions (it was an error): the values in its shape, the Jacobian rows in
      # the order of vec
    M = [deriv(tw.x[1], theta) tw.beta1[5]; tw.x[5] deriv(tw.x[5], theta)]
    vm, Jm = value_and_jacobian(M, [@knob(qf.Kn1), theta])
    vv, Jv = value_and_jacobian(vec(M), [@knob(qf.Kn1), theta])
    @test size(value(M)) == (2, 2) && value(M) == reshape(value(vec(M)), 2, 2)
    @test vm == reshape(vv, 2, 2) && Jm == Jv

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
    @test v[1:5] ≈ [t.df.beta1[end], t.df.alpha1[end], t.df.beta2[3], t.df.dx[end], t.df.phi1[end]] rtol=1e-11
    @test close_to(J, fdjac(e, ks))
    tf = TwissExpr(line; a_initial=a)
    e = [tf.beta1[end], tf.alpha2[end]]
    v, J = value_and_jacobian(e, [@knob(q3.Kn1), @knob(qd.Kn1)])
    @test v ≈ [t.beta1[end], t.alpha2[end]] rtol=1e-11
    @test close_to(J, fdjac(e, [@knob(q3.Kn1), @knob(qd.Kn1)]))

      # Tunes and phases with their whole turns, whatever rows are used (6 cells)
    qf6 = Quadrupole(Kn1=1.2, L=0.5)
    qd6 = Quadrupole(Kn1=-1.1, L=0.5)
    cells = Beamline(repeat([qf6, d, b, qd6, d, b], 6); species_ref=Species("electron"),
                     pc_ref=18e9)
    t = twiss(cells)
    t6 = TwissExpr(cells)
    e = [t6.q1, t6.q2, t6.phi1[30] - t6.phi1[3]]
    ref = [t.q1, t.q2, t.df.phi1[findfirst(==(30), t.df.index)] -
           t.df.phi1[findfirst(==(3), t.df.index)]]
    @test ref[1] > 1
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
      # (the weight makes the residual of the beta function comparable with that of the tune)
    for solver in (:levenberg_marquardt, :gauss_newton)
      res = match!(vary=[kf, kd]; solver, targets=[tw.q1 == 0.2,
                   Target(maximum(tw.beta1[:]); upper=6, weight=0.02)])
      t = twiss(cell)
      @test res.converged
      @test t.q1 ≈ 0.2 atol=1e-8
      @test maximum(t.beta1) ≈ 6 atol=1e-6
      undo!(res)
    end
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
      # A step into an unstable lattice is rejected and a shorter one taken (the first step for
      # tunes of 0.49 overshoots: more evaluations than improvements), with both solvers
    for solver in (:levenberg_marquardt, :gauss_newton)
      res = match!(vary=[kf, kd]; solver, targets=[tw.q1 == 0.49, tw.q2 == 0.49])
      @test res.converged && length(res.history) < res.evaluations
      undo!(res)
    end
      # The previous solver (Gauss-Newton steps from the SVD) gives the same tunes
    res = match!(vary=[kf, kd], targets=[tw.q1 == 0.2, tw.q2 == 0.15], solver=:gauss_newton)
    @test res.converged
    @test res.x ≈ x_tunes rtol=1e-7
    undo!(res)
    @test_throws ErrorException match!(vary=[kf], targets=[tw.q1 == 0.2], solver=:newton)
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

  @testset "Orbit-only twiss" begin
      # With only orbit columns (and for a coasting beam their δ-derivatives) the rows only need
      # factorise_a0: the columns and the summary are bitwise those of the full path (a column
      # more). A wrong a0 at the rows (that of the start) fails 15 of 18 such comparisons
    flat(x::TPS) = [x[i] for i in 0:GTPSA.numcoefs(x)-1]
    flat(x::SciBmad.AmplitudeDependentValue) = flat(x.taylor_series)
    flat(x::AbstractArray) = map(flat, x)
    flat(x) = x
    function same(bl; cols, kw...)
      to = twiss(bl; cols, kw...)
      tf = twiss(bl; cols=vcat(cols, ["beta1"]), kw...)
      return all(isequal(flat(collect(to.df[!, c])), flat(collect(tf.df[!, c]))) for c in cols) &&
             all(isequal(flat(to.summ.sdict[k]), flat(v)) for (k, v) in tf.summ.sdict)
    end
    @elements begin
      qf = Quadrupole(Kn1=0.36, L=0.5, y_offset=2e-5)
      qd = Quadrupole(Kn1=-0.33, L=0.5, Ks1=1e-3, x_offset=1e-4)
      sf = Sextupole(Kn2=1.2, L=0.2, x_offset=5e-5)
      sd = Sextupole(Kn2=-1.1, L=0.2, tilt=0.1)
      d = Drift(L=0.4)
      b = SBend(L=1.2, angle=pi/132)
      rf = RFCavity(L=1e-2, voltage=3e7, rf_frequency=5e8, zero_phase=PhaseRef.AboveTransition)
    end
    ring = Beamline([qf, sf, d, b, d, qd, sd, d, b, d]; species_ref=Species("electron"),
                    pc_ref=3e9)
    ringrf = Beamline([qf, sf, d, b, d, qd, sd, d, b, d, rf]; species_ref=Species("electron"),
                      pc_ref=3e9)
    orb = ["x", "px", "y", "py", "pz"]
    @test same(ring; cols=vcat(orb, ["dx", "dpy", "dx_2"]), chrom=2)
    @test same(ring; cols=orb, order=3, at=[3, 7])
    @test same(ring; cols=vcat(orb, ["dx"]), delta0=1e-4, chrom=2)
    @test same(ringrf; cols=vcat(orb, ["z", "a0"]), order=2)
    a = twiss(ring; cols=["a"], as_taylor_series=true, at=[1]).a[1]
    line = Beamline([qf, d, qd, d, sf, d, qd]; species_ref=Species("electron"), pc_ref=3e9)
    @test same(line; cols=orb, a_initial=a)
    D = Descriptor([1, 1, 1, 1, 1, 2], 2, [2, 2], 2)
    k = params(D)
    qf.Kn1 = 0.36 + k[1]
    qd.x_offset = 1e-4 + k[2]
    @test same(ring; cols=vcat(orb, ["dx"]), GTPSA_descriptor=D)
    scalarize!(qf)
    scalarize!(qd)
    D = Descriptor(fill(1, 6), 1, [1, 1], 1)
    k = params(D)
    qf.Kn1 = 0.36 + k[1]
    qd.x_offset = 1e-4 + k[2]
    @test same(ringrf; cols=vcat(orb, ["z"]), GTPSA_descriptor=D)
    scalarize!(qf)
    scalarize!(qd)
      # For a coasting beam z and a0 take the full path (the canonisation of factorise changes
      # their dependence on δ)
    @test !SciBmad._orbit_col("z", true) && !SciBmad._orbit_col("a0", true)
    @test SciBmad._orbit_col("z", false) && SciBmad._orbit_col("dx_2", true)
    @test !SciBmad._orbit_col("dx", false) && !SciBmad._orbit_col("beta1", true)
  end

  @testset "Transfer matrix r11 ... r66 and one-turn matrix m11 ... m66" begin
      # r_ik = ∂z_i/∂z_k from the start of the table, against central differences of tracking (to
      # their error, ~1e-9 in the z row); with parameters and δ around the fixed point for them
    @elements begin
      qf = Quadrupole(Kn1=0.36, L=0.5, y_offset=2e-5)
      qd = Quadrupole(Kn1=-0.33, L=0.5, Ks1=1e-3, x_offset=1e-4)
      sf = Sextupole(Kn2=1.2, L=0.2, x_offset=5e-5)
      d = Drift(L=0.4)
      b = SBend(L=1.2, angle=pi/132)
      rf = RFCavity(L=1e-2, voltage=3e7, rf_frequency=5e8, zero_phase=PhaseRef.AboveTransition)
    end
    ring = Beamline([qf, sf, d, b, d, qd, d, b, d]; species_ref=Species("electron"), pc_ref=3e9)
    ringrf = Beamline([qf, sf, d, b, d, qd, d, b, d, rf]; species_ref=Species("electron"),
                      pc_ref=3e9)
    rc = ["r$i$k" for i in 1:6 for k in 1:6]
    orb = ["x", "px", "y", "py", "z", "pz"]
    rmat(tw, p, l) = [getproperty(tw, Symbol("$p$i$k"))[l] for i in 1:6, k in 1:6]
    rmat(tw, l) = rmat(tw, "r", l)
    function fdmat(bl, v, l; start=1, h=1e-6, kw...)
      M = zeros(6, 6)
      for k in 1:6
        vp = copy(v); vp[k] += h
        vm = copy(v); vm[k] -= h
        rp = track(bl; v0=reshape(vp, 1, 6), n_turns=1, at=[l], start, kw...)
        rm = track(bl; v0=reshape(vm, 1, 6), n_turns=1, at=[l], start, kw...)
        M[:, k] = (rp.v[turn=1, particle=1] .- rm.v[turn=1, particle=1]) ./ 2h
      end
      return M
    end
    near(R, F) = all(abs.(R .- F) .<= 1e-8 .+ 1e-6 .* abs.(F))
    for (bl, kw) in ((ring, (rf_on=false,)), (ringrf, (;)))
      tw = twiss(bl; cols=vcat(rc, orb))
      v = [getproperty(tw, Symbol(c))[1] for c in orb]
      @test all(near(rmat(tw, l), fdmat(bl, v, l; kw...)) for l in (4, 6, 8))
    end
    tw = twiss(ring; chrom=2, cols=vcat(rc, orb)) # (mixed orders)
    v = [getproperty(tw, Symbol(c))[1] for c in orb]
    @test near(rmat(tw, 8), fdmat(ring, v, 8; rf_on=false))
    a = twiss(ring; cols=["a"], as_taylor_series=true, at=[1]).a[1]
    line = Beamline([qf, d, qd, d, sf, d, b]; species_ref=Species("electron"), pc_ref=3e9)
    tl = twiss(line; a_initial=a, cols=vcat(rc, orb))
    v = [getproperty(tl, Symbol(c))[1] for c in orb]
    @test near(rmat(tl, 7), fdmat(line, v, 7; rf_on=false))
    J = zeros(6, 6)
    for i in 1:3
      J[2i-1, 2i] = 1
      J[2i, 2i-1] = -1
    end
    R = rmat(twiss(ringrf; cols=rc), 10)
    @test maximum(abs, R'*J*R - J) < 1e-13
      # The derivative in a strength (the orbit moves with it: qd is offset, sf feeds down)
    D = Descriptor(fill(1, 6), 2, [1], 2)
    k = params(D)
    qf.Kn1 = 0.36 + k[1]
    dr12 = twiss(ring; GTPSA_descriptor=D, cols=["r12"], at=[6]).r12[6][[0, 0, 0, 0, 0, 0, 1]]
    scalarize!(qf)
    h = 1e-6
    r12(kq) = (qf.Kn1 = kq; twiss(ring; cols=["r12"], at=[6]).r12[6])
    fd = (r12(0.36 + h) - r12(0.36 - h))/2h
    qf.Kn1 = 0.36
    @test dr12 ≈ fd rtol=1e-7
    te = TwissExpr(ring)
    @test value_and_jacobian([te.r12[6]], [@knob(qf.Kn1)])[2][1] ≈ fd rtol=1e-7
      # The dependence on δ of a coasting beam (also with the variables at lower order)
    for kw in ((chrom=2,), (order=2,)), c in ("r11", "r12", "r16", "r56")
      s = getproperty(twiss(ring; kw..., cols=[c], at=[6], as_taylor_series=true),
                      Symbol(c))[6][[0, 0, 0, 0, 0, 1]]
      rd(d0) = getproperty(twiss(ring; cols=[c], at=[6], delta0=d0), Symbol(c))[6]
      @test s ≈ (rd(h) - rd(-h))/2h rtol=1e-6 atol=1e-9
    end

      # m11 ... m66: the one-turn matrix at the row, against one turn of tracking from the row
      # around to it (turn 2 at the row with `start` at it); a coasting beam's z slips per turn
    mc = ["m$i$k" for i in 1:6 for k in 1:6]
    function fdturn(bl, v, l; kw...)
      M = zeros(6, 6)
      for k in 1:6
        vp = copy(v); vp[k] += h
        vm = copy(v); vm[k] -= h
        rp = track(bl; v0=reshape(vp, 1, 6), n_turns=2, at=[l], start=l, kw...)
        rm = track(bl; v0=reshape(vm, 1, 6), n_turns=2, at=[l], start=l, kw...)
        M[:, k] = (rp.v[turn=2, particle=1] .- rm.v[turn=2, particle=1]) ./ 2h
      end
      return M
    end
    for (bl, tkw, kw) in ((ring, (;), (rf_on=false,)), (ringrf, (;), (;)),
                          (ring, (chrom=2,), (rf_on=false,)))
      tw = twiss(bl; tkw..., cols=vcat(mc, orb))
      v = [getproperty(tw, Symbol(c))[6] for c in orb]
      @test near(rmat(tw, "m", 6), fdturn(bl, v, 6; kw...))
    end
    rend(tw) = [getproperty(tw, Symbol("r$i$k"))[end] for i in 1:6, k in 1:6]
    tw = twiss(ring; cols=vcat(mc, rc))
    @test maximum(abs, rmat(tw, "m", 1) .- rend(tw)) < 1e-13
    cell = [qf, sf, d, b, d, qd, d, b, d]
    ring2 = Beamline(vcat(cell, cell); species_ref=Species("electron"), pc_ref=3e9)
    tp = twiss(ring2; stop=10, cols=vcat(mc, rc)) # (the first cell, periodic by itself)
    @test maximum(abs, rmat(tp, "m", 1) .- rend(tp)) < 1e-13 && rmat(tp, "m", 1) ≈ rend(tw)
    qf.Kn1 = 0.36 + k[1]
    dm12 = twiss(ring; GTPSA_descriptor=D, cols=["m12"], at=[6]).m12[6][[0, 0, 0, 0, 0, 0, 1]]
    scalarize!(qf)
    m12(kq) = (qf.Kn1 = kq; twiss(ring; cols=["m12"], at=[6]).m12[6])
    fd = (m12(0.36 + h) - m12(0.36 - h))/2h
    qf.Kn1 = 0.36
    @test dm12 ≈ fd rtol=1e-7
    @test value_and_jacobian([te.m12[6]], [@knob(qf.Kn1)])[2][1] ≈ fd rtol=1e-7
    for c in ("m11", "m16", "m52", "m56")
      s = getproperty(twiss(ring; chrom=2, cols=[c], at=[6], as_taylor_series=true),
                      Symbol(c))[6][[0, 0, 0, 0, 0, 1]]
      md(d0) = getproperty(twiss(ring; cols=[c], at=[6], delta0=d0), Symbol(c))[6]
      @test s ≈ (md(h) - md(-h))/2h rtol=1e-6 atol=1e-9
    end
    @test_throws ErrorException twiss(line; a_initial=a, cols=["m11"]) # (needs a periodic table)
  end

  @testset "Knobs on vectors, function knobs, display of many knobs" begin
      # A vector variable of the context is one knob per entry: the same values, derivatives and
      # match as two variables, and the vector itself is the same afterwards
    @elements begin
      qf = Quadrupole(Kn1=DefExpr(c -> c.k[1]), L=0.5)
      qd = Quadrupole(Kn1=DefExpr(c -> c.k[2]), L=0.5)
      d = Drift(L=1.0)
      b = SBend(L=1.2, angle=pi/132)
    end
    vring = Beamline([qf, d, b, qd, d, b]; species_ref=Species("electron"), pc_ref=18e9)
    vring.context.k = [0.36, -0.36]
    @elements begin
      qf2 = Quadrupole(Kn1=DefExpr(c -> c.kf), L=0.5)
      qd2 = Quadrupole(Kn1=DefExpr(c -> c.kd), L=0.5)
    end
    sring = Beamline([qf2, d, b, qd2, d, b]; species_ref=Species("electron"), pc_ref=18e9)
    sring.context.kf = 0.36
    sring.context.kd = -0.36
    tv = TwissExpr(vring)
    ts = TwissExpr(sring)
    k = vring.context.k
    kv = @knob vring.context.k
    @test [x.name for x in SciBmad._knobs([kv])] == ["k[1]", "k[2]"]
    v, J = value_and_jacobian([tv.q1, tv.beta1[4]], [kv])
    vs, Js = value_and_jacobian([ts.q1, ts.beta1[4]], [@knob(sring.context.kf),
                                                        @knob(sring.context.kd)])
    @test v == vs && J == Js
    @test vring.context.k === k && k == [0.36, -0.36]
    @test value_and_jacobian([tv.q1], [@knob(vring.context.k[2])])[2] == J[1:1, 2:2]
    @test value_and_jacobian([tv.q1], [vring.context => :k])[2] == J[1:1, :]
    ka = Any[0.36, -0.36] # (a vector that could hold the Taylor series is not written)
    vring.context.k = ka
    @test value_and_jacobian([tv.q1, tv.beta1[4]], [kv]) == (v, J)
    @test vring.context.k === ka && all(x -> x isa Float64, ka)
    vring.context.k = k
    res = match!(vary=[kv], targets=[tv.q1 == 0.07, tv.q2 == 0.06])
    ress = match!(vary=[@knob(sring.context.kf), @knob(sring.context.kd)],
                  targets=[ts.q1 == 0.07, ts.q2 == 0.06])
    @test res.converged && res.x == ress.x
    @test vring.context.k === k && k == ress.x
    @test occursin("k[2]", sprint(show, res))
    @test_throws ErrorException value(deriv(tv.q1, kv)) # (one entry, not the vector)

      # The display in the REPL shows as many knobs as fit, print shows all of them
    vring.context.c = zeros(40)
    prob = MatchProblem(vary=[@knob(vring.context.c)], targets=[tv.q1 == 0.07])
    r = SciBmad.MatchResult(prob, true, "all targets met", 0, 1, 0.0, zeros(40), zeros(40),
                            [0.07], [0.0])
    full = sprint(print, r)
    short = sprint(show, MIME"text/plain"(), r; context=(:limit=>true, :displaysize=>(20, 80)))
    @test occursin("c[40]", full) && !occursin("more", full)
    @test !occursin("c[40]", short) && occursin("more", short) && count('\n', short) <= 20
    @test sprint(show, MIME"text/plain"(), r) == full # (not limited)

      # A function knob may set element parameters: they are numbers again afterwards
    @elements begin
      qa = Quadrupole(Kn1=0.36, L=0.5)
      qb = Quadrupole(Kn1=0.36, L=0.5)
      qc = Quadrupole(Kn1=-0.36, L=0.5)
    end
    fring = Beamline([qa, d, qc, d, qb, d, qc, d]; species_ref=Species("electron"), pc_ref=18e9)
    fam = Knob(() -> qa.Kn1, v -> (qa.Kn1 = v; qb.Kn1 = v); name="qa and qb")
    v, J = value_and_jacobian([TwissExpr(fring).q1], [fam])
    @test qa.Kn1 isa Float64 && qb.Kn1 isa Float64
    @test twiss(fring).q1 ≈ v[1] rtol=1e-13 # (plain twiss runs: it aborted Julia before)
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
      @test tc.df[!, c] ≈ to.df[!, c] rtol=1e-9
    end
    @test [tc.q1[delta=1], tc.q2[delta=1]] ≈ [to.q1[delta=1], to.q2[delta=1]] rtol=1e-10
    t3 = twiss(bl; chrom=3, cols=["dbeta1_2", "dx_3"], at=[1, 5])
    o3 = twiss(bl; order=3, cols=["dbeta1_2", "dx_3"], at=[1, 5])
    @test t3.df.dbeta1_2 ≈ o3.df.dbeta1_2 rtol=1e-9
    @test t3.df.dx_3 ≈ o3.df.dx_3 rtol=1e-9
    @test [t3.q1[delta=2], t3.q2[delta=2]] ≈ [o3.q1[delta=2], o3.q2[delta=2]] rtol=1e-9
      # From the one-turn matrix around the closed orbit, by differences (its variables have names
      # not used below: a function in a testset shares the testset's variables)
    function mode_tunes(delta)
      mt_co = find_closed_orbit(bl; v0=[0. 0. 0. 0. 0. delta]).v0
      mt_M = zeros(4, 4)
      for mt_j in 1:4
        mt_vp = copy(mt_co)
        mt_vp[1,mt_j] += 1e-6
        mt_vm = copy(mt_co)
        mt_vm[1,mt_j] -= 1e-6
        mt_M[:,mt_j] = (track(bl; v0=mt_vp).data.v[1,1:4,1,end] .-
                        track(bl; v0=mt_vm).data.v[1,1:4,1,end]) ./ 2e-6
      end
      return sort(filter(>(0), angle.(SciBmad.LinearAlgebra.eigvals(mt_M)) ./ 2pi))
    end
    @test mode_tunes(0.0) ≈ sort([tc.q1[], tc.q2[]]) rtol=1e-8
    chroms = (mode_tunes(1e-4) .- mode_tunes(-1e-4)) ./ 2e-4
    @test chroms ≈ [tc.q2[delta=1], tc.q1[delta=1]] rtol=1e-5 # (q2 is the lower tune)
    @test abs(chroms[1] - chroms[2]) > 0.01
      # delta0: the functions around the closed orbit at δ (the map is tracked at δ, also with
      # higher orders), against tracking and against the expansion in δ from δ = 0
    co = find_closed_orbit(bl; v0=[0. 0. 0. 0. 0. 1e-4]).v0
    o4 = twiss(bl; order=4, cols=["dx_2", "dx_3"], at=[1])
    for kw in ((;), (order=2,), (chrom=2,))
      td = twiss(bl; delta0=1e-4, cols=["x", "px", "pz", "dx"], at=[1], kw...)
      @test sort([td.q1[], td.q2[]]) ≈ mode_tunes(1e-4) rtol=1e-8
      @test [td.x[1], td.px[1], td.pz[1]] ≈ [co[1], co[2], 1e-4] rtol=1e-11
      @test td.dx[1] ≈ tc.dx[1] + 1e-4*o4.dx_2[1] + 0.5e-8*o4.dx_3[1] rtol=1e-9
    end
    td = twiss(bl; delta0=1e-4, chrom=2, at=[])
    @test td.q1[delta=1] ≈ o4.q1[delta=1] + 1e-4*o4.q1[delta=2] + 0.5e-8*o4.q1[delta=3] rtol=1e-8
    @test abs(mode_tunes(1e-4)[2] - mode_tunes(0.0)[2]) > 1e-6
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

  @testset "Twiss table indexed with locations" begin
      # A column of the table is indexed with locations exactly as a TwissExpr (an integer is a
      # beamline index) and gives the values of the rows of the table there (found here by hand
      # from its index column); it iterates and broadcasts over all rows. The chromaticities
    @elements begin
      qf = Quadrupole(Kn1=0.36, L=0.5)
      sf = Sextupole(Kn2=1.2, L=0.2)
      d = Drift(L=0.1)
      b = SBend(L=1.2, angle=pi/132, n_steps=3)
      qd = Quadrupole(Kn1=-0.36, L=0.5)
    end
    cell = Beamline([qf, sf, b, d, qd, b, d]; species_ref=Species("electron"), pc_ref=18e9)
    tw = twiss(cell)
    te = TwissExpr(cell)
    df = tw.df
    rows_of(i) = findall(==(i), df.index) # the rows of element i, in the order of its steps
    first_row(i) = first(rows_of(i))
    @test collect(tw.beta1) == df.beta1 && maximum(tw.beta1) == maximum(df.beta1)
    @test tw.beta1 == df.beta1 && tw.beta1 ≈ twiss(cell; cols=["beta1"]).beta1
    @test tw.dx ./ tw.beta1 == df.dx ./ df.beta1 && length(tw.beta1) == size(df, 1)
    @test tw.beta1[4] == df.beta1[first_row(4)] != df.beta1[4] # a beamline index, not a row
    @test tw.beta1[qd] == df.beta1[first_row(5)] && tw.beta1[[qd]] == [tw.beta1[qd]]
    @test tw.beta1[qd, :exit] == tw.beta1[5, :exit] == df.beta1[first_row(6)]
    @test tw.beta1[[b]] == df.beta1[[first_row(3), first_row(6)]] # placed twice
    @test tw.beta1[[b], :] == df.beta1[[rows_of(3); first_row(4); rows_of(6); first_row(7)]]
    @test tw.beta1[[b], 2] == df.beta1[[rows_of(3)[3], rows_of(6)[3]]]
    @test tw.beta1[qd, :] == df.beta1[[rows_of(5); first_row(6)]]
      # An element placed several times must be chosen (with its beamline index), also with a
      # position; the message says where it is
    for loc in Any[(b,), (b, :), (b, 2), (b, :exit)]
      @test_throws ErrorException tw.beta1[loc...]
      @test_throws ErrorException te.beta1[loc...]
    end
    @test occursin("placed 2 times in the beamline (beamline indexes 3, 6)",
                   sprint(showerror, try tw.beta1[b] catch e; e end))
    @test tw.beta1[2:3] == tw.beta1[[sf, cell[3]]] == df.beta1[[first_row(2), first_row(3)]]
    @test tw.beta1[(1.0, 1.8)] == df.beta1[findall(x -> 1.0 <= x < 1.8, df.s)]
    @test tw.beta1[:] == df.beta1
    @test tw.beta1[begin] == df.beta1[1] && tw.beta1[end] == df.beta1[end]
    @test tw.beta1[cell[7], :exit] == tw.beta1[end] == tw.beta1[end, :exit]
    @test tw.beta1[begin, :exit] == tw.beta1[sf] && value(te.beta1[begin, :exit]) == tw.beta1[sf]
    @test tw.name["Sextupole::*"] == ["sf"]
    for loc in Any[(3,), (qd,), (qd, :exit), ([b],), ([b], :), ([b], 2), (qd, :), (2:3,),
                   ([sf, cell[3]],), ((1.0, 1.8),), (:,), ("q*",)]
      @test tw.beta1[loc...] == value(te.beta1[loc...])
    end
    @test tw.beta1[begin] == value(te.beta1[begin]) && tw.beta1[end] == value(te.beta1[end])
    @test_throws ErrorException twiss(cell; at=[1]).beta1[qd] # the row is not in the table
    @test_throws ErrorException twiss(cell; at=[3]).beta1[3, 1]
    @test_throws ErrorException tw.dq1 # needs order 2
      # As an index, `end` is the beamline index of the end, n+1, which is like a zero-length
      # element: its entrance and its exit are the same location
    n = length(cell.line)
    @test tw.beta1[end-1] == tw.beta1[n] == df.beta1[first_row(n)] &&
          tw.beta1[n+1] == tw.beta1[end]
    @test tw.beta1[2:end] == df.beta1[[first_row.(2:n); size(df, 1)]]
    @test tw.beta1[end, :entrance] == tw.beta1[end, :exit] == tw.beta1[end, 1] == tw.beta1[end] &&
          tw.beta1[end, :] == [tw.beta1[end]]
    @test value(te.beta1[end-1]) == tw.beta1[end-1] && value(te.beta1[end, :exit]) == tw.beta1[end]
    @test_throws ErrorException tw.beta1[end, 2] # (no steps after the end)
    @test_throws ErrorException tw.beta1[end, :bogus]
    @test_throws ErrorException tw.beta1[n+2]
    @test_throws ErrorException tw.beta1[] # (no location)
    @test_throws ErrorException te.beta1[end, 2]
      # first and last are the first and the last row
    ts = twiss(cell; at=[qd])
    @test first(ts.beta1) == last(ts.beta1) == ts.df.beta1[1] == ts.beta1[qd]
    @test last(tw.beta1) == df.beta1[end] && first(tw.beta1) == df.beta1[1]
      # `at` takes the same locations (names too), but an element is every placement; entries
      # not in the beamline are errors. An element in `at` keeps its row when an s-range has other
      # steps of it (it was lost), and qd begins at s = 2.0 exactly (the s of the steps were
      # summed; it was 1.9999999999999998)
    @test twiss(cell; at=["qd"]).df.index == [5] && twiss(cell; at=["q*", 7]).df.index == [1, 5, 7]
    @test twiss(cell; at=[b]).df.index == twiss(cell; at=[[b]]).df.index == [3, 6]
    @test track(cell; v0=zeros(1, 6), at=[b]).index == [3, 6]
    @test twiss(cell; at=[b, (1.0, 2.0)]).df.index == [3, 3, 3, 4, 6]
    @test_throws ErrorException twiss(cell; at=["nothing*"])
    @test_throws ErrorException twiss(cell; at=[n+2])
    @test_throws ErrorException twiss(cell; at=[Drift(L=1.0)])
    @test_throws ErrorException track(cell; v0=zeros(1, 6), at=["nothing*"])
      # The first row inside the first element has its phase from the beginning of the beamline
      # (it was 0)
    q4 = Quadrupole(Kn1=0.36, L=0.5, n_steps=4)
    cell4 = Beamline([q4, sf, b, d, qd, b, d]; species_ref=Species("electron"), pc_ref=18e9)
    t4, t4r = twiss(cell4), twiss(cell4; at=[(0.2, Inf)])
    @test t4r.df.s[1] == 0.25 && t4r.df.phi1[1] == t4.df.phi1[3] > 0 &&
          t4r.df.phi2[1] == t4.df.phi2[3] > 0
      # s of a location: s at the beginning of its element (Beamlines' ele.s) plus k*L/n, for the
      # table and for indexing alike, so that an s-range from the s of a row starts at that row
      # (with steps of L/3 and L/7, which do not add up exactly: the table summed the steps)
    @elements begin
      q1 = Quadrupole(Kn1=0.3, L=0.1, n_steps=3)
      d1 = Drift(L=0.7, n_steps=7)
      q2 = Quadrupole(Kn1=-0.3, L=0.1, n_steps=3)
      d2 = Drift(L=0.3)
    end
    fodo = Beamline(repeat([q1, d1, q2, d2], 20); species_ref=Species("electron"), pc_ref=18e9)
    tf = twiss(fodo; cols=String[], base_cols=["index", "s"])
    sr = tf.df.s
    @test all(r -> tf.s[(sr[r], Inf)] == sr[sr .>= sr[r]], eachindex(sr))
    @test [tf.s[i] for i in eachindex(fodo.line)] == [ele.s for ele in fodo.line] &&
          tf.s[end] == fodo.line[end].s_downstream
    @test all(r -> twiss(fodo; at=[(0.0, sr[r])], cols=String[]).s[(0.0, sr[r])] == sr[1:r-1],
              (7, 50, 151))
      # Elements made from elements made from qd (copies of copies) are qd too
    sub = Beamline(Beamline(repeat([qf, sf, b, d, qd, b, d], 2); species_ref=Species("electron"),
                            pc_ref=18e9).line[8:14]; species_ref=Species("electron"), pc_ref=18e9)
    tsub = twiss(sub)
    @test tsub.beta1[qd] == tsub.beta1[5] == tw.beta1[qd] && tsub.beta1[[b]] == tw.beta1[[b]]
    @test twiss(sub; at=[qd]).df.index == [5] &&
          track(sub; v0=zeros(1, 6), start=qd, at=[qd]).index == [5]
      # The elements of a vector (or of `at`) are found in one pass over the line: the same as one
      # at a time, also nested, repeated, an element with a copy of it, and copies of copies
    v = Any[qd, [b, sf], cell.line[6], qd, 2]
    @test tw.beta1[v] == [tw.beta1[qd]; tw.beta1[[b]]; tw.beta1[sf]; tw.beta1[6]; tw.beta1[qd];
                          tw.beta1[2]]
    @test value(te.beta1[v]) == tw.beta1[v]
    @test twiss(cell; at=v).df.index == [2, 3, 5, 6]
    @test tsub.beta1[[qd, b]] == [tsub.beta1[qd]; tsub.beta1[[b]]] == tw.beta1[[qd, b]]
    @test twiss(sub; at=[qd, b]).df.index == [3, 5, 6]
    tc = twiss(cell; chrom=2, at=[])
    @test tc.dq1 == tc.q1[delta=1] && tc.dq2 == tc.q2[delta=1]
    @test tc.dq1 ≈ value(te.dq1) rtol=1e-12
  end

  @testset "Truncation orders from the columns, unresolved terms" begin
      # Without order and chrom, twiss takes the orders the columns need (as TwissExpr): the same
      # tables as with them given. A term of a tune that the truncation does not resolve is an
      # error (it was silently zero or inexact); the resolved ones equal those of order 6
    @elements begin
      qf = Quadrupole(Kn1=0.36, L=0.5)
      sf = Sextupole(Kn2=1.2, L=0.2)
      d = Drift(L=0.1)
      b = SBend(L=1.2, angle=pi/132)
      qd = Quadrupole(Kn1=-0.36, L=0.5, Ks1=1e-3)
      sd = Sextupole(Kn2=-1.2, L=0.2)
      oc = Multipole(Kn3L=50.0)
    end
    fodo = Beamline([qf, sf, b, d, qd, sd, b, d, oc]; species_ref=Species("electron"), pc_ref=18e9)
    for (cols, kw) in ((["h3000"], (order=2,)), (["dx_2", "dbeta1"], (chrom=2,)),
                       (["h2200", "beta1"], (order=3,)), (["w1", "dbeta1_2"], (chrom=3,)))
      @test twiss(fodo; cols).df == twiss(fodo; cols, kw...).df
    end
    t6 = twiss(fodo; order=6, at=[])
    for (term, ok, bad) in (((delta=1,), (chrom=2,), (order=1,)),
                            ((delta=2,), (chrom=3,), (chrom=2,)),
                            ((J1=1,), (order=3,), (order=2,)),
                            ((delta=1, J1=1), (order=4,), (order=3,)),
                            ((J1=1, J2=1), (order=5,), (order=4, chrom=5)))
      @test getterm(twiss(fodo; at=[], ok...).q1; term...) ≈ getterm(t6.q1; term...) rtol=1e-8
      @test_throws ErrorException getterm(twiss(fodo; at=[], bad...).q1; term...)
    end
    @test_throws ErrorException twiss(fodo; order=2, at=[]).q1[J1=1]
      # The spin tune comes from the constant part of the spin map: its terms need one order less
    t6 = twiss(fodo; order=6, at=[], spin=true)
    for (term, ok, bad) in (((delta=1,), (order=1,), ()), ((delta=2,), (chrom=2,), (order=1,)),
                            ((J1=1,), (order=2,), (order=1, chrom=2)),
                            ((delta=1, J1=1), (order=3,), (order=1, chrom=3)),
                            ((J1=2,), (order=4,), (order=3, chrom=4)))
      @test getterm(twiss(fodo; at=[], spin=true, ok...).qspin; term...) ≈
            getterm(t6.qspin; term...) rtol=1e-8
      bad == () || @test_throws ErrorException getterm(twiss(fodo; at=[], spin=true, bad...).qspin;
                                                       term...)
    end
      # Indexing gives a derivative (the coefficient of getterm times the factorials), in tables
      # and in TwissExpr alike; dq1_n is q1[delta=n]
    t5 = twiss(fodo; order=5, at=[])
    te = TwissExpr(fodo)
    @test t5.q1[delta=2] == 2*getterm(t5.q1; delta=2) && t5.q2[delta=3] == 6*getterm(t5.q2; delta=3)
    @test t5.q1[J1=2] == 2*getterm(t5.q1; J1=2) && t5.q1[J1=1, J2=1] == getterm(t5.q1; J1=1, J2=1)
    @test t5.dq1 == t5.dq1_1 == t5.q1[delta=1] && t5.dq1_2 == t5.q1[delta=2]
    @test t5.dq2_3 == t5.q2[delta=3]
    @test value([te.q1[delta=2], te.dq1_2, te.q1[J1=2], te.dq2_3, te.q1[delta=1, J1=1]]) ≈
          [t5.q1[delta=2], t5.q1[delta=2], t5.q1[J1=2], t5.q2[delta=3],
           t5.q1[delta=1, J1=1]] rtol=1e-8
    @test_throws ErrorException te.dq1[J1=1] # (it was q1[J1=1], without the δ)
      # TwissExpr takes the order a term needs, for the spin tune one less than for the tunes
    tes = TwissExpr(fodo; spin=true)
    t6s = twiss(fodo; order=6, at=[], spin=true)
    @test SciBmad._leaf_order(tes.qspin[delta=1, J1=1]) == 3 &&
          SciBmad._leaf_order(tes.q1[delta=1, J1=1]) == 4 &&
          SciBmad._leaf_order(tes.qspin[delta=1]) == 1
    @test value([tes.qspin[delta=1], tes.qspin[delta=2], tes.qspin[J1=1],
                 tes.qspin[delta=1, J1=1]]) ≈
          [t6s.qspin[delta=1], t6s.qspin[delta=2], t6s.qspin[J1=1],
           t6s.qspin[delta=1, J1=1]] rtol=1e-8
  end

  @testset "Phases and tunes with few rows" begin
      # The phases of a row are known modulo 1 from its normal form; whole turns come from the
      # linear maps at every integration step when the rows are not all steps. A ring of 5 cells
      # with tunes of 0.97 and 0.79: rows half a turn and more apart. The rows of a twiss with few
      # rows equal those of the full table bitwise (6D and coasting, orders, spin, an open line),
      # and so do the tunes and the other summary values (the 6D etac and alphac used B3 of the
      # last row of the table instead of the end); with GTPSA parameters also the derivatives
    qf = Quadrupole(Kn1=1.2, L=0.5)
    d = Drift(L=0.1)
    cellp = [qf, Sextupole(Kn2=1.2, L=0.2), SBend(L=1.2, angle=pi/132, n_steps=4), d,
             Quadrupole(Kn1=-1.1, L=0.5, tilt=0.01), Sextupole(Kn2=-1.2, L=0.2),
             SBend(L=1.2, angle=pi/132, n_steps=4), d, Marker()]
    ring = Beamline([repeat(cellp, 5)..., HKicker(Kn0L=2e-4), RFCavity(L=1e-2, voltage=1e7,
                    rf_frequency=1e9, zero_phase=PhaseRef.AboveTransition)];
                    species_ref=Species("electron"), E_ref=18e9)
    num(x) = x isa AmplitudeDependentValue ? scalar(x.taylor_series) : x
    a = twiss(ring; rf_on=false, cols=["a"], as_taylor_series=true, at=[1]).a[1]
    for kw in ((;), (rf_on=false,), (rf_on=false, order=2), (rf_on=false, chrom=2), (spin=true,),
               (rf_on=false, a_initial=a))
      full = twiss(ring; kw...)
      @test num(full.q1) > 0.9 && num(full.q2) > 0.7
      for at in (Any[], Any[20], Any[3, 40, ring.line[45]], Any[(3.0, 3.5), 30])
        few = twiss(ring; at, kw...)
        @test (num(few.q1), num(few.q2)) == (num(full.q1), num(full.q2))
        @test keys(few.summ.sdict) == keys(full.summ.sdict)
        @test all(num(few.summ.sdict[k]) == num(full.summ.sdict[k]) for k in keys(full.summ.sdict))
        fe, fu = few.df, full.df
        for r in 1:size(fe, 1)
          j = findfirst(k -> fu.index[k] == fe.index[r] && fu.s[k] == fe.s[r], 1:length(fu.s))
          @test (fe.phi1[r], fe.phi2[r], fe.beta1[r]) == (fu.phi1[j], fu.phi2[j], fu.beta1[j])
        end
      end
    end
    D = Descriptor(fill(2, 6), 2, [1, 1], 1)
    qf.Kn1 = 1.2 + params(D)[1]
    d.L = 0.1 + params(D)[2]
    full = twiss(ring; GTPSA_descriptor=D)
    few = twiss(ring; GTPSA_descriptor=D, at=[40])
    qf.Kn1 = 1.2
    d.L = 0.1
    scalarize!(ring)
    coefs(t) = [t[i] for i in 0:GTPSA.numcoefs(t)-1]
    @test coefs(few.phi1[40]) == coefs(full.phi1[40]) && scalar(few.phi1[40]) > 0.5
    @test coefs(few.q1.taylor_series) == coefs(full.q1.taylor_series)
  end

  @testset "Twiss of a part of the beamline (start, stop)" begin
      # Without a_initial: the periodic Twiss of the part by itself (the middle cell of three
      # equals the cell alone). With a_initial: the open line from start (equal to the rows of
      # the open Twiss of the whole line, started with its a at start). s and index are those of
      # the whole beamline, the last row is the beginning of stop
    @elements begin
      qf = Quadrupole(Kn1=1.2, L=0.5)
      sf = Sextupole(Kn2=1.2, L=0.2)
      d = Drift(L=0.1)
      b = SBend(L=1.2, angle=pi/132, n_steps=3)
      qd = Quadrupole(Kn1=-1.1, L=0.5)
      q3 = Quadrupole(Kn1=0.3, L=0.4)
      q4 = Quadrupole(Kn1=-0.25, L=0.4)
    end
    cellp = [qf, sf, b, d, qd, b, d]
    n = length(cellp)
    cell = Beamline(cellp; species_ref=Species("electron"), pc_ref=18e9)
    ring = Beamline([repeat(cellp, 3)..., q3, d, q4, d]; species_ref=Species("electron"),
                    pc_ref=18e9)
    tc = twiss(cell)
    tp = twiss(ring; start=n+1, stop=2n+1)
    @test tp.q1 ≈ tc.q1 rtol=1e-13
    @test tp.q2 ≈ tc.q2 rtol=1e-13
    @test tp.df.beta1 ≈ tc.df.beta1 rtol=1e-12
    @test tp.df.dx ≈ tc.df.dx rtol=1e-11
    @test tp.df.phi2 ≈ tc.df.phi2 rtol=1e-12 atol=1e-13
    @test tp.df.index[1] == n+1 && tp.df.index[end] == 2n+1 && tp.df.name[end] == "qf"
    @test tp.df.s ≈ tc.df.s .+ ring.line[n+1].s
    @test tp.beta1[ring.line[n+5]] == tp.beta1[n+5] == tp.df.beta1[findfirst(==(n+5), tp.df.index)]
      # Locations of several values give those in the part, an element is the one placement in
      # the part (qd and qf are placed three times in the ring; qf at the beginning and at the end
      # of the part), a beamline index outside the part is an error, and s is that of the whole ring
    tr = twiss(Beamline(repeat(cellp, 3); species_ref=Species("electron"), pc_ref=18e9))
    @test tp.beta1[:] == tp.beta1[(0.0, Inf)] == tp.df.beta1
    @test [tp.beta1[qd]] == tp.beta1[[qd]] == tp.beta1["qd"] == [tp.beta1[n+5]] &&
          tp.beta1[[qf]] == [tp.beta1[begin], tp.beta1[end]]
    @test_throws ErrorException tp.beta1[qf]
    @test occursin("placed 2 times in the part of the beamline from element $(n+1) to element " *
                   "$(2n+1)", sprint(showerror, try tp.beta1[qf] catch e; e end))
    @test tp.df.s == tr.df.s[findfirst(==(n+1), tr.df.index):findfirst(==(2n+1), tr.df.index)]
    @test_throws ErrorException tp.beta1[1]
    @test_throws ErrorException tp.beta1[end, :exit] # (the exit of `stop`, outside the part)
    @test twiss(ring; start=n+1, stop=2n+1, at=[qd, 1, "sf", (0.0, 0.6)]).df.index == [n+2, n+5]
    @test twiss(ring; start=n+1, stop=2n+1, at=[qf]).df.index == [n+1, 2n+1]
    tep = TwissExpr(ring; start=n+1, stop=2n+1)
    @test value(tep.beta1[:]) ≈ tp.df.beta1 rtol=1e-13
    @test value(tep.beta1[qd]) ≈ tp.beta1[qd] rtol=1e-13
    @test value(tep.beta1[[qf]]) ≈ tp.beta1[[qf]] rtol=1e-13
    @test tp.beta1[begin] == tp.df.beta1[1] && tp.beta1[end] == tp.df.beta1[end]
    ts = twiss(ring; start=n+2, stop=2n+2) # the cell started one element later: the same tunes
    @test ts.q1 ≈ tc.q1 rtol=1e-12
    @test !isapprox(ts.df.beta1[1], tc.df.beta1[1]; rtol=1e-3)
    @test ts.df.beta1[1] ≈ tc.df.beta1[2] rtol=1e-11
    a0 = twiss(cell; cols=["a"], as_taylor_series=true, at=[1]).a[1]
    full = twiss(ring; a_initial=a0, cols=["a", "beta1", "alpha2", "dx", "phi1"],
                 as_taylor_series=true)
    i1, i2 = 3n+1, 3n+4
    fu = full.df
    j1 = findfirst(==(i1), fu.index)
    j2 = findfirst(==(i2), fu.index)
    to = twiss(ring; a_initial=full.a[i1], start=i1, stop=ring.line[i2])
    @test to.df.index[1] == i1 && to.df.index[end] == i2
    @test to.df.beta1 ≈ scalar.(fu.beta1[j1:j2]) rtol=1e-11
    @test to.df.alpha2 ≈ scalar.(fu.alpha2[j1:j2]) rtol=1e-10
    @test to.df.dx ≈ scalar.(fu.dx[j1:j2]) rtol=1e-10
    @test to.df.phi1 ≈ scalar.(fu.phi1[j1:j2]) .- scalar(fu.phi1[j1]) rtol=1e-10 atol=1e-12
    @test to.beta1[ring.line[i2]] == to.beta1[end]
    @test twiss(ring; a_initial=full.a[i1], start=i1, stop=i2, at=[i1+1, i2, q4]).df.index ==
          [i1+1, i2-1, i2]
    te = twiss(ring; a_initial=full.a[i1], start=i1) # to the end of the beamline
    @test te.df.index[end] == -1
    @test te.beta1[end] ≈ scalar(full.beta1[end]) rtol=1e-11
    @test_throws ErrorException twiss(ring; start=5, stop=3)
    @test_throws ErrorException twiss(ring; start=qf) # placed several times
      # As expressions: [end] and [begin] are those of the part
    tw = TwissExpr(ring; a_initial=full.a[i1], start=i1, stop=i2)
    e = [tw.beta1[end], tw.alpha1[end], tw.beta2[q4], tw.phi1[end] - tw.phi1[begin]]
    v, J = value_and_jacobian(e, [@knob(q3.Kn1), @knob(q4.Kn1)])
    @test v[1:2] ≈ [to.beta1[end], to.alpha1[end]] rtol=1e-12
    fd_part = zeros(4, 2)
    for (k, q) in enumerate((q3, q4))
      q.Kn1 += 1e-6
      vp = value(e)
      q.Kn1 -= 2e-6
      fd_part[:, k] = (vp .- value(e)) ./ 2e-6
      q.Kn1 += 1e-6
    end
    @test all(abs.(J .- fd_part) .<= 1e-6 .* (abs.(fd_part) .+ 1e-3))
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
      return track(ring; v0=copy(co)).data.v[1,5,1,end] - co[1,5]
    end
    rel_dt(δ) = -dz(δ)/beta(δ)*beta(0.0)/L
    rel_dl(δ) = -dz(δ)/L + beta(δ)/beta(0.0) - 1
    h, deg = 2e-2, 14
    δs = range(-h, h, length=25)
    V = [(δ/h)^k for δ in δs, k in 0:deg]
    for (f, tps) in ((rel_dt, t4.etac), (rel_dl, t4.alphac))
      c = (V \ f.(δs)) ./ h .^ (0:deg) # c[k+1] is the coefficient of δ^k of the fit
      for (k, rtol) in zip(0:3, (1e-9, 1e-7, 1e-5, 1e-3))
        @test getterm(tps, delta=k) ≈ (k + 1)*c[k+2] rtol=rtol # (the coefficient of δ^k)
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
    @test t3.df.index == [n] && t3.s[n] == t2.df.s[findfirst(==(n), t2.df.index)]
    # twiss does not mutate the user's initial guess
    g = [1e-3 0 0 0 0 0.0]; g0 = copy(g)
    twiss(cell(); v0=g, at=[])
    @test g == g0
    # track accepts a weight vector
    @test track(cell(); v0=zeros(3, 6), weight=ones(3)) isa TrackResult
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
