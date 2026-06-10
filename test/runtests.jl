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
        @test length(tw.table.beta_1) == N + 1

        # Beta functions: element-wise relative tolerance (beta ranges 0.59–702 m)
        @test all(isapprox.(tw.table.beta_1[1:N], ref[:,1]; rtol=1e-5, atol=0))
        @test all(isapprox.(tw.table.beta_2[1:N], ref[:,2]; rtol=1e-5, atol=0))

        # Alpha functions: atol covers elements where alpha ≈ 0 at the periodic solution
        @test all(isapprox.(tw.table.alpha_1[1:N], ref[:,3]; rtol=1e-5, atol=1e-8))
        @test all(isapprox.(tw.table.alpha_2[1:N], ref[:,4]; rtol=1e-5, atol=1e-8))

        # Ring closure: END OF BEAMLINE entry equals the start of the ring.
        # Alpha uses atol since it is ≈0 at the periodic solution (pure relative tol is meaningless).
        @test tw.table.beta_1[end]  ≈ tw.table.beta_1[1]
        @test tw.table.beta_2[end]  ≈ tw.table.beta_2[1]
        @test isapprox(tw.table.alpha_1[end], tw.table.alpha_1[1]; atol=1e-8)
        @test isapprox(tw.table.alpha_2[end], tw.table.alpha_2[1]; atol=1e-8)
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
        @test all(isapprox.(tw.table.c11[1:N], ref[:,1]; rtol=1e-5, atol=1e-8))
        @test all(isapprox.(tw.table.c12[1:N], ref[:,2]; rtol=1e-5, atol=1e-8))
        @test all(isapprox.(tw.table.c21[1:N], ref[:,3]; rtol=1e-5, atol=1e-8))
        @test all(isapprox.(tw.table.c22[1:N], ref[:,4]; rtol=1e-5, atol=1e-8))
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
        tw = twiss(ring; GTPSA_descriptor=Descriptor(6, 2))

        N = size(ref, 1)

        # Extract linear pz coefficient = ∂/∂pz evaluated at closed orbit
        # Monomial [0,0,0,0,0,1] = pz^1 (variables: x,px,y,py,z,pz)
        dbeta1  = [tw.table.beta_1[i][[0,0,0,0,0,1]]  for i in 1:N]
        dbeta2  = [tw.table.beta_2[i][[0,0,0,0,0,1]]  for i in 1:N]
        dalpha1 = [tw.table.alpha_1[i][[0,0,0,0,0,1]] for i in 1:N]
        dalpha2 = [tw.table.alpha_2[i][[0,0,0,0,0,1]] for i in 1:N]

        # Chromatic functions range up to ~5000 m; atol covers near-zero cases
        @test_broken all(isapprox.(dbeta1,  ref[:,1]; rtol=1e-4, atol=1e-5))
        @test_broken all(isapprox.(dbeta2,  ref[:,2]; rtol=1e-4, atol=1e-5))
        @test_broken all(isapprox.(dalpha1, ref[:,3]; rtol=1e-4, atol=1e-5))
        @test_broken all(isapprox.(dalpha2, ref[:,4]; rtol=1e-4, atol=1e-5))
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
