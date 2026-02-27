using SciBmad
using Test

@testset "SciBmad.jl" begin
    # See SciBmad's subpackages for all of the unit tests,
    # Those are where most of the functionalities are, hence
    # where most of the tests are. 
    
    # Here we have some regression tests for SciBmad-level functionality
    ring = include("lattices/esr-test.jl")
    foreach(x->x.tracking_method=Yoshida(order=4,num_steps=10), ring.line)
    
    co_info = find_closed_orbit(ring)
    @test co_info[2] == true # coast
    @test all(co_info[1] .< 1e-15)
    ring.line[70].Kn0L = -1e-5

    co_info = find_closed_orbit(ring)
    @test co_info[2] == true
    @test co_info[1] ≈ [-7.492687925013916e-5 0.00020547121555142397 7.89574520440494e-6 -1.9195971020399645e-5 0.0 0.0]

    tw = twiss(ring, at=[])
    @test tw.coasting_beam == true
    @test scalar.(tw.tunes[1:2]) ≈ [7.9940856425670226E-02, 1.3974282747099287E-01]
    @test tw.tunes[3][6] ≈ -2.3157101992553932E+00

    tw = twiss(ring)
    @test tw.table.beta_1[1]  ≈ 0.5864997400581597
    @test tw.table.beta_1[1]  ≈ tw.table.beta_1[end]
    @test tw.table.beta_2[1]  ≈ 0.05689366916498126
    @test tw.table.beta_2[1]  ≈ tw.table.beta_2[end]
    @test tw.table.alpha_1[1] ≈ -0.004857800200759143
    @test tw.table.alpha_1[1] ≈ tw.table.alpha_1[end]
    @test tw.table.alpha_2[1] ≈ 0.000257627493904479
    @test tw.table.alpha_2[1] ≈ tw.table.alpha_2[end]
    @test scalar(tw.table.phi_1[end]) ≈ 48.07994085642562
    @test scalar(tw.table.phi_2[end]) ≈ 44.139742827471125
    @test tw.table.orbit_x[1]  ≈ co_info[1][1]
    @test tw.table.orbit_px[1] ≈ co_info[1][2]
    @test tw.table.orbit_y[1]  ≈ co_info[1][3]
    @test tw.table.orbit_py[1] ≈ co_info[1][4]
    @test tw.table.orbit_z[1]  ≈ co_info[1][5]
    @test tw.table.orbit_pz[1] ≈ co_info[1][6]
end
