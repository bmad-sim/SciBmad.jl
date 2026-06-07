"""
Diagnostic script: compare SciBmad chromatic functions vs Tao/Bmad.

Run with:
  julia ~/.julia/dev/SciBmad/test/investigate_chromatic.jl

Shows dbeta_dpz_a (∂β₁/∂pz) for the first 20 elements side-by-side:
  - "s"         : position of element beginning (meters)
  - "SciBmad"   : tw.table.beta_1[i][[0,0,0,0,0,1]] with Descriptor(6,2)
  - "Tao"       : dbeta_dpz_a from Tao show lat -pipe -beginning
  - "Tao/SciBmad": ratio (should be ≈1 everywhere if the codes agree)

Also prints the Tao raw output directly so you can see where any jump occurs.
"""

using SciBmad, GTPSA, DelimitedFiles, Printf

lat_jl   = joinpath(@__DIR__, "lattices", "esr.jl")
ref_file = joinpath(@__DIR__, "reference", "esr-chromatic-tao.csv")

# ── SciBmad ──────────────────────────────────────────────────────────────────
ring = include(lat_jl)
tw   = twiss(ring; GTPSA_descriptor=Descriptor(6, 2))
ref  = readdlm(ref_file, ',', Float64; skipstart=1)
N    = size(ref, 1)

println("\n=== SciBmad coasting_beam = $(tw.coasting_beam) ===\n")

# s-coordinate for each element (position of its beginning)
s_vals = tw.table.s

n_show = 20
println("First $n_show elements — dbeta_dpz_a (∂β₁/∂pz):")
println("  idx │ name              │        s (m) │      SciBmad │          Tao │  Tao/SciBmad")
println("  ────┼───────────────────┼──────────────┼──────────────┼──────────────┼─────────────")
for i in 1:min(n_show, N)
    sc  = tw.table.beta_1[i][[0,0,0,0,0,1]]
    ta  = ref[i, 1]
    rat = iszero(sc) ? NaN : ta / sc
    nm  = rpad(tw.table.name[i], 18)
    @printf("  %3d │ %s│ %12.4f │ %12.4g │ %12.4g │ %12.4g\n",
            i, nm, s_vals[i], sc, ta, rat)
end

# ── Also show last 5 elements near ring end ──────────────────────────────────
println("\nLast 5 elements (near end/closure):")
println("  idx │ name              │        s (m) │      SciBmad │          Tao │  Tao/SciBmad")
println("  ────┼───────────────────┼──────────────┼──────────────┼──────────────┼─────────────")
for i in (N-4):N
    sc  = tw.table.beta_1[i][[0,0,0,0,0,1]]
    ta  = ref[i, 1]
    rat = iszero(sc) ? NaN : ta / sc
    nm  = rpad(tw.table.name[i], 18)
    @printf("  %3d │ %s│ %12.4f │ %12.4g │ %12.4g │ %12.4g\n",
            i, nm, s_vals[i], sc, ta, rat)
end

# ── Check smoothness of each dataset independently ────────────────────────────
println("\n=== Continuity check at s=0 (elements 1,2 — should match if both at s=0) ===")
@printf("  SciBmad elem 1 (s=%.3f): %g\n", s_vals[1], tw.table.beta_1[1][[0,0,0,0,0,1]])
@printf("  SciBmad elem 2 (s=%.3f): %g\n", s_vals[2], tw.table.beta_1[2][[0,0,0,0,0,1]])
@printf("  Tao     row  1:          %g\n", ref[1, 1])
@printf("  Tao     row  2:          %g\n", ref[2, 1])
println("  → SciBmad jump at s=0: $(tw.table.beta_1[1][[0,0,0,0,0,1]] == tw.table.beta_1[2][[0,0,0,0,0,1]] ? "NONE (smooth ✓)" : "JUMP ✗")")
println("  → Tao    jump at s=0: $(abs(ref[2,1]/ref[1,1]) > 10 ? "JUMP ✗ (|ratio|=$(round(abs(ref[2,1]/ref[1,1]),digits=1)))" : "smooth ✓")")

# ── Check if Tao is 1-element shifted vs SciBmad ─────────────────────────────
println("\n=== Shift test: does Tao row i match SciBmad elem i+1? ===")
println("  (If ratio ≈ 1 here, Tao is reporting exit instead of beginning for chromatic)")
println("  idx │        s_SciBmad │  SciBmad[i+1] │     Tao[i] │     ratio")
println("  ────┼──────────────────┼───────────────┼────────────┼──────────")
for i in 2:min(10, N-1)
    sc_next = tw.table.beta_1[i+1][[0,0,0,0,0,1]]
    ta_cur  = ref[i, 1]
    rat     = iszero(sc_next) ? NaN : ta_cur / sc_next
    @printf("  %3d │ %16.4f │ %13.4g │ %10.4g │ %8.4g\n",
            i, s_vals[i+1], sc_next, ta_cur, rat)
end
