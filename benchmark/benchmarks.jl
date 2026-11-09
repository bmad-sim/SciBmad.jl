# Benchmarks of twiss and find_closed_orbit. Absolute timings depend on the machine, so compare
# revisions on the same machine, e.g. with AirspeedVelocity.jl (see .github/workflows/Benchmark.yml):
#
#   benchpkg SciBmad --rev=main,dirty --script=benchmark/benchmarks.jl
#
# or locally with `include("benchmark/benchmarks.jl"); run(SUITE)`. The script runs on both
# revisions, so it only uses API that exists in both.
using SciBmad, BenchmarkTools

const SUITE = BenchmarkGroup()

# esr.jl defines the global `ring` itself, so the constant has another name
const esr = include(joinpath(@__DIR__, "..", "test", "lattices", "esr.jl"))

@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-1.2, L=0.2)
end
const fodo = Beamline([qf, sf, b, d, qd, sd, b, d], species_ref=Species("electron"), pc_ref=18e9)

# Parametric normal form: tunes and beta functions as functions of the quadrupole strengths
@elements begin
  pqf = Quadrupole(Kn1=DefExpr(c -> c.kqf), L=0.5)
  pqd = Quadrupole(Kn1=DefExpr(c -> c.kqd), L=0.5)
end
const pfodo = Beamline([pqf, sf, b, d, pqd, sd, b, d], species_ref=Species("electron"), pc_ref=18e9)
const pdesc = Descriptor([1, 1, 1, 1, 1, 1], 2, [1, 1], 1)
pfodo.context.kqf = 0.36 + GTPSA.params(pdesc)[1]
pfodo.context.kqd = -0.36 + GTPSA.params(pdesc)[2]

SUITE["twiss"] = BenchmarkGroup()
SUITE["twiss"]["esr"] = @benchmarkable twiss($esr)
SUITE["twiss"]["esr chrom=2"] = @benchmarkable twiss($esr; rf_on=false, chrom=2, cols=["beta1", "dx", "dbeta1"])
SUITE["twiss"]["fodo order=2"] = @benchmarkable twiss($fodo; order=2, cols=["beta1", "h3000", "h2100", "h1011"])
SUITE["twiss"]["fodo spin order=2"] = @benchmarkable twiss($fodo; spin=true, order=2, cols=["beta1", "nx", "ny", "nz"], as_taylor_series=true)
SUITE["twiss"]["fodo parametric"] = @benchmarkable twiss($pfodo; GTPSA_descriptor=$pdesc)

SUITE["find_closed_orbit"] = BenchmarkGroup()
SUITE["find_closed_orbit"]["esr"] = @benchmarkable find_closed_orbit($esr)
