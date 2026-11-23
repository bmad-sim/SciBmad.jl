# Ring design tutorial, chapter 12: sextupoles and chromaticity correction
#
# Two families of sextupoles per plane in each arc, and harmonic sextupoles (turned off) in the
# straight sections. Their strengths are deferred expressions of variables in the context of
# the ring: `k2f` and `k2d` for all focusing and defocusing arc sextupoles, and the vectors
# `xf` and `xd` that make the two families of each arc stronger and weaker by the same amount.
# Phase trombones next to the interaction region and at the start of the ring change the phase
# advances between the arcs without changing the tunes.

include("09-rf.jl")

@elements begin
  d2s1 = Drift(L=0.501)
  d2s2 = Drift(L=0.156)
  sfss = Sextupole(L=0.584, Kn2=0.0)
  sdss = Sextupole(L=0.584, Kn2=0.0)
  tr_m = Marker()
  tr_f = Marker()
  tr_r = Marker()
  end_7 = Marker()
end

fodossf = [qfss, d1, db, d2s1, sdss, d2s2, qdss, d1, db, d2s1, sfss, d2s2]
fodossr = [qfss, d2s2, sfss, d2s1, db, d1, qdss, d2s2, sdss, d2s1, db, d1]

# The arcs: arc i is that of sextant 2i-1. Each has two families of focusing (sf1, sf2) and
# defocusing (sd1, sd2) sextupoles, which alternate from one FODO cell to the next.
arcs = Vector{LatElement}[]
for i in 1:6
  sf1 = Sextupole(name="sf1_$(2i-1)", L=0.584, Kn2=DefExpr(c -> c.k2f + c.xf[i]))
  sf2 = Sextupole(name="sf2_$(2i-1)", L=0.584, Kn2=DefExpr(c -> c.k2f - c.xf[i]))
  sd1 = Sextupole(name="sd1_$(2i-1)", L=0.584, Kn2=DefExpr(c -> c.k2d + c.xd[i]))
  sd2 = Sextupole(name="sd2_$(2i-1)", L=0.584, Kn2=DefExpr(c -> c.k2d - c.xd[i]))
  if isodd(i) # Forward sextants 1, 5, 9
    cells = [qf, d1, b, d2s1, sd1, d2s2, qd, d1, b, d2s1, sf1, d2s2,
             qf, d1, b, d2s1, sd2, d2s2, qd, d1, b, d2s1, sf2, d2s2]
  else        # Reverse sextants 3, 7, 11
    cells = [qf, d2s2, sf1, d2s1, b, d1, qd, d2s2, sd1, d2s1, b, d1,
             qf, d2s2, sf2, d2s1, b, d1, qd, d2s2, sd2, d2s1, b, d1]
  end
  push!(arcs, repeat(cells, 10))
end

sextant1 = [tr_m; repeat(fodossf, 4); ss_to_arcf; arcs[1]; arc_to_ssf_2; repeat(fodossf_2, 4)]
sextant3 = [repeat(fodossr_2, 4); ss_to_arcr_2; arcs[2]; arc_to_ssr; repeat(fodossr, 4)]
sextant5 = [repeat(fodossf, 4); ss_to_arcf; arcs[3]; arc_to_ssf; fodossf; tr_f; ipf]
sextant7 = [ipr; tr_r; fodossr; ss_to_arcr; arcs[4]; end_7; arc_to_ssr; repeat(fodossr, 4)]
sextant9 = [repeat(fodossf, 4); ss_to_arcf; arcs[5]; arc_to_ssf; repeat(fodossf, 2);
            repeat(fodorf, 2)]
sextant11 = [repeat(fodorf, 2); repeat(fodossr, 2); ss_to_arcr; arcs[6]; arc_to_ssr;
             repeat(fodossr, 4)]

ring = Beamline([sextant1; sextant3; sextant5; sextant7; sextant9; sextant11],
                species_ref=electron, pc_ref=pc_ref)

ring.context.k2f = 1.0
ring.context.k2d = -2.0
ring.context.xf = zeros(6)
ring.context.xd = zeros(6)

# Chromaticities of +1, without synchrotron motion
tw_chrom = TwissExpr(ring, rf_on=false)
match!(vary=[@knob(ring.context.k2f), @knob(ring.context.k2d)],
       targets=[tw_chrom.dq1 == 1, tw_chrom.dq2 == 1])

# Phase trombones: tr_f and tr_r change the phase advances on both sides of the interaction
# region, and tr_m takes their sum off again, so that the tunes stay the same. They are built
# from the optics without synchrotron motion, in which the chromatic functions are matched.
ring.context.x1 = 0.0
ring.context.x2 = 0.0
ring.context.y1 = 0.0
ring.context.y2 = 0.0
phase_trombone!(ring[tr_f]; phi1=DefExpr(c -> c.x1), phi2=DefExpr(c -> c.y1), rf_on=false)
phase_trombone!(ring[tr_r]; phi1=DefExpr(c -> c.x2), phi2=DefExpr(c -> c.y2), rf_on=false)
phase_trombone!(ring[tr_m]; phi1=DefExpr(c -> -c.x1 - c.x2), phi2=DefExpr(c -> -c.y1 - c.y2),
                rf_on=false)

# The chromatic beta beat: W = 0 at the IP and at the end of arc 7. W = √(A² + B²) is zero
# when both of its components are.
match!(vary=[@knob(ring.context.xf[3]), @knob(ring.context.xd[3]),
             @knob(ring.context.xf[4]), @knob(ring.context.xd[4]),
             @knob(ring.context.x1), @knob(ring.context.x2),
             @knob(ring.context.y1), @knob(ring.context.y2)],
       targets=[tw_chrom.w1a[ip6] == 0, tw_chrom.w1b[ip6] == 0,
                tw_chrom.w2a[ip6] == 0, tw_chrom.w2b[ip6] == 0,
                tw_chrom.w1a[end_7] == 0, tw_chrom.w1b[end_7] == 0,
                tw_chrom.w2a[end_7] == 0, tw_chrom.w2b[end_7] == 0])
