# Ring design tutorial, chapter 5: constructing the ring
#
# By the mirror symmetry of the ring, the lines from the straight sections to the arcs use the
# quadrupoles of the lines from the arcs to the straight sections of the other kind of sextant.
# No matching is needed.

include("03-match-to-straight.jl")

# Forward sextants: match the straight section to the dispersion creator, and create the
# dispersion (the reverse quadrupoles, in reverse order)
mdcf = [qfss, d1, db, d2, qdr3, d1, db, d2, qfr3, d1, db, d2, qdr2, d1, db, d2]
dispcref = [qfr2, d1, bh, d2, qdr1, d1, bh, d2, qfr1, d1, bh, d2, qd, d1, bh, d2]
ss_to_arcf = [mdcf; dispcref]

# Reverse sextants (the forward quadrupoles, in reverse order)
mdcr = [qfss, d2, db, d1, qdf3, d2, db, d1, qff3, d2, db, d1, qdf2, d2, db, d1]
dispcrer = [qff2, d2, bh, d1, qdf1, d2, bh, d1, qff1, d2, bh, d1, qd, d2, bh, d1]
ss_to_arcr = [mdcr; dispcrer]

sextant1 = [repeat(fodossf, 4); ss_to_arcf; repeat(fodoaf, 20); arc_to_ssf; repeat(fodossf, 4)]
sextant3 = [repeat(fodossr, 4); ss_to_arcr; repeat(fodoar, 20); arc_to_ssr; repeat(fodossr, 4)]
sextant5 = [repeat(fodossf, 4); ss_to_arcf; repeat(fodoaf, 20); arc_to_ssf; repeat(fodossf, 4)]
sextant7 = [repeat(fodossr, 4); ss_to_arcr; repeat(fodoar, 20); arc_to_ssr; repeat(fodossr, 4)]
sextant9 = [repeat(fodossf, 4); ss_to_arcf; repeat(fodoaf, 20); arc_to_ssf; repeat(fodossf, 4)]
sextant11 = [repeat(fodossr, 4); ss_to_arcr; repeat(fodoar, 20); arc_to_ssr; repeat(fodossr, 4)]

ring = Beamline([sextant1; sextant3; sextant5; sextant7; sextant9; sextant11],
                species_ref=electron, pc_ref=pc_ref)
