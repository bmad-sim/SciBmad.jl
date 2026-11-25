# Ring design tutorial, chapter 5: constructing the ring
#
# By the mirror symmetry of the ring, the lines from the straight sections to the arcs use the
# quadrupoles of the lines from the arcs to the straight sections of the other kind of sextant.
# No matching is needed.

from .ch03_match_to_straight import *

# Forward sextants: match the straight section to the dispersion creator, and create the
# dispersion (the reverse quadrupoles, in reverse order)
mdcf = [qfss, d1, db, d2, qdr3, d1, db, d2, qfr3, d1, db, d2, qdr2, d1, db, d2]
dispcref = [qfr2, d1, bh, d2, qdr1, d1, bh, d2, qfr1, d1, bh, d2, qd, d1, bh, d2]
ss_to_arcf = mdcf + dispcref

# Reverse sextants (the forward quadrupoles, in reverse order)
mdcr = [qfss, d2, db, d1, qdf3, d2, db, d1, qff3, d2, db, d1, qdf2, d2, db, d1]
dispcrer = [qff2, d2, bh, d1, qdf1, d2, bh, d1, qff1, d2, bh, d1, qd, d2, bh, d1]
ss_to_arcr = mdcr + dispcrer

sextant1 = fodossf*4 + ss_to_arcf + fodoaf*20 + arc_to_ssf + fodossf*4
sextant3 = fodossr*4 + ss_to_arcr + fodoar*20 + arc_to_ssr + fodossr*4
sextant5 = fodossf*4 + ss_to_arcf + fodoaf*20 + arc_to_ssf + fodossf*4
sextant7 = fodossr*4 + ss_to_arcr + fodoar*20 + arc_to_ssr + fodossr*4
sextant9 = fodossf*4 + ss_to_arcf + fodoaf*20 + arc_to_ssf + fodossf*4
sextant11 = fodossr*4 + ss_to_arcr + fodoar*20 + arc_to_ssr + fodossr*4

ring = sb.Beamline(sextant1 + sextant3 + sextant5 + sextant7 + sextant9 + sextant11,
                   species_ref=electron, pc_ref=pc_ref)
