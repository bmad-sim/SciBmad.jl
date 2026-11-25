# Ring design tutorial, chapter 7: tune cell
#
# The straight section at 2 o'clock (between the sextants 1 and 3) gets its own quadrupoles,
# which set the tunes to 54.08 and 54.14. Its upstream half is matched from the periodic
# lattice functions of the arc into the periodic lattice functions of its own FODO cells; by
# mirror symmetry, its downstream half then matches back into the arc.

from .ch06_interaction_region import *

with sb.elements():
    qfss_2 = sb.Quadrupole(L=0.5, Kn1=qfss.Kn1)
    qdss_2 = sb.Quadrupole(L=0.5, Kn1=qdss.Kn1)
    qff2_2 = sb.Quadrupole(L=0.5, Kn1=qff2.Kn1)
    qdf2_2 = sb.Quadrupole(L=0.5, Kn1=qdf2.Kn1)
    qff3_2 = sb.Quadrupole(L=0.5, Kn1=qff3.Kn1)
    qdf3_2 = sb.Quadrupole(L=0.5, Kn1=qdf3.Kn1)

fodossf_2 = [qfss_2, d1, db, d2, qdss_2, d1, db, d2]
fodossr_2 = [qfss_2, d2, db, d1, qdss_2, d2, db, d1]
mssf_2 = [qff2_2, d1, db, d2, qdf2_2, d1, db, d2, qff3_2, d1, db, d2, qdf3_2, d1, db, d2]
mdcr_2 = [qfss_2, d2, db, d1, qdf3_2, d2, db, d1, qff3_2, d2, db, d1, qdf2_2, d2, db, d1]
dispcrer_2 = [qff2_2, d2, bh, d1, qdf1, d2, bh, d1, qff1, d2, bh, d1, qd, d2, bh, d1]
arc_to_ssf_2 = dispsupf + mssf_2
ss_to_arcr_2 = mdcr_2 + dispcrer_2

sextant1 = fodossf*4 + ss_to_arcf + fodoaf*20 + arc_to_ssf_2 + fodossf_2*4
sextant3 = fodossr_2*4 + ss_to_arcr_2 + fodoar*20 + arc_to_ssr + fodossr*4

ring = sb.Beamline(sextant1 + sextant3 + sextant5 + sextant7 + sextant9 + sextant11,
                   species_ref=electron, pc_ref=pc_ref)

cell_ss2 = sb.Beamline(fodossf_2, species_ref=electron, pc_ref=pc_ref)
line_2 = sb.Beamline(arc_to_ssf_2, species_ref=electron, pc_ref=pc_ref)

tw_ss2 = sb.TwissExpr(cell_ss2)
tw_2 = sb.TwissExpr(line_2, a_initial=tw_af.a[-1])
tw_ring = sb.TwissExpr(ring)

sb.match(vary=[sb.Knob(q, "Kn1") for q in (qfss_2, qdss_2, qff2_2, qdf2_2, qff3_2, qdf3_2)],
         targets=[tw_2.beta1[-1] == tw_ss2.beta1[sb.begin],
                  tw_2.alpha1[-1] == tw_ss2.alpha1[sb.begin],
                  tw_2.beta2[-1] == tw_ss2.beta2[sb.begin],
                  tw_2.alpha2[-1] == tw_ss2.alpha2[sb.begin],
                  tw_ring.q1 == 54.08, tw_ring.q2 == 54.14])
