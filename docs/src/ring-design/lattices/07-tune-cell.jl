# Ring design tutorial, chapter 7: tune cell
#
# The straight section at 2 o'clock (between the sextants 1 and 3) gets its own quadrupoles,
# which set the tunes to 54.08 and 54.14. Its upstream half is matched from the periodic
# lattice functions of the arc into the periodic lattice functions of its own FODO cells; by
# mirror symmetry, its downstream half then matches back into the arc.

include("06-interaction-region.jl")

@elements begin
  qfss_2 = Quadrupole(L=0.5, Kn1=qfss.Kn1)
  qdss_2 = Quadrupole(L=0.5, Kn1=qdss.Kn1)
  qff2_2 = Quadrupole(L=0.5, Kn1=qff2.Kn1)
  qdf2_2 = Quadrupole(L=0.5, Kn1=qdf2.Kn1)
  qff3_2 = Quadrupole(L=0.5, Kn1=qff3.Kn1)
  qdf3_2 = Quadrupole(L=0.5, Kn1=qdf3.Kn1)
end

fodossf_2 = [qfss_2, d1, db, d2, qdss_2, d1, db, d2]
fodossr_2 = [qfss_2, d2, db, d1, qdss_2, d2, db, d1]
mssf_2 = [qff2_2, d1, db, d2, qdf2_2, d1, db, d2, qff3_2, d1, db, d2, qdf3_2, d1, db, d2]
mdcr_2 = [qfss_2, d2, db, d1, qdf3_2, d2, db, d1, qff3_2, d2, db, d1, qdf2_2, d2, db, d1]
dispcrer_2 = [qff2_2, d2, bh, d1, qdf1, d2, bh, d1, qff1, d2, bh, d1, qd, d2, bh, d1]
arc_to_ssf_2 = [dispsupf; mssf_2]
ss_to_arcr_2 = [mdcr_2; dispcrer_2]

sextant1 = [repeat(fodossf, 4); ss_to_arcf; repeat(fodoaf, 20); arc_to_ssf_2;
            repeat(fodossf_2, 4)]
sextant3 = [repeat(fodossr_2, 4); ss_to_arcr_2; repeat(fodoar, 20); arc_to_ssr;
            repeat(fodossr, 4)]

ring = Beamline([sextant1; sextant3; sextant5; sextant7; sextant9; sextant11],
                species_ref=electron, pc_ref=pc_ref)

cell_ss2 = Beamline(fodossf_2, species_ref=electron, pc_ref=pc_ref)
line_2 = Beamline(arc_to_ssf_2, species_ref=electron, pc_ref=pc_ref)

tw_ss2 = TwissExpr(cell_ss2)
tw_2 = TwissExpr(line_2, a_initial=tw_af.a[end])
tw_ring = TwissExpr(ring)

match!(vary=[@knob(qfss_2.Kn1), @knob(qdss_2.Kn1), @knob(qff2_2.Kn1), @knob(qdf2_2.Kn1),
             @knob(qff3_2.Kn1), @knob(qdf3_2.Kn1)],
       targets=[tw_2.beta1[end] == tw_ss2.beta1[begin], tw_2.alpha1[end] == tw_ss2.alpha1[begin],
                tw_2.beta2[end] == tw_ss2.beta2[begin], tw_2.alpha2[end] == tw_ss2.alpha2[begin],
                tw_ring.q1 == 54.08, tw_ring.q2 == 54.14],
       solver=:gauss_newton)
