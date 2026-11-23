# Ring design tutorial, chapter 6: low-beta interaction region
#
# An interaction region with β* = 0.6 m and 0.06 m at IP6, in the straight section between the
# sextants 5 and 7. The upstream half IPF starts with the periodic lattice functions of the
# straight section, and the downstream half IPR with the lattice functions that IPF gives at
# the IP. IPR then ends with the periodic lattice functions of the straight section.

include("05-ring.jl")

@elements begin
  qef1 = Quadrupole(L=0.5, Kn1=k_ss)
  qef2 = Quadrupole(L=0.5, Kn1=-k_ss)
  def1 = Drift(L=20.46)
  qef3 = Quadrupole(L=1.6, Kn1=k_ss)
  def2 = Drift(L=3.76)
  qef4 = Quadrupole(L=1.2, Kn1=-k_ss)
  def3 = Drift(L=5.8)
  ip6 = Marker()
  der3 = Drift(L=5.3)
  qer4 = Quadrupole(L=1.8, Kn1=-k_ss)
  der2 = Drift(L=0.5)
  qer3 = Quadrupole(L=1.4, Kn1=k_ss)
  der1 = Drift(L=23.82)
  qer2 = Quadrupole(L=0.5, Kn1=k_ss)
  qer1 = Quadrupole(L=0.5, Kn1=-k_ss)
end

ipf = [qef1, d1, db, d2, qef2, d1, db, d2, def1, qef3, def2, qef4, def3, ip6]
ipr = [der3, qer4, der2, qer3, der1, qer2, d2, db, d1, qer1, d2, db, d1]

ir_f = Beamline(ipf, species_ref=electron, pc_ref=pc_ref)
ir_r = Beamline(ipr, species_ref=electron, pc_ref=pc_ref)

tw_ipf = TwissExpr(ir_f, a_initial=tw_ssf.a[end])
tw_ipr = TwissExpr(ir_r, a_initial=tw_ipf.a[end])

# β ranges from 0.06 m to 700 m here: matching log(β) weighs relative errors equally
match!(vary=[@knob(qef1.Kn1), @knob(qef2.Kn1), @knob(qef3.Kn1), @knob(qef4.Kn1)],
       targets=[log(tw_ipf.beta1[end]) == log(0.6), log(tw_ipf.beta2[end]) == log(0.06),
                tw_ipf.alpha1[end] == 0, tw_ipf.alpha2[end] == 0])
match!(vary=[@knob(qer1.Kn1), @knob(qer2.Kn1), @knob(qer3.Kn1), @knob(qer4.Kn1)],
       targets=[log(tw_ipr.beta1[end]) == log(tw_ssr.beta1[begin]),
                log(tw_ipr.beta2[end]) == log(tw_ssr.beta2[begin]),
                tw_ipr.alpha1[end] == tw_ssr.alpha1[begin],
                tw_ipr.alpha2[end] == tw_ssr.alpha2[begin]])

# The interaction region replaces the last three straight section cells of sextant 5 and the
# first three of sextant 7
sextant5 = [repeat(fodossf, 4); ss_to_arcf; repeat(fodoaf, 20); arc_to_ssf; fodossf; ipf]
sextant7 = [ipr; fodossr; ss_to_arcr; repeat(fodoar, 20); arc_to_ssr; repeat(fodossr, 4)]

ring = Beamline([sextant1; sextant3; sextant5; sextant7; sextant9; sextant11],
                species_ref=electron, pc_ref=pc_ref)
