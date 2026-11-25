# Ring design tutorial, chapter 6: low-beta interaction region
#
# An interaction region with β* = 0.6 m and 0.06 m at IP6, in the straight section between the
# sextants 5 and 7. The upstream half IPF starts with the periodic lattice functions of the
# straight section, and the downstream half IPR with the lattice functions that IPF gives at
# the IP. IPR then ends with the periodic lattice functions of the straight section.

from .ch05_ring import *

with sb.elements():
    qef1 = sb.Quadrupole(L=0.5, Kn1=k_ss)
    qef2 = sb.Quadrupole(L=0.5, Kn1=-k_ss)
    def1 = sb.Drift(L=20.46)
    qef3 = sb.Quadrupole(L=1.6, Kn1=k_ss)
    def2 = sb.Drift(L=3.76)
    qef4 = sb.Quadrupole(L=1.2, Kn1=-k_ss)
    def3 = sb.Drift(L=5.8)
    ip6 = sb.Marker()
    der3 = sb.Drift(L=5.3)
    qer4 = sb.Quadrupole(L=1.8, Kn1=-k_ss)
    der2 = sb.Drift(L=0.5)
    qer3 = sb.Quadrupole(L=1.4, Kn1=k_ss)
    der1 = sb.Drift(L=23.82)
    qer2 = sb.Quadrupole(L=0.5, Kn1=k_ss)
    qer1 = sb.Quadrupole(L=0.5, Kn1=-k_ss)

ipf = [qef1, d1, db, d2, qef2, d1, db, d2, def1, qef3, def2, qef4, def3, ip6]
ipr = [der3, qer4, der2, qer3, der1, qer2, d2, db, d1, qer1, d2, db, d1]

ir_f = sb.Beamline(ipf, species_ref=electron, pc_ref=pc_ref)
ir_r = sb.Beamline(ipr, species_ref=electron, pc_ref=pc_ref)

tw_ipf = sb.TwissExpr(ir_f, a_initial=tw_ssf.a[-1])
tw_ipr = sb.TwissExpr(ir_r, a_initial=tw_ipf.a[-1])

# β ranges from 0.06 m to 700 m here: matching log(β) weighs relative errors equally
sb.match(vary=[sb.Knob(q, "Kn1") for q in (qef1, qef2, qef3, qef4)],
         targets=[np.log(tw_ipf.beta1[-1]) == np.log(0.6),
                  np.log(tw_ipf.beta2[-1]) == np.log(0.06),
                  tw_ipf.alpha1[-1] == 0, tw_ipf.alpha2[-1] == 0])
sb.match(vary=[sb.Knob(q, "Kn1") for q in (qer1, qer2, qer3, qer4)],
         targets=[np.log(tw_ipr.beta1[-1]) == np.log(tw_ssr.beta1[sb.begin]),
                  np.log(tw_ipr.beta2[-1]) == np.log(tw_ssr.beta2[sb.begin]),
                  tw_ipr.alpha1[-1] == tw_ssr.alpha1[sb.begin],
                  tw_ipr.alpha2[-1] == tw_ssr.alpha2[sb.begin]])

# The interaction region replaces the last three straight section cells of sextant 5 and the
# first three of sextant 7
sextant5 = fodossf*4 + ss_to_arcf + fodoaf*20 + arc_to_ssf + fodossf + ipf
sextant7 = ipr + fodossr + ss_to_arcr + fodoar*20 + arc_to_ssr + fodossr*4

ring = sb.Beamline(sextant1 + sextant3 + sextant5 + sextant7 + sextant9 + sextant11,
                   species_ref=electron, pc_ref=pc_ref)
