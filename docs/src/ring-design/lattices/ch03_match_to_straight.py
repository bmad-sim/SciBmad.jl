# Ring design tutorial, chapter 3: matching the dispersion suppressors to the straight sections
#
# Four quadrupoles after each dispersion suppressor match the Twiss parameters to the periodic
# ones of the straight section FODO cells.

from .ch02_dispersion_suppressor import *

k_ss = qfss.Kn1  # The straight section strength, to start with

with sb.elements():
    qff2 = sb.Quadrupole(L=0.5, Kn1=k_ss)
    qdf2 = sb.Quadrupole(L=0.5, Kn1=-k_ss)
    qff3 = sb.Quadrupole(L=0.5, Kn1=k_ss)
    qdf3 = sb.Quadrupole(L=0.5, Kn1=-k_ss)
    qfr2 = sb.Quadrupole(L=0.5, Kn1=k_ss)
    qdr2 = sb.Quadrupole(L=0.5, Kn1=-k_ss)
    qfr3 = sb.Quadrupole(L=0.5, Kn1=k_ss)
    qdr3 = sb.Quadrupole(L=0.5, Kn1=-k_ss)

mssf = [qff2, d1, db, d2, qdf2, d1, db, d2, qff3, d1, db, d2, qdf3, d1, db, d2]
mssr = [qfr2, d2, db, d1, qdr2, d2, db, d1, qfr3, d2, db, d1, qdr3, d2, db, d1]

arc_to_ssf = dispsupf + mssf
arc_to_ssr = dispsupr + mssr

line_f = sb.Beamline(arc_to_ssf, species_ref=electron, pc_ref=pc_ref)
line_r = sb.Beamline(arc_to_ssr, species_ref=electron, pc_ref=pc_ref)

tw_f = sb.TwissExpr(line_f, a_initial=tw_af.a[-1])
tw_r = sb.TwissExpr(line_r, a_initial=tw_ar.a[-1])

sb.match(vary=[sb.Knob(q, "Kn1") for q in (qff2, qdf2, qff3, qdf3)],
         targets=[tw_f.beta1[-1] == tw_ssf.beta1[sb.begin],
                  tw_f.alpha1[-1] == tw_ssf.alpha1[sb.begin],
                  tw_f.beta2[-1] == tw_ssf.beta2[sb.begin],
                  tw_f.alpha2[-1] == tw_ssf.alpha2[sb.begin]])
sb.match(vary=[sb.Knob(q, "Kn1") for q in (qfr2, qdr2, qfr3, qdr3)],
         targets=[tw_r.beta1[-1] == tw_ssr.beta1[sb.begin],
                  tw_r.alpha1[-1] == tw_ssr.alpha1[sb.begin],
                  tw_r.beta2[-1] == tw_ssr.beta2[sb.begin],
                  tw_r.alpha2[-1] == tw_ssr.alpha2[sb.begin]])
