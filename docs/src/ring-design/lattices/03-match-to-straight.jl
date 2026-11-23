# Ring design tutorial, chapter 3: matching the dispersion suppressors to the straight sections
#
# Four quadrupoles after each dispersion suppressor match the Twiss parameters to the periodic
# ones of the straight section FODO cells.

include("02-dispersion-suppressor.jl")

k_ss = qfss.Kn1 # The straight section strength, to start with

@elements begin
  qff2 = Quadrupole(L=0.5, Kn1=k_ss)
  qdf2 = Quadrupole(L=0.5, Kn1=-k_ss)
  qff3 = Quadrupole(L=0.5, Kn1=k_ss)
  qdf3 = Quadrupole(L=0.5, Kn1=-k_ss)
  qfr2 = Quadrupole(L=0.5, Kn1=k_ss)
  qdr2 = Quadrupole(L=0.5, Kn1=-k_ss)
  qfr3 = Quadrupole(L=0.5, Kn1=k_ss)
  qdr3 = Quadrupole(L=0.5, Kn1=-k_ss)
end

mssf = [qff2, d1, db, d2, qdf2, d1, db, d2, qff3, d1, db, d2, qdf3, d1, db, d2]
mssr = [qfr2, d2, db, d1, qdr2, d2, db, d1, qfr3, d2, db, d1, qdr3, d2, db, d1]

arc_to_ssf = [dispsupf; mssf]
arc_to_ssr = [dispsupr; mssr]

line_f = Beamline(arc_to_ssf, species_ref=electron, pc_ref=pc_ref)
line_r = Beamline(arc_to_ssr, species_ref=electron, pc_ref=pc_ref)

tw_f = TwissExpr(line_f, a_initial=tw_af.a[end])
tw_r = TwissExpr(line_r, a_initial=tw_ar.a[end])

match!(vary=[@knob(qff2.Kn1), @knob(qdf2.Kn1), @knob(qff3.Kn1), @knob(qdf3.Kn1)],
       targets=[tw_f.beta1[end] == tw_ssf.beta1[begin], tw_f.alpha1[end] == tw_ssf.alpha1[begin],
                tw_f.beta2[end] == tw_ssf.beta2[begin], tw_f.alpha2[end] == tw_ssf.alpha2[begin]])
match!(vary=[@knob(qfr2.Kn1), @knob(qdr2.Kn1), @knob(qfr3.Kn1), @knob(qdr3.Kn1)],
       targets=[tw_r.beta1[end] == tw_ssr.beta1[begin], tw_r.alpha1[end] == tw_ssr.alpha1[begin],
                tw_r.beta2[end] == tw_ssr.beta2[begin], tw_r.alpha2[end] == tw_ssr.alpha2[begin]])
