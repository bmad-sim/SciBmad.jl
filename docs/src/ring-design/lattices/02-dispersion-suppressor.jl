# Ring design tutorial, chapter 2: dispersion suppressors
#
# Two arc FODO cells with half-strength bends, whose last two quadrupoles make the dispersion
# and its derivative zero at the end. The suppressors start with the periodic lattice
# functions of the arc cells.

include("01-fodo.jl")

@elements begin
  bh = SBend(L=Lc*(θ/2)/(2*sin(θ/4)), angle=θ/2, e1=θ/4, e2=θ/4) # Half-strength bend
  qff1 = Quadrupole(L=0.5, Kn1=0.3)
  qdf1 = Quadrupole(L=0.5, Kn1=-0.3)
  qfr1 = Quadrupole(L=0.5, Kn1=0.3)
  qdr1 = Quadrupole(L=0.5, Kn1=-0.3)
end

dispsupf = [qf, d1, bh, d2, qd, d1, bh, d2, qff1, d1, bh, d2, qdf1, d1, bh, d2]
dispsupr = [qf, d2, bh, d1, qd, d2, bh, d1, qfr1, d2, bh, d1, qdr1, d2, bh, d1]

ds_f = Beamline(dispsupf, species_ref=electron, pc_ref=pc_ref)
ds_r = Beamline(dispsupr, species_ref=electron, pc_ref=pc_ref)

tw_dsf = TwissExpr(ds_f, a_initial=tw_af.a[end])
tw_dsr = TwissExpr(ds_r, a_initial=tw_ar.a[end])

match!(vary=[@knob(qff1.Kn1), @knob(qdf1.Kn1)], targets=[tw_dsf.dx[end] == 0, tw_dsf.dpx[end] == 0])
match!(vary=[@knob(qfr1.Kn1), @knob(qdr1.Kn1)], targets=[tw_dsr.dx[end] == 0, tw_dsr.dpx[end] == 0])
