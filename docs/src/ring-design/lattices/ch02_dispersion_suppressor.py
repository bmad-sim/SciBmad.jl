# Ring design tutorial, chapter 2: dispersion suppressors
#
# Two arc FODO cells with half-strength bends, whose last two quadrupoles make the dispersion
# and its derivative zero at the end. The suppressors start with the periodic lattice
# functions of the arc cells.

from .ch01_fodo import *

with sb.elements():
    bh = sb.SBend(L=Lc*(θ/2)/(2*np.sin(θ/4)), angle=θ/2, e1=θ/4, e2=θ/4)  # Half-strength bend
    qff1 = sb.Quadrupole(L=0.5, Kn1=0.3)
    qdf1 = sb.Quadrupole(L=0.5, Kn1=-0.3)
    qfr1 = sb.Quadrupole(L=0.5, Kn1=0.3)
    qdr1 = sb.Quadrupole(L=0.5, Kn1=-0.3)

dispsupf = [qf, d1, bh, d2, qd, d1, bh, d2, qff1, d1, bh, d2, qdf1, d1, bh, d2]
dispsupr = [qf, d2, bh, d1, qd, d2, bh, d1, qfr1, d2, bh, d1, qdr1, d2, bh, d1]

ds_f = sb.Beamline(dispsupf, species_ref=electron, pc_ref=pc_ref)
ds_r = sb.Beamline(dispsupr, species_ref=electron, pc_ref=pc_ref)

tw_dsf = sb.TwissExpr(ds_f, a_initial=tw_af.a[-1])
tw_dsr = sb.TwissExpr(ds_r, a_initial=tw_ar.a[-1])

sb.match(vary=[sb.Knob(qff1, "Kn1"), sb.Knob(qdf1, "Kn1")],
         targets=[tw_dsf.dx[-1] == 0, tw_dsf.dpx[-1] == 0])
sb.match(vary=[sb.Knob(qfr1, "Kn1"), sb.Knob(qdr1, "Kn1")],
         targets=[tw_dsr.dx[-1] == 0, tw_dsr.dpx[-1] == 0])
