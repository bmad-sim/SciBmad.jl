# Ring design tutorial, chapter 1: FODO cells
#
# The forward and reverse FODO cells of the arcs and of the straight sections, matched to a
# phase advance of 90° per cell in both planes.

import numpy as np
import scibmad as sb

θ = np.pi/132  # Bend angle of the arc bends [rad]: 264 of them make 2π
Lc = 6.86      # Chord length of the arc bends [m]

with sb.elements():
    qf = sb.Quadrupole(L=0.5, Kn1=0.4)
    qd = sb.Quadrupole(L=0.5, Kn1=-0.4)
    b = sb.SBend(L=Lc*θ/(2*np.sin(θ/2)), angle=θ, e1=θ/2, e2=θ/2)  # Rectangular bend
    d1 = sb.Drift(L=0.609)
    d2 = sb.Drift(L=1.241)
    qfss = sb.Quadrupole(L=0.5, Kn1=0.4)
    qdss = sb.Quadrupole(L=0.5, Kn1=-0.4)
    db = sb.Drift(L=5.855)

fodoaf = [qf, d1, b, d2, qd, d1, b, d2]           # Forward arc FODO cell
fodoar = [qf, d2, b, d1, qd, d2, b, d1]           # Reverse arc FODO cell
fodossf = [qfss, d1, db, d2, qdss, d1, db, d2]    # Forward straight section FODO cell
fodossr = [qfss, d2, db, d1, qdss, d2, db, d1]    # Reverse straight section FODO cell

electron = sb.Species("electron")
pc_ref = 17.846262612447e9  # Reference momentum [eV/c]

cell_af = sb.Beamline(fodoaf, species_ref=electron, pc_ref=pc_ref)
cell_ar = sb.Beamline(fodoar, species_ref=electron, pc_ref=pc_ref)
cell_ssf = sb.Beamline(fodossf, species_ref=electron, pc_ref=pc_ref)
cell_ssr = sb.Beamline(fodossr, species_ref=electron, pc_ref=pc_ref)

tw_af = sb.TwissExpr(cell_af)
tw_ar = sb.TwissExpr(cell_ar)
tw_ssf = sb.TwissExpr(cell_ssf)
tw_ssr = sb.TwissExpr(cell_ssr)

# 90° phase advance per cell in both planes. The reverse cells use the same quadrupoles, and
# have the same phase advances (exercise 5).
sb.match(vary=[sb.Knob(qf, "Kn1"), sb.Knob(qd, "Kn1")],
         targets=[tw_af.q1 == 0.25, tw_af.q2 == 0.25])
sb.match(vary=[sb.Knob(qfss, "Kn1"), sb.Knob(qdss, "Kn1")],
         targets=[tw_ssf.q1 == 0.25, tw_ssf.q2 == 0.25])
