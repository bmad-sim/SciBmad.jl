# Ring design tutorial, chapter 15: spin tracking with ramping
#
# A ring of 60 FODO cells based on the layout of the AGS, for protons. The markers csnk and wsnk
# are the places of the partial Siberian snakes of the AGS.

import numpy as np
import scibmad as sb

θ = np.pi/60  # Bend angle [rad]
Lc = 8/3      # Chord length of the bends [m]

with sb.elements():
    q1h = sb.Quadrupole(L=4/3, Kn1=-0.0541431006647352)
    dr = sb.Drift(L=2/3)
    b = sb.SBend(L=Lc*θ/(2*np.sin(θ/2)), angle=θ, e1=θ/2, e2=θ/2)
    q2 = sb.Quadrupole(L=8/3, Kn1=0.054724356256391)
    csnk = sb.Marker()
    wsnk = sb.Marker()

cell = [q1h, dr, b, dr, q2, dr, b, dr, q1h]

proton = sb.Species("proton")
G = sb.gyromagnetic_anomaly(proton)  # The anomalous magnetic moment a = (g - 2)/2
m = sb.massof(proton)                # Mass [eV]

ring = sb.Beamline(cell*20 + [csnk] + cell*20 + [wsnk] + cell*20,
                   species_ref=proton, E_ref=51.82/G*m)
