# Ring design tutorial, chapter 15: spin tracking with ramping
#
# A ring of 60 FODO cells based on the layout of the AGS, for protons. The markers csnk and wsnk
# are the places of the partial Siberian snakes of the AGS.

using SciBmad

θ = pi/60   # Bend angle [rad]
Lc = 8/3    # Chord length of the bends [m]

@elements begin
  q1h = Quadrupole(L=4/3, Kn1=-0.0541431006647352)
  dr = Drift(L=2/3)
  b = SBend(L=Lc*θ/(2*sin(θ/2)), angle=θ, e1=θ/2, e2=θ/2)
  q2 = Quadrupole(L=8/3, Kn1=0.054724356256391)
  csnk = Marker()
  wsnk = Marker()
end

cell = [q1h, dr, b, dr, q2, dr, b, dr, q1h]

proton = Species("proton")
G = gyromagnetic_anomaly(proton) # The anomalous magnetic moment a = (g - 2)/2
m = massof(proton)               # Mass [eV]

ring = Beamline([repeat(cell, 20); csnk; repeat(cell, 20); wsnk; repeat(cell, 20)],
                species_ref=proton, E_ref=51.82/G*m)
