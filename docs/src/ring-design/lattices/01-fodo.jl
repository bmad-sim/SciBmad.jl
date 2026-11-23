# Ring design tutorial, chapter 1: FODO cells
#
# The forward and reverse FODO cells of the arcs and of the straight sections, matched to a
# phase advance of 90° per cell in both planes.

using SciBmad

θ = pi/132   # Bend angle of the arc bends [rad]: 264 of them make 2π
Lc = 6.86    # Chord length of the arc bends [m]

@elements begin
  qf = Quadrupole(L=0.5, Kn1=0.4)
  qd = Quadrupole(L=0.5, Kn1=-0.4)
  b = SBend(L=Lc*θ/(2*sin(θ/2)), angle=θ, e1=θ/2, e2=θ/2) # Rectangular bend
  d1 = Drift(L=0.609)
  d2 = Drift(L=1.241)
  qfss = Quadrupole(L=0.5, Kn1=0.4)
  qdss = Quadrupole(L=0.5, Kn1=-0.4)
  db = Drift(L=5.855)
end

fodoaf = [qf, d1, b, d2, qd, d1, b, d2]           # Forward arc FODO cell
fodoar = [qf, d2, b, d1, qd, d2, b, d1]           # Reverse arc FODO cell
fodossf = [qfss, d1, db, d2, qdss, d1, db, d2]    # Forward straight section FODO cell
fodossr = [qfss, d2, db, d1, qdss, d2, db, d1]    # Reverse straight section FODO cell

electron = Species("electron")
pc_ref = 17.846262612447e9 # Reference momentum [eV/c]

cell_af = Beamline(fodoaf, species_ref=electron, pc_ref=pc_ref)
cell_ar = Beamline(fodoar, species_ref=electron, pc_ref=pc_ref)
cell_ssf = Beamline(fodossf, species_ref=electron, pc_ref=pc_ref)
cell_ssr = Beamline(fodossr, species_ref=electron, pc_ref=pc_ref)

tw_af = TwissExpr(cell_af)
tw_ar = TwissExpr(cell_ar)
tw_ssf = TwissExpr(cell_ssf)
tw_ssr = TwissExpr(cell_ssr)

# 90° phase advance per cell in both planes. The reverse cells use the same quadrupoles, and
# have the same phase advances (exercise 5).
match!(vary=[@knob(qf.Kn1), @knob(qd.Kn1)], targets=[tw_af.q1 == 0.25, tw_af.q2 == 0.25])
match!(vary=[@knob(qfss.Kn1), @knob(qdss.Kn1)], targets=[tw_ssf.q1 == 0.25, tw_ssf.q2 == 0.25])
