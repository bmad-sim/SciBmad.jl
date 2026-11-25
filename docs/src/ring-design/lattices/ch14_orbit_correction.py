# Ring design tutorial, chapter 14: orbit correction
#
# A corrector in every drift d1 next to a quadrupole, 6.4 cm from it: horizontal next to
# focusing and vertical next to defocusing quadrupoles. Each corrector is an element of its own,
# so that its kick can be set by itself. A BPM after every bend b, bh and straight section
# drift db. The orbit with radiation damping (the sawtooth) is corrected with the horizontal
# correctors. The phase trombones are maps around the closed orbit at the time they are built, so
# they are taken out for the orbit correction and put back afterwards.

from .ch12_sextupoles import *

with sb.elements():
    d1c1 = sb.Drift(L=0.064)
    d1c2 = sb.Drift(L=0.609 - 0.064 - 0.2)
    bpm = sb.Marker()

line = sextant1 + sextant3 + sextant5 + sextant7 + sextant9 + sextant11
n = len(line)
elements = []
hkickers = []
vkickers = []
for i in range(n):
    ele = line[i]
    if ele is d1:
        # The quadrupole next to d1: before it in forward cells, after it in reverse cells (the
        # next element of the ring after the last one is the first one)
        forward = line[i-1].kind == "Quadrupole"
        j = i - 1 if forward else (i + 1) % n
        while line[j].kind != "Quadrupole":
            j = (j + 1) % n
        if line[j].Kn1 > 0:
            kicker = sb.HKicker(name=f"ch_{len(hkickers) + 1}", L=0.2)
            hkickers.append(kicker)
        else:
            kicker = sb.VKicker(name=f"cv_{len(vkickers) + 1}", L=0.2)
            vkickers.append(kicker)
        elements += [d1c1, kicker, d1c2] if forward else [d1c2, kicker, d1c1]
    else:
        elements.append(ele)
    if ele is b or ele is bh or ele is db:
        elements.append(bpm)

ring = sb.Beamline(elements, species_ref=electron, pc_ref=pc_ref, context=ring.context)

for tr in (tr_f, tr_r, tr_m):
    ring[tr].MapParams = None

# Radiation damping in every element
for ele in ring.line:
    ele.rad_damping = True

tw_orbit = sb.TwissExpr(ring)
sb.match(vary=[sb.Knob(kicker, "Kn0L") for kicker in hkickers], minimize=tw_orbit.x[[bpm]])

sb.phase_trombone(ring[tr_f], phi1=sb.DefExpr(lambda c: c.x1), phi2=sb.DefExpr(lambda c: c.y1))
sb.phase_trombone(ring[tr_r], phi1=sb.DefExpr(lambda c: c.x2), phi2=sb.DefExpr(lambda c: c.y2))
sb.phase_trombone(ring[tr_m], phi1=sb.DefExpr(lambda c: -c.x1 - c.x2),
                  phi2=sb.DefExpr(lambda c: -c.y1 - c.y2))
