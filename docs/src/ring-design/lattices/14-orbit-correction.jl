# Ring design tutorial, chapter 14: orbit correction
#
# A corrector in every drift d1 next to a quadrupole, 6.4 cm from it: horizontal next to
# focusing and vertical next to defocusing quadrupoles. Each corrector is an element of its own,
# so that its kick can be set by itself. A BPM after every bend b, bh and straight section
# drift db. The orbit with radiation damping (the sawtooth) is corrected with the horizontal
# correctors. The phase trombones are maps around the closed orbit at the time they are built, so
# they are taken out for the orbit correction and put back afterwards.

include("12-sextupoles.jl")

@elements begin
  d1c1 = Drift(L=0.064)
  d1c2 = Drift(L=0.609 - 0.064 - 0.2)
  bpm = Marker()
end

line = [sextant1; sextant3; sextant5; sextant7; sextant9; sextant11]
n = length(line)
elements = LatElement[]
hkickers = LatElement[]
vkickers = LatElement[]
for i in 1:n
  ele = line[i]
  if ele === d1
    # The quadrupole next to d1: before it in forward cells, after it in reverse cells (the
    # next element of the ring after the last one is the first one)
    forward = line[i-1].kind == "Quadrupole"
    j = forward ? i - 1 : mod1(i + 1, n)
    while line[j].kind != "Quadrupole"
      j = mod1(j + 1, n)
    end
    if line[j].Kn1 > 0
      kicker = HKicker(name="ch_$(length(hkickers) + 1)", L=0.2)
      push!(hkickers, kicker)
    else
      kicker = VKicker(name="cv_$(length(vkickers) + 1)", L=0.2)
      push!(vkickers, kicker)
    end
    append!(elements, forward ? [d1c1, kicker, d1c2] : [d1c2, kicker, d1c1])
  else
    push!(elements, ele)
  end
  if ele === b || ele === bh || ele === db
    push!(elements, bpm)
  end
end

ring = Beamline(elements, species_ref=electron, pc_ref=pc_ref, context=ring.context)

for tr in (tr_f, tr_r, tr_m)
  ring[tr].MapParams = nothing
end

# Radiation damping in every element
for ele in ring.line
  ele.rad_damping = true
end

tw_orbit = TwissExpr(ring)
match!(vary=[Knob(kicker, :Kn0L) for kicker in hkickers], minimize=tw_orbit.x[[bpm]])

phase_trombone!(ring[tr_f]; phi1=DefExpr(c -> c.x1), phi2=DefExpr(c -> c.y1))
phase_trombone!(ring[tr_r]; phi1=DefExpr(c -> c.x2), phi2=DefExpr(c -> c.y2))
phase_trombone!(ring[tr_m]; phi1=DefExpr(c -> -c.x1 - c.x2), phi2=DefExpr(c -> -c.y1 - c.y2))
