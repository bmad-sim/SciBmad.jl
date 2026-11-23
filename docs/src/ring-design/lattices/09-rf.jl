# Ring design tutorial, chapter 9: RF cavities
#
# Four FODO cells with two RF cavities between each pair of quadrupoles replace the four
# straight section cells around the 10 o'clock point (between the sextants 9 and 11). The
# voltage gives a synchrotron tune of 0.05.

include("07-tune-cell.jl")

@elements begin
  rf0 = RFCavity(L=3.41, harmon=7560, voltage=68e6/18, zero_phase=PhaseRef.AboveTransition)
  drf = Drift(L=((1.241 + 5.855 + 0.609) - 2*3.41)/3)
end

fodorf = [qfss, drf, rf0, drf, rf0, drf, qdss, drf, rf0, drf, rf0, drf]

sextant9 = [repeat(fodossf, 4); ss_to_arcf; repeat(fodoaf, 20); arc_to_ssf; repeat(fodossf, 2);
            repeat(fodorf, 2)]
sextant11 = [repeat(fodorf, 2); repeat(fodossr, 2); ss_to_arcr; repeat(fodoar, 20); arc_to_ssr;
             repeat(fodossr, 4)]

ring = Beamline([sextant1; sextant3; sextant5; sextant7; sextant9; sextant11],
                species_ref=electron, pc_ref=pc_ref)

tw_ring = TwissExpr(ring)

# Above transition, the synchrotron tune q3 is negative
match!(vary=[@knob(rf0.voltage)], targets=[tw_ring.q3 == -0.05], solver=:gauss_newton)
