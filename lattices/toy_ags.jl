using Beamlines, BeamTracking

Lb = 2 * asin(2.66666666666666652E+000 * 1.96327112309048618E-002 / 2)/1.96327112309048618E-002
@eles begin
 q1h = Quadrupole(L =  1.33333333333333348E+000, Kn1 = -4.38353456948131909E-002*1.33333333333333348E+000)
 dr = Drift(L =  6.66666666666666519E-001)
 b = SBend(L =  Lb, 
   g =  1.96327112309048618E-002, e1 = Lb*1.96327112309048618E-002/2, e2 = Lb*1.96327112309048618E-002/2)
 q2 = Quadrupole(L =  2.66666666666666696E+000, Kn1 =  2.18170351443189234E-002*2.66666666666666696E+000)
 csnk = Marker()
 wsnk = Marker()
 end1 = Marker()
end


ring = Beamline([q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h,
   q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr,
   q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b,
   dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr,
   b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2,
   dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr,
   q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b,
   dr, q2, dr, b, dr, q1h, csnk, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h,
   q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr,
   q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b,
   dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr,
   b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2,
   dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr,
   q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b,
   dr, q2, dr, b, dr, q1h, wsnk, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h,
   q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr,
   q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b,
   dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr,
   b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2,
   dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr,
   q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b, dr, q2, dr, b, dr, q1h, q1h, dr, b,
   dr, q2, dr, b, dr, q1h, end1],
   E_ref = 3.5587247443204529E+10 + 2*1.9018517828440544e10*Time(), #2.40490243299096680E+010
   species_ref = Species("proton"))

