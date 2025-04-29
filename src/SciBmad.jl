module SciBmad
using Beamlines, 
      BeamTracking, 
      NonlinearNormalForm, 
      GTPSA,
      LinearAlgebra,
      TypedTables

export twiss

const BTBL = Base.get_extension(BeamTracking, :BeamTrackingBeamlinesExt)

# Returns a Table of the Twiss parameters
# See Eq. 4.28 in EBB
function twiss(bl::Beamline; GTPSA_descriptor=GTPSA.desc_current)
  # First we track through once
  # This will get the map and tell us if coasting, etc etc
  if GTPSA_descriptor.desc == C_NULL
    GTPSA_descriptor = Descriptor(6,1)
  end

  b0 = Bunch(@vars(GTPSA_descriptor)[1:6], mem=BeamTracking.AoS, Brho_ref=bl.Brho_ref)
  linear = NonlinearNormalForm.maxord(b0.v[1]) == 1 ? true : false
  work = BTBL.get_work(b0,bl)
  track!(b0, bl; work=work)
  m = DAMap(v=b0.v)
  # function barrier
  return _twiss(m, work, bl, b0, Val{linear}())
end
function _twiss(m, work, bl::Beamline, b0::Bunch, ::Val{linear}) where {linear}
  N_ele = length(bl.line)
  a = normal(m)
  __, a1t, __ = factorize(a)
  r_cs = fast_canonize(a1t)
  a1 = a1t∘r_cs
  lf1 = compute_lattice_functions(a1, Val{linear}())
  phase = zeros(3, N_ele+1)
  s = zeros(N_ele+1)
  lf = Vector{typeof(lf1)}(undef, N_ele+1)
  lf[1] = lf1
  for i in 1:N_ele
    b0.v .= view(a1.v, 1:6) 
    track!(b0, bl.line[i])
    phase[:,i+1] .+= phase[:,i]
    s[i+1] = s[i] + bl.line[i].L
    r_cs = fast_canonize(a1, phase=view(phase, :, i+1))
    a1 = a1∘r_cs
    lf[i+1] = compute_lattice_functions(a1, Val{linear}())
  end
  t = Table(s=s, phi_x=phase[1,:], phi_y=phase[2,:], phi_z=phase[3,:],
            beta_xx=map(t->t.E[1][1,1], lf),
            alpha_xx=map(t->t.E[1][1,2], lf),
            beta_yy=map(t->t.E[2][3,3], lf),
            alpha_yy=map(t->t.E[2][3,4], lf),
            eta_x=map(t->t.H[3][1,6], lf),
            eta_y=map(t->t.H[3][3,6], lf),
            )
  return t
end

end
