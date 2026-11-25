---
tags: [skill-creator]
max_turns: 40
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

I'm using SciBmad. Here's my cell:

```julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-1.2, L=0.2)
  cav = RFCavity(L=0.1, voltage=5e6, harmon=15, zero_phase=PhaseRef.AboveTransition)
end
ring = Beamline([qf, sf, b, d, qd, sd, b, d, cav], species_ref=Species("electron"), pc_ref=18e9)
```

Write a script chrom_match.jl that sets both chromaticities to +2 using sf and sd, then prints the new sextupole strengths, the two chromaticities, the synchrotron tune (with the RF on), and the horizontal dispersion at the start of qd. Run it and put the printed results in results.md.
