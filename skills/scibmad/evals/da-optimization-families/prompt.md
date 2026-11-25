---
tags: [skill-creator]
max_turns: 60
timeout_seconds: 3600
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

The dynamic aperture of my little test ring is bad. Figure out which resonance driving terms and detuning coefficients matter most, then use the sextupoles to improve the dynamic aperture while keeping both chromaticities at +1. Assume emittances of 10 nm and 1 nm. Put the script in da_opt.jl and a short summary in results.md.

```julia
using SciBmad
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  d = Drift(L=0.3)
  b = SBend(L=2.0, angle=pi/20)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  s1 = Sextupole(Kn2=8.0, L=0.2)
  s2 = Sextupole(Kn2=-12.0, L=0.2)
  s3 = Sextupole(Kn2=8.0, L=0.2)
  s4 = Sextupole(Kn2=-12.0, L=0.2)
end
ring = Beamline(repeat([qf, s1, d, b, d, qd, s2, d, b, d, qf, s3, d, b, d, qd, s4, d, b, d], 5),
                species_ref=Species("electron"), pc_ref=3e9)
```
