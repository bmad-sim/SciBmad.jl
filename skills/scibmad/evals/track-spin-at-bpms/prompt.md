---
tags: [skill-creator]
max_turns: 40
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

Using this SciBmad lattice:

```julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  bpm1 = BPM()
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  bpm2 = BPM()
  sd = Sextupole(Kn2=-1.2, L=0.2)
end
fodo = Beamline([qf, bpm1, sf, b, d, qd, bpm2, sd, b, d], species_ref=Species("electron"), pc_ref=18e9)
```

track three electrons starting at x = 1, 2 and 3 mm (all other coordinates zero) for 200 turns with spin, saving them at bpm1 and bpm2. Report (a) x of the third particle at bpm2 after turn 200, (b) how many particles are still alive after turn 200, (c) the horizontal (x) component of the spin of particle 1 at the end of the beamline after turn 200 if its spin starts along z, and (d) the largest |x| of particle 2 at bpm1 over all saved turns. Script track_bpms.jl, run it, results in results.md.
