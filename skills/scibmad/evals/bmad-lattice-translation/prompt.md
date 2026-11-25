---
tags: [skill-creator]
max_turns: 40
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

Convert this Bmad lattice to SciBmad and give me the tunes, the chromaticities and the momentum compaction:

```
beginning[e_tot] = 3e9
parameter[particle] = electron
parameter[geometry] = closed

qf: quadrupole, l = 0.4, k1 = 0.8
qd: quadrupole, l = 0.4, k1 = -0.75
qs: quadrupole, l = 0.2, k1 = 0.05, tilt = 0.3
sf: sextupole, l = 0.2, k2 = 5.0
sd: sextupole, l = 0.2, k2 = -8.0
b: sbend, l = 2.0, angle = 0.1
d: drift, l = 0.5

cell: line = (qf, d, sf, d, b, d, qd, d, sd, d, b, d, qs, d)
use, cell
```

Script bmad_ring.jl, run it, results in results.md.
