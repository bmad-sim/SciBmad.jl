---
tags: [skill-creator]
max_turns: 40
timeout_seconds: 1800
allowed_tools: [Read, Glob, Grep, Skill, Write, Edit, Bash]
---

My FODO cell (SciBmad) has the quads tied together through the beamline context:

```julia
@elements begin
  qf = Quadrupole(L=0.5)
  d = Drift(L=1.0)
  qd = Quadrupole(L=0.5)
  b = SBend(L=2.0, angle=pi/20)
end
fodo = Beamline([qf, d, b, d, qd, d, b, d], species_ref=Species("electron"), pc_ref=18e9)
fodo.context.kq = 0.36
qf.Kn1 = DefExpr(c -> c.kq)
qd.Kn1 = DefExpr(c -> -c.kq)
```

Find the kq that gives a horizontal tune of 0.16, keeping qd = -qf. Then tell me how fast q1 and q2 change with kq at that point, using exact derivatives (not finite differences). Afterwards, changing fodo.context.kq must still move both quads. Put the script in match_kq.jl, run it, and write the results to results.md.
