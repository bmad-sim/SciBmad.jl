---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Julia
  language: julia
  name: julia
---

(twiss)=
# Twiss

```{code-cell} julia
:tags: [remove-cell]
using SciBmad
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

```{docstring} twiss
```

## Open lattices

By default `twiss` computes the periodic Twiss functions of a ring. For a line that is not 
periodic, such as a transfer line, give the Twiss functions at its start with `a_initial`: the 
map `a` from normal form coordinates to laboratory coordinates. The `a` of a periodic lattice is 
the `a` column of its `twiss` table. Here the periodic Twiss functions of a FODO cell continue 
into a line that starts with the cell:

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  d = Drift(L=1.0)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  q3 = Quadrupole(Kn1=0.2, L=0.5)
end
cell = Beamline([qf, d, qd, d], species_ref=Species("electron"), pc_ref=18e9)
a = twiss(cell, cols=["a"], as_taylor_series=true, at=[1]).a[1]

line = Beamline([qf, d, qd, d, q3, d], species_ref=Species("electron"), pc_ref=18e9)
twiss(line, a_initial=a, cols=["beta1", "alpha1", "beta2", "alpha2"])
```

Through the first four elements these are the periodic Twiss functions of the cell. The same 
works for matching: the Twiss functions at the end of the line can be matched to a target as in 
[](#optimize), with `a_initial` given to `twiss`.
