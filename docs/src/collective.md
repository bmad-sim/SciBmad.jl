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

(collective)=
# Collective Effects

```{code-cell} julia
:tags: [remove-cell]
using SciBmad
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```
All collective effects in SciBmad act at each element slice when specified. Gaussian intra-beam scattering is currently implemented and GPU-parallelized. Work is currently ongoing to implement space charge (Gaussian and PIC) and wakefields (short and long range) into SciBmad.

## Intra-Beam Scattering

The particles in a bunch scatter off of each other, which moves momentum from one plane to
another. In most rings this slowly increases the energy spread and the emittances of the
bunch, and it is especially important for dense bunches at low energy.

SciBmad computes intra-beam scattering (IBS) from the bunch itself: at the start of each
element (or slice), it computes the mean and the spread of the particle coordinates, and
applies the IBS of a Gaussian bunch with that spread over the length of the element (or
slice). This is made of a damping part and random kicks. IBS is turned on per element with
the `ibs_model` parameter (see [IBSParams](#ibs.params)):

- `IBSModel.None`: no IBS (the default)
- `IBSModel.Gaussian`: both the damping and the random kicks
- `IBSModel.GaussianDampingOnly`: only the damping
- `IBSModel.GaussianFluctuationsOnly`: only the random kicks

How strong IBS is depends on how many particles are in the bunch. Usually far fewer particles
are tracked than in the real bunch, so each tracked particle (a "macroparticle") stands for
many real ones, which is set with the `weight` of the macroparticles. For example, tracking
1000 macroparticles for a bunch of $10^{10}$ electrons:

```{code-cell} julia
using Statistics # For std

@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  d = Drift(L=1.2)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
end

fodo = Beamline([qf, b, d, qd, b, d],
        species_ref=Species("electron"), pc_ref=100e6)

# Turn on IBS in every element:
foreach(x -> x.ibs_model = IBSModel.Gaussian, fodo)

n_particles = 1000
v0 = randn(n_particles, 6) .* [1e-4 1e-5 1e-5 1e-6 1e-3 1e-4]
res = track(fodo, v0=v0, weight=1e10/n_particles, n_turns=100)

println("Energy spread before: ", std(res.v[:,6,1]))
println("Energy spread after:  ", std(res.v[:,6,end]))
```

With $10^6$ electrons instead (`weight=1e6/n_particles`), the energy spread would hardly
change. The weights can also differ per macroparticle, given as a vector; the number of real
particles in the bunch is then the sum of the weights. Particles that are lost do not count.

IBS is computed once per slice of an element (see
[Integration Steps and Slices](#track.steps)), from the bunch at the start of the slice. By
default an element is one slice. If the bunch changes a lot within an element, for example in
a long element where the beam size changes, increase `n_slices`.
