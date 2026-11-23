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

(rd.tracking)=
# 10. Introduction to Long Term Tracking

```{code-cell} julia
:tags: [remove-cell]
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

Up to this point, tracking has implicitly been limited to tracking once through the lattice, which
`twiss` does to compute the closed orbit and the lattice functions. SciBmad's `track` tracks any
number of particles over any number of turns, with spin, radiation, intra-beam scattering, and
lattice parameters that change with time (see [Track](../track.md)). The tracking is parallelized
on the CPU, and can run on GPUs.

## Initializing a Beam

The initial particles are given to `track` as a matrix with one row per particle and the six phase
space coordinates $x, p_x, y, p_y, z, p_z$ as its columns. Here are seven electrons with various
$z$ and $p_z$:

```{code-cell} julia
beam = [0  0  0  0  0     0
        0  0  0  0  0.1   0
        0  0  0  0  0.15  0
        0  0  0  0  0    -0.01
        0  0  0  0  0     0.015
        0  0  0  0  0     0.02
        0  0  0  0  0     0.03]
```

Such a matrix can also be read from a file, for example a text file with one row per particle with
`readdlm` of the `DelimitedFiles` package.

## Tracking

We track these electrons through the ring with the RF cavities of [chapter 9](rf.md), with
radiation damping in every element:

```{code-cell} julia
include("lattices/09-rf.jl");
for ele in ring.line
  ele.rad_damping = true
end
```

With radiation damping, the electrons lose energy in every bend, which the RF cavities replace.
The closed orbit is then not zero: it has a small $p_z$, which changes along the ring (the energy
is lost in the bends and gained in the cavities), and a $z$ such that the cavities restore the
energy lost in one turn. The particles above are relative to the closed orbit:

```{code-cell} julia
co = find_closed_orbit(ring).v0 # 1×6 matrix
v0 = co .+ beam
res = track(ring, v0=v0, n_turns=30)
```

The tracking saves the particles at the end of the ring after every turn. The state of each particle
after the last turn is in `res.state`: `0x1` is alive, and other values are the ways a particle
can be lost (see [Track](../track.md)):

```{code-cell} julia
res.state[turn=30]
```

The particles with the largest $z$ or $p_z$ are lost: they started outside of the RF bucket, the
region of stable longitudinal motion. The trajectories in the $z$-$p_z$ plane:

```{code-cell} julia
using CairoMakie
fig = Figure(size=(700, 450))
ax = Axis(fig[1, 1], xlabel="z [m]", ylabel="pz", limits=(-0.3, 0.3, -0.04, 0.04))
for i in 1:7
  scatter!(ax, res.z[turn=:, particle=i], res.pz[turn=:, particle=i], markersize=6,
           label="particle $i")
end
axislegend(ax, position=:rb)
fig
```

`res.z[turn=:, particle=i]` is $z$ of particle `i` after every turn, from turn 0 (the initial
coordinates) to turn 30. The coordinates of a lost particle stay at the values it had where it was
lost (here outside of the plot). The RF wavelength, the length of the ring divided by the harmonic
number, is 0.51 m.

## Emittance and Radiation

The emittance of a beam is the average action $\langle J\rangle$ of its particles in each normal
mode. With radiation, the emittances approach equilibrium values, at which the radiation damping and
the random excitation by the emission of photons (the radiation fluctuations) balance. To see this
in a reasonable time, here is a small ring of 10 sectors, for positrons at 13 GeV:

```{code-cell} julia
@elements begin
  q1 = Quadrupole(L=0.45, Kn1=1.0, tilt=0.01) # Tilted to couple the modes
  q2 = Quadrupole(L=0.55, Kn1=-1.0)
  d = Drift(L=0.3)
  bs = SBend(L=0.5, angle=0.01)
  rf = RFCavity(L=0.1, rf_frequency=500e6, voltage=1e8, zero_phase=PhaseRef.AboveTransition)
end
simple_ring = Beamline(repeat([d, q1, d, bs, d, q2, d, rf, bs], 10),
                       species_ref=Species("positron"), pc_ref=13e9)
for ele in simple_ring.line
  ele.rad_damping = true
  ele.rad_fluctuations = true
end
tws = twiss(simple_ring, cols=["a"], at=[1])
```

The summary values `damp1`, `damp2`, `damp3` are the damping decrements per turn: the amplitudes of
the modes decrease by the factor $e^{-\mathrm{damp}}$ per turn, and the emittances by
$e^{-2\,\mathrm{damp}}$. The column `a` is the matrix $A$ of the normalizing map at the start of the
ring: it takes the normalized coordinates $\tilde{x}$, in which the motion of each mode is a
(damped) rotation, to the phase space coordinates relative to the closed orbit. In the normalized
coordinates, the action of mode 1 is $J_1 = (\tilde{x}_1^2 + \tilde{x}_2^2)/2$, and likewise for the
other modes.

A beam of 100 particles is started with smaller emittances than the equilibrium ones, by drawing the
normalized coordinates from Gaussian distributions:

```{code-cell} julia
using Random, LinearAlgebra, Statistics
Random.seed!(1)
A = tws.a[1]
co_s = find_closed_orbit(simple_ring).v0
n = 100
emit0 = [1e-9, 1e-12, 1e-8] # Initial emittances of the three modes [m]
xn = randn(n, 6) .* sqrt.(repeat(emit0, inner=2))'
v0 = co_s .+ xn*A'
res = track(simple_ring, v0=v0, n_turns=20000, save_every_n_turns=100)
```

The emittance of mode 1 after each saved turn:

```{code-cell} julia
Ai = inv(A)
emit1 = [mean(sum(abs2, ((res.v[turn=t] .- co_s)*Ai')[:, 1:2], dims=2))/2 for t in res.turns]
fig = Figure(size=(700, 350))
ax = Axis(fig[1, 1], xlabel="turn", ylabel="ε₁ [m]")
lines!(ax, res.turns, emit1)
fig
```

The emittance approaches its equilibrium value with the damping time: after $n$ turns, the
difference from the equilibrium has decreased by the factor $e^{-2 n\,\mathrm{damp1}}$.

## Exercises

1. **Initial parameters:** Experiment with different initial particle coordinates and plot their
   trajectories. How much can you change $p_z$ without losing the particle? Track particles and plot
   their longitudinal phase space. Find the stable and unstable fixed points, and the invariant
   curves in the longitudinal phase space. (Turn radiation damping off to see invariant curves.)

2. **Equilibrium emittance:** Fit the emittance of the small ring with
   $\varepsilon(n) = \varepsilon_\infty + (\varepsilon_0 - \varepsilon_\infty) e^{-2n\,\mathrm{damp1}}$ to
   find the equilibrium emittance $\varepsilon_\infty$. (Bmad computes it from the radiation
   integrals: $7.12\cdot10^{-8}$ m.) How large is the statistical error with 100 particles?

3. **Emittance of mode 2:** Compute the emittance of mode 2 as well. What makes it nonzero? Hint:
   look at the definition of the quadrupole `q1`.
