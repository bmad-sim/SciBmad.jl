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

(rd.phasespace)=
# 8. Particle Phase Space Coordinates

```{code-cell} julia
:tags: [remove-cell]
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

## Phase Space

```{figure} figures/local-coords.svg
:width: 48%

Particle coordinates are relative to the reference trajectory.
```

```{figure} figures/coordinatesystem1.svg
:width: 48%

Particle phase space.
```

The first figure shows the reference trajectory (the branch coordinate system), which was discussed
in [Machine Coordinates](coordinates.md). The branch coordinates $(x, y, z)$ associated with a given
point a distance $s$ along the reference trajectory have their origin at that point and the $z$ axis
tangent to the reference trajectory.

A particle being simulated has its own trajectory, as shown in the figure. Given a particle at some
point on its trajectory, there is a point at position $s$ on the reference trajectory such that the
$z$ coordinate of the particle is zero in the $(x, y, z)$ coordinate frame associated with that
point. With this, the particle's position and momentum $\mathbf{P}$ can be described using the
coordinates

$$
(x(s), y(s), P_x(s), P_y(s), P_z(s), t(s))
$$

where $t(s)$ is the time at which the particle is at the given point. From now on, to simplify the
notation, the $s$ dependence will be dropped.

For tracking, canonical phase space coordinates are used, with the convention that an upper case
$P$ denotes the (unnormalized) momentum and a lower case $p$ the phase space momentum. The phase
space coordinates are

$$
(x, p_x, y, p_y, z, p_z)
$$

where

$$
p_x = \frac{P_x}{P_0}, \quad p_y = \frac{P_y}{P_0}, \quad p_z = \frac{P - P_0}{P_0}, \quad
z = -\beta c\,(t - t_\mathrm{ref})
$$

with

- $P_0$ the reference momentum, see [Reference Energy](#rd.refenergy).
- $\beta c$ the velocity of the particle.
- $t_\mathrm{ref}(s)$ the time at which the reference particle reaches the point $s$. The reference
  particle is a fictitious particle that can be imagined to travel on the reference trajectory.
  Frequently, this reference particle is thought of as describing the center of a bunch of particles.

Notes:

- Do not confuse the canonical $z$ coordinate with the $z$ coordinate of the particle in the
  $(x, y, z)$ coordinate frame. By construction, the latter is always zero.
- For a bunch of particles at a given $s$ position, in general, the particles will have differing
  times $t$.
- If the reference particle has the same $\beta$ as a particle, the canonical $z$ is the
  longitudinal distance of the particle from the reference particle. Positive $z$ means that the
  particle is in front of the reference particle.
- $p_z$ is also called $\delta$.

## Example

A beamline with a bend and a quadrupole, for positrons with 10 MeV total energy:

```{code-cell} julia
using SciBmad

@elements begin
  b = SBend(L=0.5, angle=0.5) # g_ref = 1/bending radius = 1 1/m
  q = Quadrupole(L=0.6, Kn1=10.0)
end
lat = Beamline([b, q], species_ref=Species("positron"), E_ref=10e6)
```

A particle is tracked through it with `track`, starting with $y$ = 1 cm, $p_x$ = 0.06 and
$p_z$ = −0.2, with spin tracking turned on (`spin=true`). The coordinates are saved at every
integration step (`at=:`), and, with more integration steps, also inside the elements:

```{code-cell} julia
b.n_steps = 20
q.n_steps = 20
v0 = [0 0.06 0.01 0 0 -0.2] # x, px, y, py, z, pz
res = track(lat, v0=v0, spin=true, at=:)
```

The result holds the particle at the saved locations: for example its coordinates at the end of
the beamline after the first (and only) turn:

```{code-cell} julia
res.v[end, turn=1, particle=1]
```

Spin tracking tracks the rotation of the spin as a quaternion, so that the spin can be computed
afterwards for any initial spin. With the initial spin along $x$:

```{code-cell} julia
s = track_spin(res.q, [1, 0, 0])
s[end, turn=1, particle=1] # The spin (x, y, z) at the end
```

The orbit, with $p_x$ = 0.06 at the start and changed to $p_x$ = 0.1:

```{code-cell} julia
using CairoMakie
fig = Figure(size=(700, 350))
ax = Axis(fig[1, 1], xlabel="s [m]", ylabel="[mm]")
lines!(ax, res.s, 1e3 .* res.x[:, turn=1, particle=1], label="x")
lines!(ax, res.s, 1e3 .* res.y[:, turn=1, particle=1], label="y")
res2 = track(lat, v0=[0 0.1 0.01 0 0 -0.2], spin=true, at=:)
lines!(ax, res2.s, 1e3 .* res2.x[:, turn=1, particle=1], label="x (px = 0.1)", linestyle=:dash)
axislegend(ax, position=:lt)
fig
```

`res.s` is the $s$ position of every saved location, and `res.x[:, turn=1, particle=1]` is $x$ at
all of them. See [Track](../track.md) for everything a `TrackResult` holds.

## Exercises

1. **Phase space:** Construct a beamline with a single drift of 2 m length. Set the reference
   particle to be a proton with a reference energy of $10^{12}$ eV (so that the approximation that
   the particle velocity is the speed of light can be made), and set the initial particle position
   to $p_x$ = 0.2, $p_z$ = 1 with all other coordinates zero. Calculate from first principles what
   the phase space coordinates will be at the end of the beamline, and check that your answer
   agrees with `track`.

2. **Fringe fields:** Quadrupoles have hard-edge fringe fields at both ends by default
   (`q.fringe_at` is `Fringe.BothEnds`). Track the particle of the example again with
   `q.fringe_at = Fringe.NoEnd`. How much do the fringe fields change the orbit at the end? Why are
   they so important here?
