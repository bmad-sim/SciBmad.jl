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

(rd.spin)=
# 15. Spin Tracking with Ramping

```{code-cell} julia
:tags: [remove-cell]
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

## Introduction

SciBmad computes the most important quantities of polarization analysis, such as the invariant spin
field and the amplitude-dependent spin tune (with `twiss(…; spin=true)`, see [Twiss](#twiss.spin)),
and tracks spins (with `track(…; spin=true)`). In this chapter, we will ramp the energy of a storage
ring through a spin-orbit resonance, calculate the polarization transmission, and compare it with
the resonance strength computed from the invariant spin field, with the Froissart-Stora formula.

In a perfectly flat ring (a ring in which the only magnetic fields on the closed orbit are vertical),
the closed-orbit spin tune is $\nu_0 = G\gamma$, where $G = (g-2)/2$ is the anomalous gyromagnetic
$g$-factor and $\gamma$ the Lorentz factor of the reference particle. When $\nu_0$ is close to a
frequency of the orbital motion, $\kappa = k_0 + k_1 Q_1 + k_2 Q_2$ with integers $k_i$, the spin
motion is driven resonantly. Each turn, the spin is rotated a little around a direction that
depends on the orbit, and the rotations add up. We will ramp our ring through the resonance
$\nu_0 = 60 - Q_2$.

## Example: Ramping Through a Spin-Orbit Resonance

1. The ring is a ring of 60 FODO cells, based on the layout of the AGS, for protons:

```{code-cell} julia
include("lattices/15-spin.jl");
tw = twiss(ring, spin=true)
tw.q1, tw.q2, tw.qspin
```

   The spin tune is the fractional part of $G\gamma = 51.82$ (with the sign convention of SciBmad,
   $-0.18$). The resonance is at

```{code-cell} julia
κ = 60 - tw.q2
```

2. Ramp $G\gamma$ from $\kappa - 0.1$ to $\kappa + 0.1$ in 5000 turns. The reference energy is made
   a function of time with `Time()` (see [Time-Dependent Parameters and Ramping](../timedependent.md)).
   The time of one turn is the length of the ring divided by the velocity of the protons:

```{code-cell} julia
γ = κ/G
T = ring.line[end].s_downstream/(sqrt(1 - 1/γ^2)*C_LIGHT) # Revolution time [s]
n_turns = 5000
E_start = (κ - 0.1)/G*m
E_end = (κ + 0.1)/G*m
ring.E_ref = E_start + (E_end - E_start)/(n_turns*T)*Time()
```

3. Track one proton with a vertical amplitude through the ramp, with spin. The ring has no RF
   cavities, so the energy of the particle is ramped with the reference energy
   (`ramp_particle_energy_without_rf=true`):

```{code-cell} julia
y0 = 1e-5
res = track(ring, v0=[0 0 y0 0 0 0], spin=true, n_turns=n_turns,
            ramp_particle_energy_without_rf=true)
```

4. Plot the vertical component of the spin, for a spin that is vertical at the start, against the
   turn number. `track_spin` computes the spins from the tracked spin rotations (quaternions):

```{code-cell} julia
s = track_spin(res.q, [0, 1, 0])
s_y = s[turn=:, particle=1][2, :]
using CairoMakie
fig = Figure(size=(700, 350))
ax = Axis(fig[1, 1], xlabel="turn", ylabel="S_y")
lines!(ax, res.turns, s_y)
fig
```

   The vertical spin is partially flipped: it tends to a value of

```{code-cell} julia
s_y[end]
```

5. The resonance strength $\epsilon$ is computed from the invariant spin field $\hat{n}$ (see
   [Twiss](#twiss.spin)). Near an isolated resonance, at a distance $\Delta = \nu_0 - \kappa$ from it,
   $\hat{n}$ for a particle with an orbital amplitude is tilted away from its direction $\hat{n}_0$
   on the closed orbit by the angle $\theta$ with $\sin\theta = \epsilon/\sqrt{\epsilon^2 + \Delta^2}$,
   where $\epsilon$ is proportional to the amplitude. So for $|\Delta| \gg \epsilon$,
   $\epsilon = |\Delta|\,|\hat{n} - \hat{n}_0|$.

   Set $G\gamma = \kappa + 0.05$, and compute the invariant spin field as a Taylor series in the
   phase space coordinates (`as_taylor_series=true`, to first order) at the start of the ring:

```{code-cell} julia
Δ = 0.05
ring.E_ref = (κ + Δ)/G*m
twn = twiss(ring, spin=true, order=1, as_taylor_series=true, at=[1],
            cols=["nx", "ny", "nz", "beta2", "alpha2"])
```

   The derivatives of $\hat{n}$ with respect to $y$ and $p_y$ are coefficients of the Taylor series:
   `t[[0,0,1,0,0,0]]` is the coefficient of $y$. A particle with the vertical action $J_2$ is at
   $y = \sqrt{2J_2\beta_2}\cos\phi$, $p_y = -\sqrt{2J_2/\beta_2}\,(\sin\phi + \alpha_2\cos\phi)$. Its
   action is that of the tracked particle, which started at $y = y_0$, $p_y = 0$. The tilt of
   $\hat{n}$, averaged over the phase $\phi$:

```{code-cell} julia
β2, α2 = scalar(twn.beta2[1]), scalar(twn.alpha2[1])
J2 = y0^2*(1 + α2^2)/(2β2)
n = [twn.nx[1], twn.ny[1], twn.nz[1]]
dn_dy = [t[[0,0,1,0,0,0]] for t in n]
dn_dpy = [t[[0,0,0,1,0,0]] for t in n]
tilt = [sqrt(sum(abs2, dn_dy .* (sqrt(2J2*β2)*cos(ϕ)) .-
                       dn_dpy .* (sqrt(2J2/β2)*(sin(ϕ) + α2*cos(ϕ)))))
        for ϕ in range(0, 2pi, 37)[1:36]]
ϵ = Δ*sum(tilt)/36
```

6. The Froissart-Stora formula gives the vertical spin long after crossing an isolated resonance with
   a constant speed, for a spin that was vertical long before:

   $$
   S_y(\theta \to \infty) = 2\exp\left(-\frac{\pi\epsilon^2}{2\tilde{\alpha}}\right) - 1
   $$

   Here $\theta$ is the machine azimuth, $\epsilon$ is the resonance strength, and
   $\tilde{\alpha} = \frac{d}{d\theta} G\gamma = \frac{1}{2\pi}\times[\text{change in } G\gamma
   \text{ per turn}]$.

```{code-cell} julia
α̃ = 0.2/n_turns/(2pi)
2*exp(-pi*ϵ^2/(2α̃)) - 1
```

   This agrees with the tracking above to the accuracy of the approximations: the resonance is
   crossed from a finite distance, and $\epsilon$ was computed at a finite distance from the
   resonance.

## Exercises

1. **Siberian snakes:** The AGS has two partial Siberian snakes, which rotate the spin around the
   longitudinal axis: the cold snake (at `csnk`) by $18.3^\circ$ and the warm snake (at `wsnk`) by
   $10.57^\circ$. Model them as point-like rotations of the spin, with a `transport_map` (see
   [MapParams](../element.md)) that rotates the spin quaternion `q` by the unit quaternion
   $(\cos(\vartheta/2), \sin(\vartheta/2)\,\hat{e})$, which rotates the spin by the angle $\vartheta$
   around the unit vector $\hat{e}$. Compute the spin tune at different energies with
   `twiss(ring, spin=true)`, and verify that it is *not* the fractional part of $G\gamma$. Why is this
   the case? (Without the snakes, set the maps back to `nothing`.)

2. **Resonance spectrum:** Compute the resonance strengths $\epsilon$ as above for the resonances
   $\kappa = N \pm Q_2$ with $G\gamma$ between 30 and 60, for the same vertical amplitude. What do
   you observe about the spectrum? Can you explain any of its features? Hint: look at the structure
   of the lattice.

3. **Crossing speed:** Track through the resonance with different numbers of turns, from 100 to
   10000, and compare the final $S_y$ with the Froissart-Stora formula.
