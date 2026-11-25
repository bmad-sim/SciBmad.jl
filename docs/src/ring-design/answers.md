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

(rd.answers)=
# Answers to Selected Exercises

```{code-cell} julia
:tags: [remove-cell]
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

```{code-cell} python
:tags: [remove-cell]
import numpy as np
import scibmad as sb
sb.options.display_width = 100
sb.options.display_height = 30
```

## Chapter 1: FODO Cells

**Exercises 1 to 3:** See the lattice file
[`lattices/01-fodo.jl`](https://github.com/bmad-sim/SciBmad.jl/tree/main/docs/src/ring-design/lattices/01-fodo.jl).

**Exercise 4:** For a FODO cell of thin quadrupoles with the focal length $f$ and the length $L$
between them (half the cell length $L_c = 2L$), the phase advance $\mu$ per cell is given by
$\sin(\mu/2) = L/(2f)$. For $\mu = 90°$ and $L = 9.21$ m, $f = L/\sqrt{2} = 6.513$ m, i.e. an
integrated strength $1/f = 0.1535$ m⁻¹, close to $K_1 L$ of the thick quadrupoles,
$0.3128 \cdot 0.5 = 0.1564$ m⁻¹. The beta functions at the quadrupoles are
$\beta_\pm = L_c (1 \pm \sin(\mu/2))/\sin\mu$: 31.44 m and 5.395 m, compared to 30.63 m and 5.549 m at
the beginnings of the thick quadrupoles:

```{code-cell} julia
using SciBmad
f = 9.21/sqrt(2)
@elements begin
  tqf = Multipole(Kn1L=1/f)
  tqd = Multipole(Kn1L=-1/f)
  dd = Drift(L=9.21)
end
thin = Beamline([tqf, dd, tqd, dd], species_ref=Species("electron"), pc_ref=17.846262612447e9)
tw = twiss(thin)
tw.q1, tw.q2, tw.beta1[tqf], tw.beta2[tqf]
```

```{code-cell} python
import numpy as np
import scibmad as sb
f = 9.21/np.sqrt(2)
with sb.elements():
    tqf = sb.Multipole(Kn1L=1/f)
    tqd = sb.Multipole(Kn1L=-1/f)
    dd = sb.Drift(L=9.21)
thin = sb.Beamline([tqf, dd, tqd, dd], species_ref=sb.Species("electron"),
                   pc_ref=17.846262612447e9)
tw = sb.twiss(thin)
tw.q1, tw.q2, tw.beta1[tqf], tw.beta2[tqf]
```

**Exercise 5:** The reverse cell is the forward cell in reverse order, started at another element.
Starting at another point of a periodic cell does not change its tunes, and neither does reversing
the order of its elements (the transfer matrix of the reversed cell has the same trace). So the
same quadrupole strengths give the same phase advances. The Twiss parameters at the start of the
cells are different, because the cells start at different points of the same periodic optics, and
the reversal changes the sign of α.

**Exercise 6:** The arc length is $L = L_c\,\theta/(2\sin(\theta/2))$:

```{code-cell} julia
θ = pi/132
6.86*θ/(2*sin(θ/2)) - 6.86 # [m]
```

```{code-cell} python
θ = np.pi/132
6.86*θ/(2*np.sin(θ/2)) - 6.86  # [m]
```

## Chapter 2: Dispersion Suppressor

**Exercise 3:** In a drift, $dx/ds = p_x/\sqrt{(1 + p_z)^2 - p_x^2 - p_y^2}$. Its derivative with
respect to δ at $p_x = p_y = 0$ (on the closed orbit of our ring) is
$d\eta_x/ds = \partial p_x/\partial\delta$, the column `dpx`. With an orbit, the two differ, by the
derivative of the denominator.

**Exercise 5:** `maxiter`, e.g. {jl}`match!(...; maxiter=100)`{py}`sb.match(..., maxiter=100)`.

## Chapter 4: Machine Coordinates

**Exercise 1:** A particle on the reference trajectory is at $-\Delta x$ from the axis of a
quadrupole offset by $\Delta x$. A focusing quadrupole ($K_1 > 0$) kicks it towards the axis of the
quadrupole: $p_x \approx +K_1 L \Delta x$.

**Exercise 3:** Through a patch with `dz` and `dy_rot`, the position moves by `dz` in the direction
$\theta$, and $\theta$ increases by `dy_rot`:

```{code-cell} julia
@elements begin
  b4 = SBend(L=0.5, angle=0.5)
  p4 = Patch(dz=1.0, dy_rot=pi/4)
  q4 = Quadrupole(L=0.6, Kn1=0.23)
end
patched = Beamline([b4, p4, q4], species_ref=Species("electron"), E_ref=10e6)
X, Z, θ = 0.0, 0.0, 0.0
for ele in patched.line
  if ele.kind == "Patch"
    lx, lz, dθ = 0.0, ele.dz, -ele.dy_rot
  elseif ele.g_ref == 0
    lx, lz, dθ = 0.0, ele.L, 0.0
  else
    g, L = ele.g_ref, ele.L
    lx, lz, dθ = (cos(g*L) - 1)/g, sin(g*L)/g, g*L
  end
  global X += lx*cos(θ) + lz*sin(θ)
  global Z += -lx*sin(θ) + lz*cos(θ)
  global θ -= dθ
end
X, Z, θ # Bmad: -0.43292, 1.93274, 0.28540
```

```{code-cell} python
with sb.elements():
    b4 = sb.SBend(L=0.5, angle=0.5)
    p4 = sb.Patch(dz=1.0, dy_rot=np.pi/4)
    q4 = sb.Quadrupole(L=0.6, Kn1=0.23)
patched = sb.Beamline([b4, p4, q4], species_ref=sb.Species("electron"), E_ref=10e6)
X, Z, θ = 0.0, 0.0, 0.0
for ele in patched.line:
    if ele.kind == "Patch":
        lx, lz, dθ = 0.0, ele.dz, -ele.dy_rot
    elif ele.g_ref == 0:
        lx, lz, dθ = 0.0, ele.L, 0.0
    else:
        g, L = ele.g_ref, ele.L
        lx, lz, dθ = (np.cos(g*L) - 1)/g, np.sin(g*L)/g, g*L
    X += lx*np.cos(θ) + lz*np.sin(θ)
    Z += -lx*np.sin(θ) + lz*np.cos(θ)
    θ -= dθ
X, Z, θ  # Bmad: -0.43292, 1.93274, 0.28540
```

## Chapter 8: Particle Phase Space Coordinates

**Exercise 1:** The phase space momenta $p_x$, $p_y$ and $p_z$ do not change through a drift, and
therefore have the same values at the end of the drift as at the beginning. Since $p_y$ is zero, $y$
stays zero. The horizontal momentum is $P_x = (1 + p_z)\,p_x P_0 = 0.4\,P_0$, not to be confused with
the phase space momentum $p_x = 0.2$. The longitudinal momentum is
$P_s = \sqrt{(1 + p_z)^2 - p_x^2}\,P_0 = 3.97994974842648\,P_0$. With the starting position $x_0 = 0$,
the position at the end is $x = x_0 + L P_x/P_s = 0.20100756305184242$ m, where $L$ is the length of
the drift.

The reference time at the end is $t_\mathrm{ref} = L/(\beta_\mathrm{ref} c) = 2/c$, with the reference
velocity approximated as the speed of light. The time of the particle is
$t = L\sqrt{1 + (P_x/P_s)^2}/(\beta c) = 2.010075630518424/c$, also with $\beta = 1$. So
$z = -\beta c\,(t - t_\mathrm{ref}) = -0.010075630518424$ m. The exact result of SciBmad differs
slightly, due to the approximation $\beta = \beta_\mathrm{ref} = 1$:

```{code-cell} julia
drift = Beamline([Drift(L=2.0)], species_ref=Species("proton"), E_ref=1e12)
track(drift, v0=[0 0.2 0 0 0 1.0]).v[turn=1, particle=1]
```

```{code-cell} python
drift = sb.Beamline([sb.Drift(L=2.0)], species_ref=sb.Species("proton"), E_ref=1e12)
sb.track(drift, v0=[[0, 0.2, 0, 0, 0, 1.0]]).v(turn=1, particle=1)
```

## Chapter 9: RF Cavities

**Exercise 2:** The cavity is short, so the particle crosses it at nearly the same time $t$ relative
to the reference particle, and $z = -\beta c\,(t - t_\mathrm{ref})$ changes by the ratio of the
velocities before and after the cavity:

```{code-cell} julia
p = Species("proton")
cav = RFCavity(L=1e-6, voltage=1e9, rf_frequency=1e9, zero_phase=PhaseRef.Accelerating)
lat = Beamline([cav], species_ref=p, pc_ref=1e9)
v = track(lat, v0=[0 0 0 0 1e-3 0]).v[turn=1, particle=1]
β(pc) = pc/sqrt(pc^2 + massof(p)^2)
v[5]/1e-3, β(1e9*(1 + v[6]))/β(1e9) # The ratio of z, and the ratio of the velocities
```

```{code-cell} python
p = sb.Species("proton")
cav = sb.RFCavity(L=1e-6, voltage=1e9, rf_frequency=1e9, zero_phase=sb.PhaseRef.Accelerating)
lat = sb.Beamline([cav], species_ref=p, pc_ref=1e9)
v = sb.track(lat, v0=[[0, 0, 0, 0, 1e-3, 0]]).v(turn=1, particle=1)
def β(pc):
    return pc/np.sqrt(pc**2 + sb.massof(p)**2)
v[4]/1e-3, β(1e9*(1 + v[5]))/β(1e9)  # The ratio of z, and the ratio of the velocities
```

The two ratios differ by $1.4\cdot10^{-4}$: inside the cavity, the particle and the reference particle
move with different velocities, which changes $t - t_\mathrm{ref}$ a little. This is proportional to
the length of the cavity: with `L = 1e-9`, the ratios agree to $1.4\cdot10^{-7}$.

## Chapter 11: Deferred Expressions and Contexts

**Exercise 1:**
{jl}`q.Kn1 = DefExpr(c -> 0.7*clamp(c.hh, 0, 1))`{py}`q.Kn1 = sb.DefExpr(lambda c: 0.7*np.clip(c.hh, 0, 1))`{pytext}` (np.clip is not traced into Julia: this function is called from Julia each time, which is slower)`

**Exercise 2:**

```{code-cell} julia
@elements begin
  dd = Drift(L=DefExpr(c -> 0.5 + c.ds))
  bb = SBend(L=0.5, angle=0.1)
  de = Drift(L=DefExpr(c -> 0.5 - c.ds))
end
moving = Beamline([dd, bb, de], species_ref=Species("electron"), E_ref=10e6)
moving.context.ds = 0.1
moving[bb].s, sum(ele.L for ele in moving.line)
```

```{code-cell} python
with sb.elements():
    dd = sb.Drift(L=sb.DefExpr(lambda c: 0.5 + c.ds))
    bb = sb.SBend(L=0.5, angle=0.1)
    de = sb.Drift(L=sb.DefExpr(lambda c: 0.5 - c.ds))
moving = sb.Beamline([dd, bb, de], species_ref=sb.Species("electron"), E_ref=10e6)
moving.context.ds = 0.1
moving[bb].s, sum(ele.L for ele in moving.line)
```

**Exercise 3:** With a variable for the offset, e.g.
{jl}`q.x_offset = DefExpr(c -> c.girder_dx)`{py}`q.x_offset = sb.DefExpr(lambda c: c.girder_dx)` and
{jl}`b.x_offset = DefExpr(c -> c.girder_dx)`{py}`b.x_offset = sb.DefExpr(lambda c: c.girder_dx)`. (A
girder also moves its elements when it is rotated, which
this does not describe.)

## Chapter 13: Comparing Lattices

**Exercise 2:** {jl}`setit(k) = (q.Ks0L = k)`{py}`def setit(k): q.Ks0L = k`

## Chapter 15: Spin Tracking with Ramping

**Exercise 1:** The rotation around the longitudinal axis $\hat{z}$ by the angle $\vartheta$ is the
quaternion $r = (\cos(\vartheta/2), 0, 0, \sin(\vartheta/2))$. The map multiplies the spin
quaternion `q` of the particle by it, $r\,q$:

```{code-cell} julia
include("lattices/15-spin.jl")
function snake(v, q, p)
  q === nothing && return v, q # Without spin tracking
  c, s = cos(p/2), sin(p/2)
  return v, (c*q[1] - s*q[4], c*q[2] - s*q[3], c*q[3] + s*q[2], c*q[4] + s*q[1])
end
ring[csnk].transport_map = snake
ring[csnk].transport_map_params = deg2rad(18.3)
ring[wsnk].transport_map = snake
ring[wsnk].transport_map_params = deg2rad(10.57)
for Gγ in (45.5, 46.0, 51.82)
  ring.E_ref = Gγ/G*m
  println("Gγ = ", Gγ, ": spin tune ", twiss(ring, spin=true, at=[]).qspin)
end
```

```{code-cell} python
from lattices.ch15_spin import *
# A transport map runs inside the tracking (for every particle, also on GPUs): it is Julia code
snake = sb.julia.seval("""
function snake(v, q, p)
  q === nothing && return v, q # Without spin tracking
  c, s = cos(p/2), sin(p/2)
  return v, (c*q[1] - s*q[4], c*q[2] - s*q[3], c*q[3] + s*q[2], c*q[4] + s*q[1])
end""")
ring[csnk].transport_map = snake
ring[csnk].transport_map_params = np.deg2rad(18.3)
ring[wsnk].transport_map = snake
ring[wsnk].transport_map_params = np.deg2rad(10.57)
for Gγ in (45.5, 46.0, 51.82):
    ring.E_ref = Gγ/G*m
    print(f"Gγ = {Gγ}: spin tune", sb.twiss(ring, spin=True, at=[]).qspin)
```

With the snakes, the closed-orbit spin is no longer vertical everywhere, and its spin tune is not
$G\gamma$: away from the integers, it is shifted towards the half integer, and at integer values of
$G\gamma$ it does not reach 0, so that these resonances are avoided.
