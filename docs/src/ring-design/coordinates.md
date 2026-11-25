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

(rd.coords)=
# 4. Machine Coordinates

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

## Coordinate Systems

```{figure} ../figures/coordinates.svg
:width: 70%

The branch coordinates follow the bends of the machine, and each element has element body
coordinates, which are the same as the branch coordinates unless the element is misaligned.
```

As explained in the [Coordinates](../coordinates.md) chapter, three coordinate systems are used
to describe the positions of the lattice elements:

**Global coordinates.** The $(X, Y, Z)$ global (also called floor) coordinate system (notice that
capital letters are used for the coordinate axes) is independent of the machine and is "attached"
to the building the accelerator is in. Typically, the $Y$ axis points vertically up and $(X, Z)$
is the horizontal plane.

**Branch coordinates.** The global coordinate system is not convenient for describing where
particles are as they move through the lattice. For this there is the branch (in Bmad also called
local, laboratory, or reference) curvilinear coordinate system. Branch coordinates are also used
to describe the nominal position (without any "misalignments") of the lattice elements. They start
with a curve, often called the "reference trajectory". The distance along the reference trajectory
is denoted by $s$. At each point along the reference trajectory, there is a Cartesian $(x, y, z)$
coordinate system with its origin at the reference trajectory point. The $z$ axis is always
tangent to, and the $x$ and $y$ axes are always transverse to, the reference trajectory.

**Element body coordinates.** Elements can be shifted ("misaligned") from their nominal position.
To describe things like electric and magnetic fields or apertures, which depend on the actual
position of an element, element body coordinates are used. They are attached to the physical
element. Without misalignments, the element body coordinates are the branch coordinates.

## Element Geometry Types

```{figure} ../figures/ele-coord-frame.svg
:width: 70%

A) Straight elements, B) elements with a bend, C) elements with a patch. All elements have an
entrance and an exit coordinate frame.
```

All lattice elements have an "entrance" end and an "exit" end. A particle enters the element at
the entrance end and exits at the exit end. Elements have one of three geometries, based on how
the branch coordinates transform from the entrance end of the element to the exit end:

**Straight geometry.** Used for elements like drifts and quadrupoles. The reference trajectory is
a straight line and the orientation of the $(x, y, z)$ axes is the same at every point inside the
element.

**Bend geometry.** Used for elements with a bend (a nonzero curvature `g_ref` of the reference
trajectory, e.g. an `SBend`). The reference trajectory is an arc of a circle. The $(x, y, z)$
coordinates rotate about an axis which is perpendicular to the $z$ axis. Normally (with
`tilt_ref = 0`) the rotation axis is parallel to the $y$ axis, so that the orientation of the
$y$ axis does not change.

**Patch geometry.** Used for elements with patch parameters (`dx`, `dz_rot`, ...). The exit
coordinates can be arbitrarily positioned with respect to the entrance coordinates (see
[Patch Elements](#rd.patch) below).

The branch coordinate system is constructed by taking the ordered list of lattice elements and
connecting the exit frame of one element to the entrance frame of the next, just like LEGO blocks.

## Branch Coordinates Relative to Global Coordinates

For any given point $s$ on the reference trajectory, the branch coordinate system is described with
respect to the global coordinate system by six parameters: the global position $(X, Y, Z)$, and
three angles: the azimuth angle $\theta$ in the $(X, Z)$ plane, the elevation angle $\phi$ and the
roll angle $\psi$. At the beginning of the beamline ($s = 0$), the branch coordinates are aligned
with the global coordinates, and all three angles are zero. For a machine that lies in the
horizontal plane, $\phi$ and $\psi$ are zero everywhere.

SciBmad does not compute global coordinates yet. For a horizontal machine they are easy to
compute, though, with the equations of [Lattice Element Positioning](#s:ele.pos): through a
straight element of length $L$ the position moves by $L$ in the direction $\theta$, and through a
bend with the bend angle $\alpha = g L$ and the radius $\rho = 1/g$ it moves by
$(\rho(\cos\alpha - 1), \rho\sin\alpha)$ in the $(x, z)$ coordinates at its entrance, and
$\theta$ decreases by $\alpha$. As a loop through the elements of a beamline:

```{code-cell} julia
using SciBmad

@elements begin
  d = Drift(L=0.5)
  b = SBend(L=0.5, g_ref=1.0, Kn0=1.001, e1=0.1) # g_ref = 1/ρ: the curvature of the reference
  q = Quadrupole(L=0.6, Kn1=0.23)
end
simple = Beamline([d, b, q], species_ref=Species("electron"), E_ref=10e6)

X, Z, θ = 0.0, 0.0, 0.0
for ele in simple.line
  L = ele.L
  g = ele.g_ref
  # Displacement through the element in its entrance coordinates (x, z)
  if g == 0
    lx, lz = 0.0, L
  else
    lx, lz = (cos(g*L) - 1)/g, sin(g*L)/g
  end
  # Rotate by θ into global coordinates
  global X += lx*cos(θ) + lz*sin(θ)
  global Z += -lx*sin(θ) + lz*cos(θ)
  global θ -= g*L
  println(rpad(ele.name, 3), " X = ", round(X, digits=5), "  Z = ", round(Z, digits=5),
          "  θ = ", round(θ, digits=5))
end
```

```{code-cell} python
import numpy as np
import scibmad as sb

with sb.elements():
    d = sb.Drift(L=0.5)
    # g_ref = 1/ρ: the curvature of the reference
    b = sb.SBend(L=0.5, g_ref=1.0, Kn0=1.001, e1=0.1)
    q = sb.Quadrupole(L=0.6, Kn1=0.23)
simple = sb.Beamline([d, b, q], species_ref=sb.Species("electron"), E_ref=10e6)

X, Z, θ = 0.0, 0.0, 0.0
for ele in simple.line:
    L = ele.L
    g = ele.g_ref
    # Displacement through the element in its entrance coordinates (x, z)
    if g == 0:
        lx, lz = 0.0, L
    else:
        lx, lz = (np.cos(g*L) - 1)/g, np.sin(g*L)/g
    # Rotate by θ into global coordinates
    X += lx*np.cos(θ) + lz*np.sin(θ)
    Z += -lx*np.sin(θ) + lz*np.cos(θ)
    θ -= g*L
    print(f"{ele.name:3s} X = {round(X, 5)}  Z = {round(Z, 5)}  θ = {round(θ, 5)}")
```

These are the global coordinates at the exit end of each
element.{jltext}` (In a script, the `{jl}`global`{jltext}`
keyword is needed to change a variable defined outside of a loop.)`

## Element Misalignments

```{figure} figures/pitch.svg
:width: 62%

Effect of an offset and a pitch on a straight element. The `x_pitch` of Bmad is the `y_rot` of
SciBmad.
```

```{figure} figures/tilt.svg
:width: 33%

Effect of a tilt on a straight element.
```

Once the branch coordinates are established, the position of any physical element can be shifted
("misaligned"). The parameters that determine a misalignment are in the
[`AlignmentParams`](#alignment.params) of the element:

- `x_offset`, `y_offset`, `z_offset` offset the element in the $x$, $y$, and $z$ directions.
- `y_rot` rotates the element around the $+y$ axis: a `y_rot` of $\pi/2$ would rotate the element so
  that the body $+z$ axis points along the $+x$ axis of the branch coordinates. (This is Bmad's
  `x_pitch`.)
- `x_rot` rotates the element around the $+x$ axis: an `x_rot` of $\pi/2$ would point the body $+z$
  axis along the $-y$ axis. (Bmad's `y_pitch` is `-x_rot`.)
- `tilt` rotates the element around the $+z$ axis.

The offsets and rotations are defined with respect to the center of the element. For bends, see
[Bend Element Misalignment Transformation](#s:bend.mis).

A misaligned quadrupole kicks a particle that goes through it on the reference trajectory. For
example, a quadrupole offset horizontally by 1 mm and rotated by 0.4 mrad:

```{code-cell} julia
qm = Quadrupole(L=1.0, Kn1=0.23, x_offset=1e-3, y_rot=4e-4)
misalign = Beamline([qm], species_ref=Species("electron"), E_ref=10e6)
res = track(misalign, v0=zeros(1, 6))
res.v[turn=1, particle=1] # x, px, y, py, z, pz at the exit
```

```{code-cell} python
qm = sb.Quadrupole(L=1.0, Kn1=0.23, x_offset=1e-3, y_rot=4e-4)
misalign = sb.Beamline([qm], species_ref=sb.Species("electron"), E_ref=10e6)
res = sb.track(misalign, v0=np.zeros((1, 6)))
res.v(turn=1, particle=1)  # x, px, y, py, z, pz at the exit
```

The particle comes out of the quadrupole displaced and with a horizontal momentum, as if kicked by a
dipole. `track` is described in [Track](../track.md), and the phase space coordinates in
[Particle Phase Space Coordinates](phase-space.md).

(rd.patch)=
## Patch Elements

```{figure} figures/patch.svg
:width: 48%

The exit coordinates of a patch are set by its offsets and rotations. (In Bmad, `x_offset`,
`y_offset`, `z_offset`, `x_pitch`, `y_pitch` and `tilt`; in SciBmad, `dx`, `dy`, `dz`, `dy_rot`,
`dx_rot` and `dz_rot`.)
```

Patch elements are used to shift the reference trajectory. As a consequence, the nominal placements
of all elements downstream of the patch are affected. This is useful for simulating things like
injection or extraction lines, where the patch is used to reorient the reference trajectory so that
it follows the injection or extraction line.

The parameters of a patch are in its [`PatchParams`](#patch.params). They set the placement of the
exit frame relative to the entrance frame:

1. Initially the exit coordinates coincide with the entrance coordinates.
2. The origin of the exit coordinates is translated by (`dx`, `dy`, `dz`).
3. The rotations are applied: `dy_rot` rotates the $+z$ axis towards the $+x$ axis (a rotation around
   the $+y$ axis), and `dx_rot` rotates the $+z$ axis towards the $-y$ axis (a rotation around the $+x$
   axis).
4. `dz_rot` rotates the exit coordinates around their $+z$ axis.

A patch represents a field-free space, so a particle travels through a patch as in a drift. The
difference is that in a patch there is a coordinate transformation from the entrance coordinates to
the exit coordinates. For example, with a patch that moves the exit frame by 1 m along $z$ and
rotates it by 45°:

```{code-cell} julia
p = Patch(dz=1.0, dy_rot=pi/4)
patched = Beamline([p], species_ref=Species("electron"), E_ref=10e6)
res = track(patched, v0=[1e-3 0 0 0 0 0])
res.v[turn=1, particle=1]
```

```{code-cell} python
p = sb.Patch(dz=1.0, dy_rot=np.pi/4)
patched = sb.Beamline([p], species_ref=sb.Species("electron"), E_ref=10e6)
res = sb.track(patched, v0=[[1e-3, 0, 0, 0, 0, 0]])
res.v(turn=1, particle=1)
```

The particle started at $x$ = 1 mm moving along the entrance $z$ axis. In the exit coordinates it
is at $x = \sqrt{2}$ mm and moves with $p_x = -\sin(45°)$.

## Exercises

1. **Quadrupole offsets:** A quadrupole of strength $K_1$ and length $L$, offset horizontally by
   $\Delta x$, kicks a particle on the reference trajectory approximately like a dipole with the
   kick $K_1 L \Delta x$. Check this with `track`, for a short quadrupole and small offsets. Which
   sign does the kick have?

2. **Tilt:** Track a particle that starts with $x$ = 1 mm through the quadrupole `qm` above, after
   setting its tilt with `qm.tilt = 0.1`. Can you predict the change of $y$ and $p_y$? What would a
   tilt of $\pi/4$ do?

3. **Global coordinates with a patch:** Extend the loop above to patches that only have `dz` and
   `dy_rot`, and compute the global coordinates at the exit ends of the elements of the line
   `[b, p, q]` with `b = SBend(L=0.5, angle=0.5)`, the patch `p` above and
   `q = Quadrupole(L=0.6, Kn1=0.23)`. Check your answer: Bmad gives at the exit of `q`
   $X = -0.43292$, $Z = 1.93274$, $\theta = 0.28540$.
