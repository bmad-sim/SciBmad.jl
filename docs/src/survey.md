(survey)=
# Survey

A survey gives the position and orientation of the [branch coordinate system](#s:floor)
within the global `floor` coordinate system at the ends of every element of a branch or lattice.
It also gives the element [body coordinates](#s:lab.body.transform) at the ends of every element,
which differ from the branch coordinates when an element has alignment shifts
(`AlignmentParams`) or is a bend with a finite `tilt_ref`.

A survey is computed with the `survey` function:

```julia
survey(branch::Branch; floor0 = FloorCoords()) -> BranchSurvey
survey(lat::Lattice; floor0 = FloorCoords()) -> LatticeSurvey
```

The survey is a separate data structure: it is not stored in the `Branch` or `Lattice`, and it
is not updated when the lattice is changed. To get the new geometry after changing the lattice,
call `survey` again.

## Survey Data Structure

A survey is a tree with one `BranchSurvey` per branch and one `ElementSurvey` per element:

```
LatticeSurvey            name, branches
└── BranchSurvey         name, lattice_index, elements
    └── ElementSurvey    name, kind, index, s, s_downstream,
                         branch_entrance, branch_exit, body_entrance, body_exit
```

- A `LatticeSurvey` is indexed by the branch index or the branch name, e.g. `sv[2]` or
  `sv["ring"]`, and gives the `BranchSurvey` of that branch.
- A `BranchSurvey` is indexed in the same way as the `Branch` it is a survey of: `sv[i]` is the
  `ElementSurvey` of the element `branch[i]`. `lattice_index` is the index of the branch in
  the lattice, or `-1` if the branch is not in a lattice.
- An `ElementSurvey` has the element `name`, `kind`, and `index` in the branch, the longitudinal
  positions `s` and `s_downstream` of the entrance and exit ends, and four `FloorCoords`:
  the branch and body coordinates at the entrance and exit ends of the element.

A `FloorCoords` is the position and orientation of a coordinate frame in floor coordinates:

| Property | Description |
|:--|:--|
| `r` | Position `(x, y, z)` of the frame origin [m] |
| `q` | Orientation as a unit quaternion `(q0, qx, qy, qz)` |
| `x`, `y`, `z` | Components of `r` [m] |
| `theta`, `phi`, `psi` | [Floor orientation angles](#s:floor) [rad] |

Rotating a vector expressed in the frame by `q` gives the vector in floor coordinates. The
orientation in terms of the angles is `Ry(theta) * Rx(-phi) * Rz(psi)`, the same convention as
Bmad, so a positive `phi` tilts the `z`-axis toward `+Y`.
A `FloorCoords` can be constructed with `FloorCoords(r, q)` or `FloorCoords(r, theta, phi, psi)`.
`FloorCoords()` is at the origin with the frame axes along the floor axes.

## Example

```julia
using SciBmad

ring = Branch([
  Marker(name="start", species_ref=Species("electron"), pc_ref=1e9),
  SBend(name="b1", L=2, g_ref=pi/4),
  Quadrupole(name="q1", L=0.5, Kn1=0.3, x_offset=1e-3),
  SBend(name="b2", L=2, g_ref=pi/4),
]; name="ring")

lat = Lattice([ring])
sv = survey(lat)
```

Displaying a `BranchSurvey` shows a table of the branch coordinates at the exit end
of each element:

```julia
julia> sv["ring"]
BranchSurvey: ring
 Branch coordinates at the exit end of each element:
  Index   Name    Kind         s_downstream [m]   x [m]      y [m]   z [m]         theta [rad]   phi [rad]   psi [rad]
  1       start   Marker       0.0                0.0        0.0     0.0           0.0           0.0         0.0
  2       b1      SBend        2.0                -1.27324   0.0     1.27324       -1.5708       0.0         0.0
  3       q1      Quadrupole   2.5                -1.77324   0.0     1.27324       -1.5708       0.0         0.0
  4       b2      SBend        4.5                -3.04648   0.0     6.66134e-16   -3.14159      0.0         0.0
```

The quadrupole is offset by `x_offset = 1e-3` in its own frame. Since the quadrupole is
after a 90 degree bend, the body coordinates are shifted along the floor `z`-axis:

```julia
julia> e = sv["ring"][3];

julia> round.(e.body_exit.r - e.branch_exit.r; digits=12)
3-element StaticArraysCore.SVector{3, Float64} with indices SOneTo(3):
 0.0
 0.0
 0.001
```

## Branch Coordinates

The branch coordinates are constructed element by element as described in
[](#s:floor):
- An element without `BendParams` or `PatchParams` is straight: the exit frame is a distance `L`
  along the `z`-axis of the entrance frame.
- For an element with `BendParams`, the reference curve is a circular arc with curvature
  `g_ref`, bending in the plane rotated by `tilt_ref` about the `z`-axis.
- For an element with `PatchParams`, the exit frame is offset by `(dx, dy, dz)` in the
  entrance frame and rotated by `Ry(dy_rot) * Rx(dx_rot) * Rz(dz_rot)`.

For a single `Branch`, the branch coordinates at the entrance of the first element are
`floor0`. For a `Lattice`, the branches are surveyed in order. If the first element of a branch
is connected by a fork (`ForkParams`) to an element in an earlier branch, the branch starts at the
position and orientation of that element. Otherwise, the branch starts at `floor0`.

## Body Coordinates

The body coordinates are computed as described in [](#s:lab.body.transform). The alignment
offsets and rotations `Ry(y_rot) * Rx(x_rot) * Rz(tilt)` are applied at the center of the element,
which for a bend is the center of the chord. For a bend, the body coordinates are also rotated
by `tilt_ref` about the `z`-axis, so that the bend is in the body `x`-`z` plane. Elements with
`PatchParams` do not have alignment shifts, so their body coordinates are the branch coordinates.
