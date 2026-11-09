(c:coords)=
# Coordinates

%---------------------------------------------------------------------------------------------------
(s:phase.space)=
## Phase Space Coordinate System

In construction...

%---------------------------------------------------------------------------------------------------
(s:nomen)=
## Nomenclature

A `bend` element is any element that has a non-zero reference bend strength `g_ref`. 
A `patch` element is any element that has a non-zero `PatchParams` parameters.
No element is allowed to be both a bend and a patch element at the same time.

%---------------------------------------------------------------------------------------------------
(s:coords)=
## Coordinate Systems Used to Describe a Machine

```{figure} figures/coordinates.svg
:width: 70%
:name: f:coords
:class: caption-left

The coordinate systems to describe a machine: The "branch" curvilinear coordinate system follows the bends
of the accelerator. The "branch reference curve" is the {math}`x = y = 0` curve of the curvilinear coordinate
system. Each lattice element has "element body" coordinates which, if the element has no
alignment shifts (not "misaligned"), is the same as the branch coordinates.
```

%---------------------------------------------------------------------------------------------------

SciBmad uses two coordinate systems to describe a machine as illustrated in the figure above. 
For each lattice branch, a `branch` coordinate system is used. This curvilinear coordinate
system defines the nominal position of the lattice elements.

The `branch reference curve` is the {math}`x = y = 0` curve of the curvilinear coordinate system. 
The branch reference curve does not have to be continuous and, in particular, the coordinate
system through an element with `PatchParams` parameters will generally be discontinuous. 
If there are no `BendParams` parameters with a finite
`tilt_ref`, the branch reference curve will be in the {math}`(x,z)` plane with {math}`y = 0`. Since 
most machines are essentially horizontal, the {math}`x` coordinate is typically thought of as the
"horizontal" coordinate and the {math}`y` coordinate is typically thought of as the "vertical"
coordinate.
 
The "nominal" position of a lattice element is the position of the element without any
[alignment shifts](#alignment:params)
(these position and orientation shifts which are sometimes referred to as "misalignments"). 
Each lattice element has "`element body`"
coordinates which are attached to the physical element, and the electric and magnetic
fields of an element are described with respect to `body` coordinates.  
If an element has no
alignment shifts, the `body` coordinates of the element are aligned with the 
`branch` coordinates.
The transformation between `branch` and `body` coordinates is given in
[xxx](#s:lab.body.transform).

%---------------------------------------------------------------------------------------------------
(s:ent.exi)=
## Element Entrance and Exit Coordinates

```{figure} figures/ele-coord-frame.svg
:width: 70%
:name: f:ele.coord.frame
:class: caption-left

Lattice elements can be imagined as "LEGO blocks" which
fit together to form the branch coordinate system. How elements
join together is determined in part by their entrance and exit coordinate frames. A) For
straight line elements the entrance and exit frames are colinear. B) For elements with a
`BendParams` parameter group, the 
transformation from entrance to exit coordinates is a rotation about the bend center of curvature.
C) For elements with `PatchParams` parameters, the 
exit frame may be arbitrarily positioned with respect to the entrance frame.
```

%---------------------------------------------------------------------------------------------------

As discussed in the next section, the branch coordinate system is constructed starting with the first
element in a branch and then building up the coordinate system element-by-element.
Elements have an "`entrance`" and an "`exit`" coordinate frame as
illustrated in the above figure.
These coordinate frames are attached to the element and are part of the `element body coordinates`. 

Most element kinds have a "straight" geometry as shown in
{numref}`f:ele.coord.frame`A. That is, the reference curve through the element is a straight line
segment with the {math}`x` and {math}`y` axes always pointing in the same direction.
For an element with [BendParams](#bend:params) parameters, 
the reference curve is a segment of a circular arc as shown in
{numref}`f:ele.coord.frame`B. With the `tilt_ref` parameter set to zero, the rotation axis
between the entrance and exit frames is parallel to the {math}`y`-axis ([xxx](#s:floor)).
For [Patch](#patch:params)
elements ({numref}`f:ele.coord.frame`C), the exit face can be
arbitrarily oriented with respect to the entrance end.
For elements with a `PatchParams` group, the interior reference curve 
is dependent upon the patch parameter settings ([xxx](#patch:params)) and, in general,
will have a discontinuity.

%---------------------------------------------------------------------------------------------------
(s:ref.construct)=
## Branch Coordinate Construction

The construction of a branch coordinate system starts at the first element of the branch,
where {math}`s` is zero.

Starting from the beginning element, succeeding elements are concatenated together
to form the branch coordinate system. All elements have an "`upstream`" and a "`downstream`"
end. 
The `downstream` end of an element is always farther (at greater {math}`s`-position) 
from the beginning element than the `upstream` end of the element. Normally, particles will travel
in the {math}`+s` direction, so particles will enter an element at the upstream end and exit at the
downstream end.

The element's `upstream` end is the same as the element's `entrance` end 
({numref}`f:ele.coord.frame`) and the `downstream` end is the same 
as the element's `exit`.

The procedure to connect elements together to form the branch coordinates is to ignore 
`element body` alignment shifts and mate the `downstream` reference frame of
the element with the `upstream` reference frame of the next element in
the branch so that the {math}`(x,y,z)` coordinate axes coincide.
If there are body alignment shifts, the `entrance` and `exit` frames will move with the element. 
However, this does not affect the branch coordinate system.

%---------------------------------------------------------------------------------------------------
(s:floor)=
## Coordinate Frames

Following MAD, the position of a coordinate frame within a fixed Cartesian coordinate system
with axes {math}`(X, Y, Z)` is characterized by a vector {math}`\bf V`
```{math}
  {\bf V} = 
  \begin{pmatrix}
    X \\ Y \\ Z 
  \end{pmatrix}
```
 
The orientation of a coordinate frame is described by a unitary rotation matrix {math}`\bf W`. 
The column vectors of {math}`\bf W` are the unit vectors spanning the branch coordinate axes in
the order {math}`(x, y, z)`. {math}`\bf W` can be expressed in terms of the
three angles {math}`\theta`, {math}`\phi`, and {math}`\psi` via the formula
```{math}
:label: www
  {\bf W} &= {\bf R}_{y}(\theta) \; {\bf R}_{x}(\phi) \; {\bf R}_{z}(\psi) \\
  &= \begin{pmatrix}
    \cos\theta \cos\psi + \sin\theta \sin\phi \sin\psi & 
        -\cos\theta \sin\psi + \sin\theta \sin\phi \cos\psi & 
         \sin\theta \cos\phi \\
    \cos\phi \sin\psi & \cos\phi \cos\psi & -\sin\phi \\
    \cos\theta \sin\phi \sin\psi - \sin\theta \cos\psi & 
         \sin\theta \sin\psi + \cos\theta \sin\phi \cos\psi & 
         \cos\theta \cos\phi 
  \end{pmatrix}
```
where
```{math}
:label: wtt0t
  {\bf R}_{y}(\theta) &= 
  \begin{pmatrix}
    \cos\theta  & 0 & \sin\theta \\
    0           & 1 & 0          \\
    -\sin\theta & 0 & \cos\theta 
  \end{pmatrix}, \\
  {\bf R}_{x}(\phi) &= 
  \begin{pmatrix}
    1 & 0 & 0                \\
    0 & \cos\phi & -\sin\phi \\
    0 & \sin\phi &  \cos\phi 
  \end{pmatrix}, \\
  {\bf R}_{z}(\psi) &= 
  \begin{pmatrix}
    \cos\psi & -\sin\psi & 0 \\
    \sin\psi &  \cos\psi & 0 \\
    0        &  0        & 1                
  \end{pmatrix}
```
Notice that these are Tait-Bryan angles and not Euler angles.

An alternative representation of the {math}`\bf W` matrix (or any other rotation matrix) is to specify the
axis {math}`\bf u` (normalized to 1) and angle of rotation {math}`\beta`
```{math}
:label: wctux2
  {\bf W} = \begin{pmatrix}
    \cos \beta + u_x^2 \left(1 - \cos \beta \right) & 
    u_x \, u_y \left(1 - \cos \beta \right) - u_z \sin \beta & 
    u_x \, u_z \left(1 - \cos \beta \right) + u_y \sin \beta \\ 
    u_y \, u_x \left(1 - \cos \beta \right) + u_z \sin \beta & 
    \cos \beta + u_y^2\left(1 - \cos \beta \right) & 
    u_y \, u_z \left(1 - \cos \beta \right) - u_x \sin \beta \\ 
    u_z \, u_x \left(1 - \cos \beta \right) - u_y \sin \beta & 
    u_z \, u_y \left(1 - \cos \beta \right) + u_x \sin \beta & 
    \cos \beta + u_z^2\left(1 - \cos \beta \right)
  \end{pmatrix}
```

%---------------------------------------------------------------------------------------------------
(s:ele.pos)=
## Lattice Element Positioning

The lattice standard, again following MAD, computes {math}`\bf V` and {math}`\bf W` 
by starting at the first element of the lattice and iteratively using the equations
```{math}
:label: wws
\begin{align}
  {\bf V}_1 &= {\bf W}_0 \; {\bf L} + {\bf V}_0, 
    \\
  {\bf W}_1 &= {\bf W}_0 \; {\bf S}
\end{align}
```
where {math}`({\bf V}_0, {\bf W}_0)` and {math}`({\bf V}_1, {\bf W}_1)` are respectively
the position and orientation of the branch coordinates
at the entrance and exit ends of an element,
{math}`{\bf L}` is the displacement vector through the element, 
and matrix {math}`{\bf S}` is the rotation of
the branch coordinate system through the element.
For all elements whose reference curve
through them is a straight line, the corresponding {math}`\bf L` and {math}`\bf S` are
```{math}
:label: l00l
  {\bf L} = 
  \begin{pmatrix}
      0 \\ 0 \\ L
  \end{pmatrix},
  \quad
  {\bf S} = 
  \begin{pmatrix}
      1 & 0 & 0 \\ 
      0 & 1 & 0 \\
      0 & 0 & 1
  \end{pmatrix},
```
Where {math}`L` is the length of the element. 

%---------------------------------------------------------------------------------------------------

```{figure} figures/tilt-bend.svg
:width: 80%
:name: f:tilt.bend
:class: caption-left

A) Rotation axes (bold arrows) for four different `tilt_ref` angles of {math}`\theta_{tr} = 0`, 
{math}`\pm \pi/2`, and {math}`\pi`. 
{math}`(x_0, y_0, z_0)` are the branch coordinates at the entrance end of the bend with
the {math}`z_0` axis being directed into the page. Any rotation axis will be displaced by a distance of
the bend radius `rho` from the origin. B) The {math}`(x, y, z)` coordinates at the exit end of the bend
for the same four `tilt_ref` angles. In this case the bend angle is taken to be {math}`\pi/2`.
```

%---------------------------------------------------------------------------------------------------

For a `bend`, the axis of rotation is dependent upon the bend's [`tilt_ref`](#bend:params) angle
as shown in {numref}`f:tilt.bend`A. The axis of rotation points in the negative {math}`y_0`
direction for `tilt_ref` = 0 and is offset by the bend radius `rho`. Here {math}`(x_0, y_0, z_0)`
are the branch coordinates at the entrance end of the bend with the {math}`z_0` axis being directed into
the page in the figure.  For a non-zero `tilt_ref`, the rotation axis is itself rotated about the
`z_0` axis by the value of `tilt_ref`. {numref}`f:tilt.bend`B shows the exit coordinates for four
different values of `tilt_ref` and for a bend angle `angle` of {math}`\pi/2`.

For a bend, {math}`\bf S` is given using Eq. [](#wctux2) with 
```{math}
:label: ustt
\begin{align}
  {\bf u} &= (\sin\theta_{tr}, -\cos\theta_{tr}, 0) \\
  \beta &= \alpha_b
\end{align}
```
where {math}`\theta_{tr}` is the `tilt_ref` angle. The {math}`\bf L` vector for a `bend` is given by 
```{math}
:label: lrztt
  {\bf L} = {\bf R}_{z}(\theta_{tr}) \; {\bf \tilde L}, \quad
  {\bf \tilde L} = 
  \begin{pmatrix}
    \rho (\cos\alpha_b - 1) \\ 0 \\ \rho \, \sin\alpha_b
  \end{pmatrix}
```
where {math}`\alpha_b` is the bend [angle](#bend:params) and {math}`\rho` being the bend radius
({math}`\rho`). Notice that since {math}`\bf u` is perpendicular to {math}`z`, the curvilinear reference coordinate
system has no "torsion". Note: The branch coordinate system can be related to a Frenet-Serret coordinate system, but 
the two coordinate systems do not coincide.
Frenet-Serret coordinates use the radial direction in a bend as a coordinate axis while
with branch coordinates the radial direction can point anywhere in the {math}`(x,y)` plane.

Note: An alternative equation for {math}`\bf S` for a bend is
```{math}
:label: srrr
  {\bf S} = {\bf R}_{z}(\theta_{tr}) \; {\bf R}_{y}(-\alpha_b) \; {\bf R}_{z}(-\theta_{tr})
```
The bend transformation above is so constructed that the transformation is equivalent to rotating
the branch coordinate system around an axis that is perpendicular to the plane of the bend. This
rotation axis is invariant under the bend transformation. 
For example, for {math}`\theta_{tr} = 0` (or {math}`\pi`) the {math}`y`-axis is
the rotation axis and the {math}`y`-axis of the branch coordinates before the bend will be
parallel to the {math}`y`-axis of the branch coordinates after the bend as shown in {numref}`f:tilt.bend`. 
That is, a lattice with only bends with {math}`\theta_{tr} = 0` or {math}`\pi` will lie in the 
horizontal plane (this assuming that the {math}`y`-axis starts out pointing along 
the {math}`Y`-axis as it does by default).
For {math}`\theta_{tr} = \pm\pi/2`, the bend axis is the {math}`x`-axis. 
A value of {math}`\theta_{tr} = +\pi/2` represents a downward pointing bend.


%---------------------------------------------------------------------------------------------------
(s:position.transform)=
## Particle Position Transformation

A point {math}`{\bf Q}_g = (X, Y, Z)` defined in the global coordinate system, when expressed in the
coordinate system defined by {math}`({\bf V}, {\bf W})` is
```{math}
:label: rwrv
  {\bf Q}_{VW} = {\bf W}^{-1} \left( {\bf Q}_g - {\bf V} \right)
```
This is essentially the inverse of Eq. {eq}`wws`. That is, position vectors transform inversely to the
transformation of the coordinate system.

Using Eq. {eq}`rwrv` with Eqs. {eq}`wws`, the transformation of a 
particle's position {math}`{\bf q} = (x,y,z)` and momentum {math}`{\bf P} = (P_x, P_y, P_z)` 
when the coordinate frame is transformed from frame
{math}`({\bf V}_{i-1}, {\bf W}_{i-1})` to frame {math}`({\bf V}_i, {\bf W}_i)` is
\begin{align}
{\bf q}_i &= {\bf S}_i^{-1} \, \left( {\bf q}_{i-1} - {\bf L}_i \right),
\label{rwlr} \\
{\bf P}_i &= {\bf S}_i^{-1} \, {\bf P}_{i-1}
\label{pps}
\end{align}

Notice that since {math}`{\bf S}` (and {math}`{\bf W}`) is the product of orthogonal 
rotation matrices, {math}`{\bf S}` is itself orthogonal and its inverse is just the transpose
```{math}
  {\bf S}^{-1} = {\bf S}^T
```

%---------------------------------------------------------------------------------------------------
(s:lab.body.transform)=
## Transformation Between Branch and Element Body Coordinates

The `element body` coordinate system is the coordinate system attached to an element. Without any
alignment shifts, the [`branch` coordinates](#s:ref.construct) and `element body` coordinates
are the same. With alignment shifts, the transformation between `branch` and `element body`
coordinates depends upon whether the branch coordinate system is [straight](#s:straight.mis) or
[bent](#s:bend.mis).

When tracking a particle through an element, the particle starts at the [`nominal`](#s:coords) 
upstream end of the element with the particle's position expressed in branch
coordinates. Tracking from the nominal upstream end to the actual upstream face of the element
involves first transforming to element body coordinates (with {math}`s = 0` in the equations below) and
then propagating the particle as in a field free drift space from the particle's starting position
to the actual element face. Depending upon the element's orientation, this tracking may involve
tracking backwards. Similarly, after a particle has been tracked through the physical element to the
actual downstream face, the tracking to the nominal downstream end involves transforming to
branch coordinates (using {math}`s = L` in the equations below) and then propagating the particle as
in a field free drift space to the nominal downstream edge.

%---------------------------------------------------------------------------------------------------
(s:straight.mis)=
### Straight Element Misalignment Transformation

For straight line elements, given a branch coordinate frame {math}`\Lambda_s` with an origin a distance
{math}`s` from the beginning of the element, misalignments will shift the coordinates to a new reference
frame denoted {math}`E_s`. Since misalignments are defined with respect to the middle of the element, the
transformation from {math}`\Lambda_s` and {math}`E_s` can be accomplished using a three step process:
```{math}
:label: llee
  \Lambda_s \longrightarrow \Lambda_\text{mid} 
  \longrightarrow E_\text{mid} \longrightarrow E_s
```
where {math}`\Lambda_\text{mid}` and {math}`E_\text{mid}` are the branch and element reference frames at the
center of the element. All transformations use Eqs. [](#wws).

The transformations are:
1. {math}`\Lambda_s \longrightarrow \Lambda_\text{mid} `: Translation to the element midpoint.
For this transformation, {math}`\bf S` is the unit matrix and {math}`{\bf L} = (0, 0, L/2 - s)`.

1. {math}`\Lambda_\text{mid} \longrightarrow E_\text{mid}`: Misalignment transformation with
```{math}
:label: swww2


\bf L =
\begin{pmatrix}
\text{x_offset} \\ \text{y_offset} \\ \text{z_offset}
\end{pmatrix},
\qquad
\bf S = {\bf R}_y (\text{y_rot}) \; {\bf R}_x (\text{x_rot}) \; {\bf R}_z (\text{tilt})

```

1. {math}`E_\text{mid} \longrightarrow E_s`: Translation to body coordinates {math}`E_s`.
For this transformation, {math}`\bf S` is the unit matrix and {math}`{\bf L} = (0, 0, s - L/2)`.

Notice that with this definition of how elements are misaligned, the position of the center of a
non-bend misaligned element depends only on the offsets, and is independent of the pitches and tilt.

%-----------------------------------------------------------------------------
(s:bend.mis)=
### Bend Element Misalignment Transformation

For `Bend` element positioning, besides the standard `AlignmentParams` parameters, there is the
`tilt_ref` ({math}`\theta_{tr}`) parameter (see [](#bend:params)). 
The latter affects both the reference orbit and the bend position. 
Furthermore, `tilt_ref` is calculated with respect to
the coordinates at the beginning of the bend while, like straight elements, `tilt`, offsets, and
pitches are calculated with respect to the center of the bend. The different reference frame used
for `tilt_ref` versus everything else means that six transformations are needed to get from the
branch frame to the element body frame (see Eq. [](#llee)). Symbolically:
```{math}
  \Lambda_s \longrightarrow \Lambda_\text{mid-arc} \longrightarrow
  \Omega_\text{mid-chord} \longrightarrow \Omega_\text{offset} \longrightarrow \Omega_\text{tilt_ref}
  \longrightarrow E_\text{mid-arc} \longrightarrow E_s
```
All transformations use Eqs. [](#wws).

The transformations are:
1. {math}`\Lambda_s \longrightarrow \Lambda_\text{mid-arc}`: Transformation from the starting 
branch coordinate system at a longitudinal position along the bend arc of {math}`s` from the upstream 
end of the bend to the branch coordinates at the center point of the arc.
This is a rotation around the center of curvature of the bend and is given by 
Eq. [](#ustt) and Eq. [](#lrztt) with the substitution {math}`\alpha_b \rightarrow (L/2 - s)/\rho`.

2. {math}`\Lambda_\text{mid-arc} \longrightarrow \Omega_\text{mid-chord}`:
This translates (no rotation) the origin of {math}`\Lambda_\text{mid-arc}` 
to the center of the bend chord.
For this transformation, {math}`\bf S` is the unit matrix and
{math}`{\bf L} = \rho(\cos(\alpha_b/2) - 1) \, (\cos\theta_{tr}, \sin\theta_{tr}, 0)` 

3. {math}`\Omega_\text{mid-chord} \longrightarrow \Omega_\text{offset}`: Element misalignment.
This transformation uses Eqs. [](#swww2).

4. {math}`\Omega_\text{offset} \longrightarrow \Omega_\text{tilt_ref}`: Rotation by `tilt_ref`
to get the coordinate system aligned with body coordinates. This transformation uses
{math}`{\bf L} = 0` and {math}`{\bf S} = {\bf R}_z(\theta_{tr})`.

5. {math}`\Omega_\text{tilt_ref} \longrightarrow E_\text{mid-arc}`: Translation to the mid point
on the arc. For this transformation, {math}`\bf S` is the unit matrix and
{math}`{\bf L} = \rho(1 - \cos(\alpha_b/2)) \, (1, 0, 0)` 

6. {math}` E_\text{mid-arc} \longrightarrow E_s`: Transformation along the bend arc to {math}`E_s`.
This is a rotation around the center of curvature of the bend and is given by 
Eq. [](#ustt) and Eq. [](#lrztt) with the substitution {math}`\alpha_b = (s - L/2)/\rho`
and {math}`\theta_{tr} = 0`.

Notice that with this definition of how elements are misaligned, the position of the center of a
misaligned element depends only on the offsets and is independent of the rotations.





