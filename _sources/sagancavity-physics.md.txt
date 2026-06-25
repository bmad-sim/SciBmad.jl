(sagancavity.physics)=
# SaganCavity Physics

## Overview

The `SagaņCavity` RF cavity tracking method, implemented in `BeamTracking`, 
uses a symplectic kick-drift-kick model and
has both longitudinal energy gain and transverse focusing effects.
DC solenoid and multipole fields can be superimposed on the RF fields.
For a listing of parameters used in the tracking, see the 
[SaganCavity Tracking Method](#sagancavity.tracking) and 
[RF parameter group](#rf:params) documentation .

## Justification

The SaganCavity tracking method occupies a practical middle ground
between trying to integrate through a cavity's field which can be slow
and the simple zero-length thin-lens cavity model that provides rapid computation
but generally ignores transverse focusing entirely and which can 
lead to unphysical emittance growth for simulations in low-energy
linacs.

The model used for SaganCavity tracking is kick-drift-kick. 
Rosenzweig--Serafini (R&S) {cite}`Rosenzweig:RF` edge focusing is
included via a symplectic fringe Hamiltonian along with a
pondermotive focusing term that is applied for standing-wave cavities.

%---------------------------------------------------------------------------------------------------
(sc:kdk)=
## Kick-Drift-Kick Model

The cavity of total length `L` has an "active" length specified by `L_active`
which is the length over which there is a finite RF field. The active length
is centered longitudinally in the element. If there are (DC) multipole or solenoid
fields, these fields exist along the entire element and not just in the active region.

`L_active` is divided into `num_cells` equal-width cells. 
Inside each cell, particles are tracked ("drifted") ignoring the RF field but
including any DC solenoid field. 
Longitudinal energy kicks are applied at the cell boundaries, with half-kicks at 
the entrance and exit ends of the active region and full kicks at interior boundaries. 
Additionally, entrance and exit fringe kicks are applied at the ends of the active region.
Note: Due to the fact that in an actual cavity the fields are not uniform longitudinally,
increasing the number of kick points does not necessarily increase the simulation accuracy.

To simplify the algorithm, 
both the energy kicks at the cell ends and the edge kicks at the ends of the active region are
applied by first converting from the standard $(z, p_z)$ **momentum** particle coordinates
to $(t, p_E)$ **energy** coordinates where $t$ is the time and $p_E = E/P_0 c$ with E being 
the energy, and $P_0$ beging the reference momentum. 
After any kicks have been applied, a conversion back to momentum coordinates is done.

%---------------------------------------------------------------------------------------------------
(sc:edge)=
## Edge Kick

At the entrance and exit ends of the active region, the radial electric fringe
field produces a transverse kick. Following R&S {cite}`Rosenzweig:RF`
(their Eq. (12)), the appropriate fringe Hamiltonian to reproduce this, in energy coordinates, is:
```{math}
:label: eq:Hf
H_f = \mp \frac{q}{2 P_0 c} \, G \, \cos(\phi_{RF})\,(x^2 + y^2),
```
where $q$ is the particle charge, and $G$ is the gradient which is equal to $V/L_\text{active}$
with $V$ being the voltage.
In the above equation, [$\phi_{RF}$](#sc:phase)
is the RF phase and the minus sign applies at the entrance end of the active region
and the plus sign at the exit end. 
If `L_active` is zero (giving infinite gradient), the edge focusting is not applied.

Applying Hamilton's equations in energy-time phase space coordinates
yields the fringe kicks
```{math}
:label: eq:edgekick
\begin{aligned}
  \Delta p_x &= \mp \frac{q}{P_0 c} \, G \cos(\phi_{RF})\, x, \\
  \Delta p_y &= \mp \frac{q}{P_0 c} \, G \cos(\phi_{RF})\, y, \\
  \Delta p_E &= \mp \frac{\pi f_\text{rf} q}{P_0 c^2} \, G
                \sin(\phi_{RF})\,(x^2 + y^2).
\end{aligned}
```
where $f_\text{rf}$ is the RF frequency given by the `rf_frequency` parameter.
The entrance kick will be defocusing and
the exit kick will be focusing for accelerated particles. 

The spin fringe kick is calculated using the Thomas-BMT equation with the integrated radial electric field being
```{math}
\int_\text{kick} {\bf E} = \pm G \, \cos(\phi_{RF}) \, (x, y, 0)
```

%---------------------------------------------------------------------------------------------------
(sc:energy)=
## Energy Kick

Kick points at the cell boundaries are indexed from zero to $N_c$ (set with the `num_cells` parameter).
The energy kick at the $n^{th}$ kick point is
```{math}
:label: eq:vk
dE(n) = \frac{q \, \kappa_n \, V }{N_c} \cos( \phi_{RF}(n) ),
```
where $V$ is the voltage set by the `voltage` parameter, and $\kappa_n = 0.5$ at the
two end kick points and $\kappa_n = 1$ at interior kick points.

The spin kick is calculated using the Thomas-BMT equation with the integrated electric field being
```{math}
\int_\text{kick} {\bf E} = (0, 0, c \cdot dE / q)
```

%---------------------------------------------------------------------------------------------------
(sc:phase)=
## RF Phase

At the $n^{th}$ kick point, the RF phase $\phi_{RF}(n)$ in the above equations is taken to be 
```{math}
\phi_{RF}(n) = 2 \pi f_\text{rf} \left( t - t_0 - t_\text{trans}(n) \right) + \phi_0
```
where $t$ is the particle time, and $\phi_0$ is set by the `phi0` parameter.
In the above equation, $t_0$ is element reference time which depends upon the setting
of the `zero_phase` parameter and whether relative or absolute time is being used. 
The transit time $t_\text{trans}(n)$ accounts for the finite transit time of the reference particle 
to go from the beginning of the active region to kick point $n$.
```{math}
t_\text{trans}(n) = t_{out} + \sum_{j=1}^{n} t_c(j)
```
where
```{math}
t_{out} = \frac{L_{out}}{c \,\beta_\text{ref}(out)}, \qquad
t_c(j) = \frac{L}{c \, \beta_\text{ref}(j) N_s}
```
with $t_{out}$ being the reference transit time in the region just before the active
region from the element entrance end to active region entrance end, and
$t_c(j)$ is the reference transit time across the $j^{\text{th}}$ cell (indexed from 1 to $N_c$).
In the above equations, $L_{out} = (L - L_\text{active}) / 2$ and
$\beta_\text{ref}$ is the reference velocity which is 
computed assuming that the reference energy increases by
$\kappa_n dE_\text{ref} /N_c$ at the $n^{th}$ kick point with $dE_\text{ref}$ being
set by the `dE_ref` paramter.
This construction ensures that with the voltage set commensurate with `dE_ref` and `phi_0`, 
a particle on the zero orbit will see a constant RF phase at each kick
point. This is important for correctly modeling sub-relativistic
beams where the phase velocity of the reference particle can differ
significantly from the speed of light.

%---------------------------------------------------------------------------------------------------
(sc:ponder)=
## Standing-Wave Pondermotive Focusing

For a *traveling-wave* cavity, there is a forward-propagating wave
and this wave provides both acceleration and the edge kick of Section {ref}`sc:edge`.
A *standing-wave* cavity is modeled as the superposition of a
forward-propagating and a backward-propagating wave of equal amplitude.
The backward wave does not contribute to the net acceleration (its kick averages to zero
for a particle traversing many cells), but the backwards wave
produces a transverse focusing force called the pondermotive force.

Because the phase of the backward wave as seen by a forward-traveling
particle varies rapidly, the pondermotive kick is treated using a first-order approximation
which give a time-averaged force.
The appropriate Hamiltonian corresponding to R&S Eq. (4) with $\eta(\phi) = 1$ (no harmonics) is:
```{math}
:label: eq:Hp
H_p = -\frac{\kappa_n \, L \, q^2 G^2}{16 N_s P_0^2 c^2 (1 + p_z)}
      (x^2 + y^2),
```
The pondermotive kick is applied at each kick point. 
Like the edge focusing, if `L_active` is zero which would
give an infinite $G$, the pondermotive kick is not applied.

The resulting phase-space kicks are
```{math}
:label: eq:pond
\begin{aligned}
  \Delta p_x &= -\frac{\kappa_n \, L \, q^2 G^2}
                      {8 N_s P_0^2 c^2 (1 + p_z)} \, x, \\
  \Delta p_y &= -\frac{\kappa_n \, L \, q^2 G^2}
                      {8 N_s P_0^2 c^2 (1 + p_z)} \, y, \\
  \Delta z   &= -\frac{\kappa_n \, L \, q^2 G^2}
                      {16 N_s P_0^2 c^2 (1 + p_z)^2}
                (x^2 + y^2).
\end{aligned}
```

Several features are noteworthy. The kick is always focusing and
independent of RF phase (since the backward-wave phase is being
averaged over).  It is quadratic in $G$, analogous to the net
focusing produced by alternating-gradient structures.  The
longitudinal shift $\Delta z$ arises from the transverse oscillation
that the backward wave induces: the sinusoidal transverse motion
slightly delays the particle, reducing $z$.

%---------------------------------------------------------------------------------------------------
## Solenoid and Multipole Components

A solenoid field component can be superimposed on the cavity.
In this case, the free drifts within each cell are replaced by symplectic tracking through a
uniform solenoid field.

If there are multipoles, multipole kicks are applied at the ends of the element
and at the kick points at the cell ends alongside the RF kicks.

%---------------------------------------------------------------------------------------------------
## References

```{bibliography}
```
