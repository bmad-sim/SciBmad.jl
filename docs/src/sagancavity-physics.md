(sagancavity.physics)=
# SaganCavity Physics

## Introduction

The `SagaņCavity` RF cavity tracking method, implemented in `BeamTracking`, 
has both longitudinal energy gain and transverse focusing effects.
For a listing of parameters used in the tracking see the 
[SaganCavity Tracking Method](#sagancavity.tracking) section.

## Justification

The SaganCavity model described here occupies a practical middle ground
between trying to integrate through a cavity's field which can be slow
and the simple zero-length thin-lens cavity model that provides rapid computation
but generally ignores transverse focusing entirely,
leading to unphysical emittance growth predictions in low-energy
linacs.

The SaganCavity model is based on a kick-drift-kick symplectic integrator applied to a paraxial
Hamiltonian. Rosenzweig--Serafini (R&S) edge focusing is
included via a symplectic fringe Hamiltonian, and an additional
pondermotive focusing term is applied for standing-wave cavities.
Static (DC) solenoid and multipole components can be superimposed on the cavity.


(sc:kdk)=
## Kick-Drift-Kick Model

The cavity of total length `L` has an "active" length specified by `L_active`
which is the length over which there is a finite RF field. The active length
is centered longitudinally in the element. If there are (DC) multipole or solenoid
fields, these fields exist along the entire element.

`L_active` is divided into `num_cells` equal-width
cells. Inside each cell particles are tracked as free particles
(or through a solenoid field if a solenoid component is present).
Longitudinal energy kicks are applied at the cell boundaries, with
half-kicks at the entrance and exit ends of the element and full kicks
at interior boundaries. The energy kick at the $n^{th}$ kick point is

```{math}
:label: eq:vk
dE(n) = \frac{Q \, \kappa(n) \, V }{N_c} \cos( \phi_{RF}(n) ),
```
where $Q$ is the particle charge, $V$ is the `voltage`, $N_c$ is the
number of cells, and $\kappa = 0.5$ at the
two end kick points and $\kappa = 1$ at interior kick points.
In the above equation $\phi_{RF}$ is the rf phase
```{math}
\phi_{RF} = 2 \pi \, f_\text{rf} \, \left( 
t - t_\text{ref} - t_t(n)
\right) + \text{phi0}
```
where $t$ is the particle time and $t_\text{ref}$ is element reference time which depends upon the setting
of the `zero_phase` parameter and whether relative or absolute time is being used. 
The transit time $t_t(n)$ accounts for the finite transit time of the reference particle 
to go from the beginning of the active region to kick point $n$ where $n$ runs from zero to $N_c$
```{math}
t_t(n) = \sum_{j=1}^{n} t_{0j},
```
where $t_{0j} = L/(c\beta_{0j} N_s)$ is the transit time of the
reference particle across the $j^{\text{th}}$ cell, with $\beta_{0j}$ the
reference velocity in that cell. 
The reference velocity is computed assuming that the reference particle gains an energy
of $\kappa(n) dE_\text{ref} /N_c$ at the $n^{th}$ kick point.
This construction ensures that with the voltage set commensurate with $dE_\text{ref}$, 
a particle on the zero orbit will see a constant RF phase at each kick
point. This is important for correctly modeling sub-relativistic
beams where the phase velocity of the reference particle can differ
significantly from the speed of light.

(sec:edge)=
## Edge Focusing

At the entrance and exit of the cavity, the radial electric fringe
field produces a first-order transverse kick.  Following R&S {cite}`rosenzweig`
(their Eq. (12)), this is made symplectic by deriving it from the
fringe Hamiltonian

```{math}
:label: eq:Hf
H_f = \mp \frac{q}{2 P_0 c} \, G \, \cos(\phi_{RF})\,(x^2 + y^2),
```
where $G$ is the gradient $V/L_\text{active}$ and
the minus sign applies at the entrance and the plus sign at the
exit. If `L_active` is zero, the edge focusting is not applied.

Applying Hamilton's equations in energy-time phase space
yields the symplectic fringe kicks
```{math}
:label: eq:edgekick
\begin{aligned}
  \Delta p_x &= \mp \frac{q}{P_0 c} \, G
                \cos(\phi_{RF})\, x, \\
  \Delta p_y &= \mp \frac{q}{P_0 c} \, G
                \cos(\phi_{RF})\, y, \\
  \Delta p_E &= \mp \frac{\pi f_\text{rf} q}{P_0 c^2} \, G
                \sin(\phi_{RF})\,(x^2 + y^2).
\end{aligned}
```

The coupling between the transverse kick and the energy kick
(the $\Delta p_E$ term) is required for symplecticity.
The sign convention is such that the entrance kick is defocusing and
the exit kick is focusing for an accelerating cavity, with the net
first-order effect being a focusing half-integer phase advance
consistent with the R&S result.

(sec:pond)=
## Standing-Wave Pondermotive Focusing

For a *traveling-wave* cavity, the forward-propagating wave
provides both acceleration and the edge kick of Section {ref}`sec:edge`.
A *standing-wave* cavity is modeled as the superposition of a
forward and a backward wave of equal amplitude.  The backward wave
does not contribute to the net acceleration (its kick averages to zero
for an ultra-relativistic particle traversing many cells), but it
produces an additional transverse focusing force called the
pondermotive force.

Because the phase of the backward wave as seen by the forward-traveling
particle varies rapidly, the kick is applied in a time-averaged sense.
Using R&S Eq. (4) with $\eta(\phi) = 1$ (no harmonics), the
time-averaged pondermotive kick is derived from the Hamiltonian
```{math}
:label: eq:Hp
H_p = -\frac{\kappa \, L \, q^2 G^2}
            {16 N_s P_0^2 c^2 (1 + p_z)}
      (x^2 + y^2),
```
applied at each kick point. Like the edge focusing, if `L_active` is zero which would
give an infinite $G$, the pondermotive kick is not applied.

The resulting phase-space kicks are
```{math}
:label: eq:pond
\begin{aligned}
  \Delta p_x &= -\frac{\kappa \, L \, q^2 G^2}
                      {8 N_s P_0^2 c^2 (1 + p_z)} \, x, \\
  \Delta p_y &= -\frac{\kappa \, L \, q^2 G^2}
                      {8 N_s P_0^2 c^2 (1 + p_z)} \, y, \\
  \Delta z   &= -\frac{\kappa \, L \, q^2 G^2}
                      {16 N_s P_0^2 c^2 (1 + p_z)^2}
                (x^2 + y^2).
\end{aligned}
```

Several features are noteworthy.  The kick is always focusing and
independent of RF phase (since the backward-wave phase is being
averaged over).  It is quadratic in $G$, analogous to the net
focusing produced by alternating-gradient structures.  The
longitudinal shift $\Delta z$ arises from the transverse oscillation
that the backward wave induces: the sinusoidal transverse motion
slightly delays the particle, reducing $z$.  Each kick is applied at
the $N_s + 1$ kick points with the half-kick factor $\kappa$.

## Solenoid and Multipole Components

A solenoid field component can be superimposed on the cavity by
specifying a solenoid strength `ks`.  In this case the free
drift within each cell is replaced by symplectic tracking through a
uniform solenoid, using the exact analytic solenoid map.

Thin-lens multipole kicks (normal and skew, up to arbitrary order)
can also be applied at the kick points alongside the RF kicks.  The
combination is still symplectic because each individual map is
symplectic.

## References

```{bibliography}
```
