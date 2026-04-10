(sagan.cavity)=
# SaganCavity

## Introduction

The `SagaņCavity` RF cavity tracking method, implemented in `BeamTracking`, 
has both longitudinal energy
gain and transverse focusing effects.


## Justification

A fully self-consistent approach would integrate particle motion through 
the cavity's electromagnetic field map, but such maps are generally only
approximately Maxwellian, are not symplectic, and are computationally
expensive.  At the other extreme, zero-length thin-lens cavity models
provide rapid computation but ignore transverse focusing entirely,
leading to unphysical emittance growth predictions in low-energy
linacs.

The SaganCavity model described here occupies a practical middle ground.  
It is based on a kick-drift-kick symplectic integrator applied to the paraxial
Hamiltonian, using analytic RF field expressions consistent with
Maxwell's equations. Rosenzweig--Serafini (R&S) edge focusing is
included via a symplectic fringe Hamiltonian, and an additional
pondermotive focusing term is applied for standing-wave cavities.
Static (DC) solenoid and multipole components can be superimposed on the cavity.

(sc:params)=
## Input parameters.

RF parameters that can be set are:
```{code} julia
voltage::T                    # Voltage in Volts
rf_frequency::T               # RF frequency in Hz
harmon::T                     # Harmonic number
phi0::T                       # RF phase relative to the zero phase.
zero_phase::PhaseReference.T  # RF phase at phi0 = 0.
traveling_wave::Bool          # Traveling wave or standing wave?
is_crabcavity::Bool           # Is this a crab cavity?
```
See the documentation on the [RF parameter group](#rf.group) for more details.

The `SaganCavity` tracking method itself has parameters:
```{code} yaml
num_cells::Int                # Negative => Use approx half wavelength between cells, Zero => single kick.
L_active::Float64             # Negative => Use L as the active length.
radiation_damping_on::Bool
radiation_fluctuations_on::Bool
```


Example:
```{code} yaml
sc1 = RFCavity(L = 2.0, voltage = 1e6, rf_frequency =1e9, dE_ref = 0.1e9, traveling_wave = false, phi0 = 0.1,
           tracking_method = SaganCavity(num_cells = 2, L_active = 1.6, zero_phase = PhaseReference.BelowTransition),
           Ksol = 0.01)
```

(sc:kdk)=
## Kick-Drift-Kick Model

The cavity of total length $L$ is divided into $N_s$ equal-width
sections.  Inside each section particles are tracked as free particles
(or through a solenoid field if a solenoid component is present).
Longitudinal energy kicks are applied at the section boundaries, with
half-kicks at the entrance and exit ends of the element and full kicks
at interior boundaries.  The voltage (integrated longitudinal field)
at a kick point is

```{math}
:label: eq:vk
V_k = \frac{r_q \, \kappa \, V_\text{tot}}{N_s}
      \cos\!\bigl(2\pi(\phi_t + \phi_\text{ref})\bigr),
```

where $r_q$ is the charge of the tracked particle relative to the
reference particle charge, $V_\text{tot} = G_t L$ is the total cavity
voltage with $G_t$ the accelerating gradient, and $\kappa = 0.5$ at the
two end kick points and $\kappa = 1$ at interior kick points.

The reference RF phase $\phi_\text{ref}(n)$ at the $n^{\text{th}}$ kick point
accounts for the finite transit time of the reference particle across
each section:

```{math}
\phi_\text{ref}(n) = \phi_\text{ref}(0) + f_\text{rf} \sum_{j=1}^{n} t_{0j},
```

where $t_{0j} = L/(c\beta_{0j} N_s)$ is the transit time of the
reference particle across the $j^{\text{th}}$ section, with $\beta_{0j}$ the
reference velocity in that section.  This construction ensures that
the reference particle always sees a constant RF phase at each kick
point, which is important for correctly modeling sub-relativistic
beams where the phase velocity of the reference particle differs
significantly from $c$.

The reference energy $E_0$ is updated at each kick point to match the
energy gained by the reference particle, so that the Bmad phase-space
coordinates remain properly normalized throughout the element.

The number of integration steps $N_s$ is set by the `n_rf_steps`
parameter.  For relativistic beams a small value (e.g. $N_s = 1$)
is often sufficient; for low-energy linacs a larger value is needed
to resolve the phase slippage.  The model is rigorously symplectic at
each step since the drift and kick maps are both individually
symplectic.

(sec:edge)=
## Edge Focusing

At the entrance and exit of the cavity, the radial electric fringe
field produces a first-order transverse kick.  Following R&S {cite}`rosenzweig`
(their Eq. (12)), this is made symplectic by deriving it from the
fringe Hamiltonian

```{math}
:label: eq:Hf
H_f = \mp \frac{q}{2 P_0 c} \, G_t
      \cos(\phi_t + \phi_\text{ref})\,(x^2 + y^2),
```

where the minus sign applies at the entrance and the plus sign at the
exit.  Applying Hamilton's equations in energy-time phase space
yields the symplectic fringe kicks

```{math}
:label: eq:edgekick
\begin{aligned}
  \Delta p_x &= \mp \frac{q}{P_0 c} \, G_t
                \cos(\phi_t + \phi_\text{ref})\, x, \\
  \Delta p_y &= \mp \frac{q}{P_0 c} \, G_t
                \cos(\phi_t + \phi_\text{ref})\, y, \\
  \Delta p_E &= \mp \frac{\pi f_\text{rf} q}{P_0 c^2} \, G_t
                \sin(\phi_t + \phi_\text{ref})\,(x^2 + y^2).
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
H_p = -\frac{\kappa \, L \, q^2 G_t^2}
            {16 N_s P_0^2 c^2 (1 + p_z)}
      (x^2 + y^2),
```

applied at each kick point.  The resulting phase-space kicks are

```{math}
:label: eq:pond
\begin{aligned}
  \Delta p_x &= -\frac{\kappa \, L \, q^2 G_t^2}
                      {8 N_s P_0^2 c^2 (1 + p_z)} \, x, \\
  \Delta p_y &= -\frac{\kappa \, L \, q^2 G_t^2}
                      {8 N_s P_0^2 c^2 (1 + p_z)} \, y, \\
  \Delta z   &= -\frac{\kappa \, L \, q^2 G_t^2}
                      {16 N_s P_0^2 c^2 (1 + p_z)^2}
                (x^2 + y^2).
\end{aligned}
```

Several features are noteworthy.  The kick is always focusing and
independent of RF phase (since the backward-wave phase is being
averaged over).  It is quadratic in $G_t$, analogous to the net
focusing produced by alternating-gradient structures.  The
longitudinal shift $\Delta z$ arises from the transverse oscillation
that the backward wave induces: the sinusoidal transverse motion
slightly delays the particle, reducing $z$.  Each kick is applied at
the $N_s + 1$ kick points with the half-kick factor $\kappa$.

## Solenoid and Multipole Components

A solenoid field component can be superimposed on the cavity by
specifying a solenoid strength `ks`.  In this case the free
drift within each section is replaced by symplectic tracking through a
uniform solenoid, using the exact analytic solenoid map.

Thin-lens multipole kicks (normal and skew, up to arbitrary order)
can also be applied at the kick points alongside the RF kicks.  The
combination is still symplectic because each individual map is
symplectic.

## Bmad Implementation

The model is implemented in the Bmad accelerator simulation
library {cite}`bmad` as the `bmad_standard` tracking method
for `lcavity` elements.  The number of integration steps
$N_s$ is controlled by the `n_rf_steps` element attribute.
The cavity type (traveling wave or standing wave) is selected via the
`cavity_type` attribute; if set to `standing_wave`,
the pondermotive kicks of Section {ref}`sec:pond` are included in
addition to the edge kicks of Section {ref}`sec:edge`.

Transfer maps through the cavity can be computed to arbitrary order
using Bmad's Lie-algebraic map engine, since the underlying
kick-drift-kick structure is fully symplectic.  These maps can be used
for normal-mode analysis, chromaticity calculations, and other
lattice-analysis tasks in low-energy linac lattices.

## Conclusion

A symplectic kick-drift-kick model for linac RF cavities has been
described.  The model correctly captures:

- longitudinal acceleration with finite transit-time phase
  slippage for sub-relativistic beams;
- Rosenzweig--Serafini transverse edge focusing at the cavity
  ends, implemented via a symplectic fringe Hamiltonian;
- pondermotive transverse focusing for standing-wave cavities,
  applied as a time-averaged kick derived from a symplectic
  Hamiltonian;
- optional solenoid and multipole components within the cavity.

The model is computationally efficient: for relativistic beams, a
single-step integration ($N_s = 1$) is often sufficient.  It is
implemented in the open-source Bmad library and is available to any
Bmad-based program.

## References

```{bibliography}
```
