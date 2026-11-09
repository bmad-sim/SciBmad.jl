(sagancavity.tracking)=
# SaganCavity Tracking

## Introduction

The `SaganCavity` tracking method is used to track through RF cavities. 
This method, implemented in `BeamTracking`, has both longitudinal energy
gain and transverse focusing effects. See the [SaganCavity Physics](#sagancavity.physics)
section for details on how particles are tracked. This section discusses the parameters
that affect the tracking.

## SaganCavity Parameters.

The `SaganCavity` tracking method itself has parameters:
```{code} yaml
n_cells::Int                  # Negative => Use approx half wavelength for the cell width, Zero => single kick.
L_active::Float64             # Negative => Use L as the active length.
```
Radiation is turned on with the element settings `rad_damping` and `rad_fluctuations` (see
[Integration Steps and Slices](#track.steps)).

The cavity of total length `L` has an `active` length specified by `L_active`
which is the length over which there is a finite RF field. If not set or is negative, 
`L_active` defaults to `L`.
The active region is always centered on the element and `L_active` is not permitted to be greater
than `L`.

The `n_cells` parameter is the number of RF cells. The `SaganCavity` drift-kick model puts
kicks at the ends of the cells with a "drift" in between.
If not set, the number of cells is chosen such that the cell width
is approximately one-half wavelength and is commensurate with `L_active`.
If `n_cells` is set to zero, a single kick is applied at the center of the element.

## RF Parameters

Other RF parameters that can be set are:
```{code} julia
voltage::T                    # Voltage in Volts
rf_frequency::T               # RF frequency in Hz
harmon::T                     # Harmonic number
phi0::T                       # RF phase relative to the zero phase.
zero_phase::PhaseRef.T        # RF phase at phi0 = 0.
traveling_wave::Bool          # Traveling wave or standing wave?
is_crabcavity::Bool           # Is this a crab cavity?
```
See the documentation on the [RF parameter group](#rf.params) for more details.

## Other Parameters

Besides the length `L`, other parameters that are used for tracking are [solenoid
and multipole parameters](#multipole.sol.params). Solenoid and multipoles are DC fields that span
on the entire length of the element (not just over `L_active`).

Example:
```{code} yaml
sc1 = RFCavity(L = 2.0, voltage = 1e6, rf_frequency =1e9, dE_ref = 0.1e9, traveling_wave = false, phi0 = 0.1,
           tracking_method = SaganCavity(n_cells = 2, L_active = 1.6), zero_phase = PhaseRef.BelowTransition,
           Ksol = 0.01)
```
