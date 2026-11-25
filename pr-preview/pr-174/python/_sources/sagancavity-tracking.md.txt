(sagancavity.tracking)=
# SaganCavity Tracking

## Introduction

The `SaganCavity` tracking method tracks through RF cavities. It includes both the energy gain
and the transverse focusing of the cavity. How particles are tracked is described in
[SaganCavity Physics](#sagancavity.physics). This page describes the parameters that affect the
tracking.

## SaganCavity Parameters

The `SaganCavity` tracking method itself has two parameters:
```{code} julia
n_cells::Int       # Number of cells. Negative (default): cells of about half a wavelength. Zero: one kick
L_active::Float64  # Length with RF field. Negative (default): the element length L
```
Radiation is turned on with the element settings `rad_damping` and `rad_fluctuations` (see
[Integration Steps and Slices](#track.steps)).

The cavity of total length `L` has an active length `L_active`, the length over which there is
an RF field. If it is not set or negative, it is `L`. The active region is always centered on the
element, and `L_active` cannot be greater than `L`.

The `n_cells` parameter is the number of RF cells. The `SaganCavity` model puts kicks at the ends
of the cells, with a "drift" in between. If `n_cells` is not set, the cells divide `L_active`
into equal parts of about half a wavelength each. If `n_cells` is zero, a single kick is applied
at the center of the element.

## RF Parameters

Other RF parameters that can be set are:
```{code} julia
voltage::T                    # Voltage in Volts
rf_frequency::T               # RF frequency in Hz
harmon::T                     # Harmonic number
phi0::T                       # RF phase relative to the zero phase [rad]
zero_phase::PhaseRef.T        # RF phase at phi0 = 0.
traveling_wave::Bool          # Traveling wave or standing wave?
is_crabcavity::Bool           # Is this a crab cavity?
```
See the documentation on the [RF parameter group](#rf.params) for more details.

## Other Parameters

Besides the length `L`, the [solenoid and multipole parameters](#multipole.sol.params) are used
for tracking. The solenoid and multipole fields are static, and span the entire length of the
element, not just `L_active`.

`dE_ref` is the change of the reference energy through the cavity. It should be the energy gain
of the reference particle. An element with `dE_ref` must be the first element of a `Beamline`.

For example, a 1 MV standing-wave cavity on crest (the default `zero_phase` is
`PhaseRef.Accelerating`, and `phi0` is 0), which gives an electron 1 MeV, with a solenoid field:
```{code} julia
sc1 = RFCavity(L = 2.0, voltage = 1e6, rf_frequency = 1e9, dE_ref = 1e6, traveling_wave = false,
               tracking_method = SaganCavity(n_cells = 2, L_active = 1.6), Ksol = 0.01)
```
