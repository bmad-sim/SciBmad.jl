## RF Parameter Group

The `RFParams` parameter group holds parameters for RF cavity elements.
User settable RF parameters are:
```{code} julia
voltage::T                    # Voltage in Volts
rf_frequency::T               # RF frequency in Hz
harmon::T                     # Harmonic number
phi0::T                       # RF phase relative to the zero phase.
zero_phase::PhaseReference.T  # RF phase at phi0 = 0.
traveling_wave::Bool          # Traveling wave or standing wave?
is_crabcavity::Bool           # Is this a crab cavity?
```
Where `T` is some type like a float or Taylor series. The `zero_phase` enum sets what the
RF phase is at zero `phi0`. Possible settings for `zero_phase`
are:
- `PhaseReference.Accelerating`      # Zero phase is the maximum accelerating phase.
- `PhaseReference.BelowTransition`   # Zero phase is at the stable zero crossing for particles below transition.
- `PhaseReference.AboveTransition`   # Zero phase is at the stable zero crossing for particles above transition.

The `RFParams` parameter group structure has the following components:
```{code} julia
mutable struct RFParams{T} <: AbstractParams
  rate::T                           # RF frequency in Hz or Harmonic number
  voltage::T                        # Voltage in Volts
  phi0::T                           # Phase at reference energy
  const rate_meaning::RateMeaning.T # false = frequency in Hz, true = harmonic number, -1 = Not set
  zero_phase::PhaseReference.T      # Determines the RF phase at phi0 = 0
  traveling_wave::Bool              # Traveling wave or standing wave cavity?
  is_crabcavity::Bool               # Is this a crab cavity?
end
```
The `rate_meaning` enum records whether `rf_frequency` or `harmon` has been set by the User.
Possible settings for `rate_meaning` are:
- `RFFrequency`       # `rf_frequency` has been set.
- `Harmon`            # `harmon` has been set.
- `Indeterminate`     # Neither `rf_frequency` nor `harmon` has been set.


