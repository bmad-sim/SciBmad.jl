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

(track1)=
# Track

```{code-cell} julia
:tags: [remove-cell]
using SciBmad
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

## Basics

The `track` function can be used for all particle tracking, including spin tracking with
radiation, intra-beam scattering, etc. At the highest level, all one needs to do is provide
the beamline to track through and a matrix of size `n_particles x 6` of the initial particle
canonical phase space coordinates. Note that SciBmad uses the same six canonical coordinates
as in Bmad.

The parameters of the beamline, including [deferred expressions](defexpr.md), are evaluated
once when `track` is called, and kept for all turns of that call. Changes to them take effect
at the next call of `track`. To have parameters change during tracking (for example a ramp),
use [time-dependent parameters](timedependent.md).

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  bpm1 = BPM()
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  bpm2 = BPM()
  sd = Sextupole(Kn2=-1.2, L=0.2)
end

fodo = Beamline([qf, bpm1, sf, b, d, qd, bpm2, sd, b, d],
        species_ref=Species("electron"), pc_ref=18e9)

v0 = [1e-3 0 0 0 0 0;
      1e-3 0 0 0 0 1e-3]
res = track(fodo, v0=v0, n_turns=10)
```

The outputted structure is a `TrackingResult`, which contains quite a lot of information
about the tracking. First, `config` is a `TrackingConfig` structure that contains all the
configuration settings we may have provided to `track`, for example `n_turns`. For a
description of all settings, see the [`TrackingConfig`](#configuration-settings) docstring.

The integer value `start` specifies the beamline index of the element where tracking was commenced (at the entrance end). This can be changed by specifying the `start` keyword argument to `track`. The turns that the particle coordinates were saved at are also included, with 0 being the initial coordinates.

The four vectors `s`, `index`, `name`, and `kind` specify each of the locations in the beamline where phase space coordinates were saved (the s-positions, and the containing elements' beamline indexes, names, and kinds). By default, the phase space coordinates are only saved at the end of the beamline, as in the above example. Further down on this page we discuss saving coordinates at different locations in the beamline.

Next, we have the `TrackColumn`s: `state`, `x`, `px`, `y`, `py`, `z`, `pz`, and `v`. All `TrackColumn`s store the particle data at every selected location and turn. If only one location is saved, then all `TrackColumn`s can be indexed using the syntax `x[turn=n, particle=i]` where `turn` and `particle` can be integers, ranges/vectors of indexes, or colons `:` which specifies all. Both `turn` and `particle` default to a colon `:`.

```{code-cell} julia
res.x[turn=3]               # x of all particles after turn 3
res.x[turn=3, particle=1]   # x of particle 1
res.v[turn=3, particle=1]   # all six coordinates x, px, y, py, z, pz
res.x[]                     # [particle, saved turn]
```

The values of `state` are 8-bit integers that correspond to the following constants defined in `BeamTracking`:

```julia
const STATE_PREBORN                 = 0x0
const STATE_ALIVE                   = 0x1
const STATE_LOST                    = 0x2
const STATE_LOST_NEG_X              = 0x3
const STATE_LOST_POS_X              = 0x4
const STATE_LOST_NEG_Y              = 0x5
const STATE_LOST_POS_Y              = 0x6
const STATE_LOST_PZ                 = 0x7
const STATE_LOST_Z                  = 0x8
const STATE_IMPLICIT_NONCONVERGENCE = 0x9
```

Therefore, if a particle's state is `0x1`, that means it is alive and well! Once a particle
is "lost" in tracking, its phase space coordinates are frozen at the point where it is lost.

```{docstring} track
```

```{docstring} TrackResult
```

```{docstring} TrackColumn
```

## Saving Particle Data at Certain Locations

The particle data can also be saved at other locations given to `track` through the `at` keyword argument, which accepts a vector containing any combination of: 

- s-ranges as a tuple from start to stop, e.g. `(0., 1.23)` saves the coordinates for all integration steps between `s=0` and `s=1.23`
- Beamline element indexes as integers or index ranges (only at the beginning of the element)
- `LatElement`s themselves (only at the beginning of the element)
- A colon `:` for every integration step

For example, to save the phase space coordinates at the entrance of the BPMs in the above line, use:

```{code-cell} julia
res = track(fodo, v0=v0, n_turns=10, at=[bpm1, bpm2])
```

When there is more than one location saved, indexing for all `TrackColumn`s require specifying which location(s) the data should be indexed at as the first indexing argument, using any of the location indexing formats passable to `at` (element, element index, array of elements/indexes, s-ranges, colon).

```{code-cell} julia
res.x[bpm2, turn=10]          # x at the beginning of qd after turn 10
```

If the end of a beamline is saved, then `end` can be specified as the location for that point. A location that is several places (e.g. `[[d]]` for every placement of element `d`, `[qd, :]` for all integration steps through `qd`, an s-range) gives one value per place, in the index after the particle. All of this is the exact same indexing scheme as for the columns of `twiss` (see [Twiss](twiss.md)).

The s, and the beamline index, name and kind of the element of every saved location are in `res.s`, `res.index`, `res.name` and `res.kind`, so that for example the orbit through the beamline can be plotted:

```julia
res = track(fodo, v0=v0, n_turns=1, at=:) # : = Every integration step

using CairoMakie
lines(res.s, res.x[:, turn=1, particle=1]) # x vs. s for every integration step 
```

## Starting at a Different Location
The tracking can also start at the beginning of another element, given with `start`. The first
turn then goes from there to the end of the beamline (for example, `n_turns=1` tracks through
the rest of a linac), and the locations before the start are not reached in the first turn, so
they are `NaN` there (with `state` equal to `BeamTracking.STATE_PREBORN = 0x0`):

(THIS NEEDS FIXING BEFORE NEW SCIBMAD RELEASE!!)

```{code-cell} julia
res = track(fodo, v0=v0, n_turns=2, at=[qf, qd], start=qd)
res.x[qf, particle=1]       # [x at turn 0, NaN (not reached in turn 1), x after turn 2]
```
(END FIX)

Eventually when backwards tracking is supported by SciBmad, there will be an option to track backwards to the beginning of the beamline from any given `start`.

## Saving the Reference Time/Energy

The reference time and energy at the chosen locations can be saved too, with `save_reference=true`:
they are then `res.t_ref` and `res.beta_gamma_ref` (one value per location and turn). This extra save makes the tracking slightly slower, so it is off by default:

```{code-cell} julia
res = track(fodo, v0=v0, n_turns=2, at=[qf, qd], save_reference=true)
res.t_ref[qd]               # at the beginning of qd, for every saved turn
```

## Spin Tracking

To do spin tracking, simply set `spin=true`. This will initialize all particles with their
own spin [quaternion](https://en.wikipedia.org/wiki/Quaternion), each of which will be
tracked along with the particle. A quaternion is essentially a rotation matrix - by tracking
quaternions instead of spin 3-vectors, we can track one particle once and then observe the
spin dynamics for any initial spin direction for that particle.

```{code-cell} julia
v0 = [1e-3 0 0 0 0 1e-3]
res = track(fodo, v0=v0, n_turns=10, spin=true)
```

The spin quaternions are then `res.q`, for example `res.q[turn=5, particle=1]` (its four
components) from the beginning up through turn 5.

To compute a particle's spin 3-vector at each turn given some initial spin direction, we can
use the post-processing function `track_spin`:

```{code-cell} julia
s0 = [0, 1, 0] # initial vertical spin vector
s = track_spin(res.q, s0)
```

( THIS ALSO NEEDS TO BE FIXED)

`s` is now a tensor indexed as `res.data.v`, but with the spin 3-vector index (between 1 and 3)
as the second dimension: `s[1, :, 1, end]` is the spin of particle 1 at the end of the last
turn. If you'd like to initialize different initial spin directions for each particle (perhaps
along the invariant spin field), give the initial spins as a matrix of size `n_particles x 3`.
(END FIX)

## Configuration Settings

While configuration settings can be set as keyword arguments at the level of `track`, a
`TrackConfig` structure may be provided instead, to the keyword argument `config`. All of
these settings are described below in the docstring for `TrackConfig`.

```{docstring} TrackConfig
```

## The Bunch Struct

While initial particle phase space coordinates, quaternions, macroparticle weights, and more
can be set at the level of `track`, these are ultimately used to construct a `Bunch`, which
may alternatively be provided to `track` with the keyword argument `bunch`. A `Bunch`
contains all information about a particle bunch, including its species, current reference
time `t_ref` (used to evaluate [time-dependent parameters](timedependent.md)), and current
reference energy `p_over_q_ref`. All `Bunch` properties are described in the below
docstring.

If we'd like to continue tracking more turns with a bunch used in previous tracking, we can simply provide this bunch to `track`:

```{code-cell} julia
res = track(fodo, v0=v0, n_turns=5)

# track 5 more turns:
res2 = track(fodo, bunch=res.bunch, n_turns=5)
```

```{docstring} Bunch
```

## Tracking a Different Species

(THIS ALSO NEEDS FIXING - TRACK SHOULD ACCEPT A DIFFERENT SPECIES AT THE LEVEL OF TRACK (EASY FIX))

By default, the particles tracked are of the reference species of the beamline. To track a
different species, give `track` a `Bunch` of that species. Just like in a real accelerator,
the magnets keep the same fields, and each particle responds to them according to its own
charge, mass, and energy. For example, muons with the same momentum as the design electrons
follow the same orbit, but they are slower, so they fall behind (their `z` changes more):

```{code-cell} julia
b = Bunch(v=copy(v0), species=Species("muon"))
res = track(fodo, bunch=b, n_turns=10)
res.v[turn=10, particle=1]
```
(END fix)

A particle with the opposite charge, such as a positron in an electron ring, is bent the other
way by the dipoles and is quickly lost, just as it would be in the real ring.


(track.callbacks)=
## Callbacks

The `Bunch` struct may also contain a tuple of functions `callbacks`. These functions are
executed **every integration step**, and allow users to inject their own code into the
tracking if desired. An element with `n_slices` slices of `n_steps` steps each (see
[Integration Steps and Slices](#track.steps)) executes them `n_slices*n_steps` times, and
elements without integration steps (patches, maps, and `SaganCavity`) once. For example, `twiss` uses callbacks to compute the Twiss parameters at every integration step, inside of elements.

Callbacks are required to have the function signature defined below

```julia
function mycallback(i, coords::BeamTracking.Coords, p::TrackPoint)
  # Do stuff...
  return
end
```

In the tracking, each callback is executed and provided the arguments shown above in the
callback signature. These are:

- `i`: Particle index
- `coords`: A `BeamTracking.Coords` struct, containing the fields:
  - `coords.state`: particle state vector of length `n_particles`
  - `coords.v`: particle phase space matrix of size `n_particle x 6`
  - `coords.q`: `nothing` if no spin tracking, else, particle spin quaternion matrix of size `n_particle x 4`
  - `coords.weight`: `nothing` if uniform macroparticle weights, else weights vector of length `n_particles`
- `p`: A `TrackPoint`, the point of the tracking, with the properties:
  - `p.s`: Current s-position from the beginning of the beamline
  - `p.t_ref`: Current reference time of the bunch **from the beginning of tracking**
  - `p.step`: Current integration step from the beginning of the beamline (the steps of the
    elements counted as above, the last one of an element at its exit). This is how `track` finds
    the locations of `at`
  - `p.beta_gamma_ref`: Current reference Lorentz $\beta\gamma$ of the bunch
  - `p.last_ds_step`: Last integration step length before executing this callback
  - `p.last_g`: Last integration coordinate system curvature as a tuple `(gx, gy)` in lab coordinates before executing this callback

The functions `transforms_out!` and `transforms_in!` can be applied inside the callback to `coords` in order to transform the particle coordinates out of the element integration body frame and back into the laboratory frame. This can be useful if, for example, an element is misaligned, but you want to see the phase space coordinates in the laboratory frame. The transformations may be called by
simply executing `transforms_out!(i, coords, p)` or `transforms_in!(i, coords, p)`

Any of these quantities may be used in the callback. Also, as long as callbacks are written
to be GPU compatible, they will work on the GPU. For example, this callback prints the $x$
coordinate of each particle on the GPU in laboratory coordinates, each integration step:

```julia
using CUDA

function mycallback(i, coords, p)
  transforms_out!(i, coords, p) # Transform out of body integration coordinates
  CUDA.@cuprintln("Hi! I'm particle $i at $(p.t_ref) and my x-coordinate is $(coords.v[i,1])")
  transforms_in!(i, coords, p) # Transform back into body integration coordinates
  return
end
```

```{docstring} TrackPoint
```

## Coordinates Number Type
By default, particle phase space (and quaternion) coordinates are 64-bit floats. However, there may be cases where the often-significant performance gains of using 32-bit floats outweigh the cost of lower precision. For example, almost every neural network implementation in major machine learning libraries defaults to 32-bit precision, because it is "good enough". Furthermore, some GPUs (such as Apple Metal) _only_ support 32-bit floats. 

SciBmad tracking is also fully compatible with end-to-end 32-bit float tracking. To use this, simply initialize the particle phase space coordinates (and optionally spin quaternions) using 32-bit floats,

```{code-cell} julia
v0 = zeros(Float32, 1, 6) # 1 32-bit particle
res = track(fodo, v0=v0)
```

:::{note}
When 32-bit floats are used for the particle coordinates, _all_ numbers in the tracking (e.g. magnet strengths, lengths, and time-dependent parameters) are 32-bit floats too.
:::

(cpuparallel)=
## CPU Parallelization

On the CPU, all tracking will automatically be parallelized using explicitly-programmed
single instruction, multiple data (SIMD) instructions if your hardware supports it. To
disable explicit SIMD (and rely on potential compiler auto-vectorization), run `track` with
the keyword argument `use_explicit_SIMD=false`.

### Multithreading

SciBmad tracking is also compatible with CPU multithreading. To do so, Julia must be started
with threads. In a terminal:

```
$ julia --threads=auto
```

where `auto` will infer a number of threads to use based on your system, or alternatively an
integer number of threads may be specified. To check in a Julia session that multiple
threads are available, run the command

```julia
println(Threads.nthreads())
```

If this number is > 1, then you are ready to do multithreaded tracking. Simply set the
`track` configuration setting `use_cpu_multithreading=true`.

:::{note}
There is a cost to launching threads on the CPU. Using multithreading with a small number of
particles may actually slow down your tracking. We recommend checking with a small number of
turns first to see if multithreading can improve your tracking performance, before doing a
full long-term run.
:::

### Multiple Processes/Nodes

(Documentation in development, see
[Distributed](https://docs.julialang.org/en/v1/manual/distributed-computing/) and
[SlurmClusterManager](https://github.com/JuliaParallel/SlurmClusterManager.jl) in the
meantime)

(gpuparallel)=
## GPU Parallelization

**All SciBmad tracking is fully GPU parallelized and branchless.** To do particle tracking
on the GPU, either initialize a `Bunch` with the corresponding device `GPUArray`, or provide
the `GPUArray` to `track`. For example, to do GPU-parallelized spin tracking on a CUDA GPU,

```julia
using CUDA

n_particles = 100000

# Random initial phase space coordinates:
v0 = CUDA.rand(Float64, n_particles, 6) .* 1e-5

res = track(fodo, v0=v0, spin=true, n_turns=100)
```

:::{note}
Julia is a just-in-time (JIT) compiled language, meaning that many functions (including the
tracking kernels in SciBmad) are only compiled once they are called with specific types
(e.g., a `GPUArray`). We try to precompile for as many cases as possible, but precompilation
for `GPUArray`s is not possible. As such, the first "turn" in tracking will have a latency
in order for the JIT compiler to compile SciBmad's GPU tracking kernels.
:::

Apple Metal GPUs only support 32-bit floats, so on a Mac use `Float32` coordinates; all of
the tracking, including spin tracking, will then be done in 32-bit:

```julia
using Metal
v0 = MtlArray(rand(Float32, n_particles, 6) .* 1f-5)
res = track(fodo, v0=v0, spin=true, n_turns=100)
```

### Multiple GPUs

(Documentation in development, see
[CUDA: Multiple GPUs](https://cuda.juliagpu.org/stable/usage/multigpu/#Multiple-GPUs) in the
meantime)
