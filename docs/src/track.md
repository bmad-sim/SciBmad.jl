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

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-1.2, L=0.2)
end

fodo = Beamline([qf, sf, b, d, qd, sd, b, d],
        species_ref=Species("electron"), pc_ref=18e9)

v0 = [1e-3 0 0 0 0 1e-3]
res = track(fodo, v0=v0, n_turns=10)
```

The outputted structure is a `TrackingResult`, which contains quite a lot of information
about the tracking. First, `config` is a `TrackingConfig` structure that contains all the
configuration settings we may have provided to `track`, for example `n_turns`. For a
description of all settings, see the [`TrackingConfig`](#configuration-settings) docstring.

Next, we have a matrix `state`, which corresponds to each particle's state at each saved
turn. For example, particle 1's state at turn 3 is indexed with `res.state[1,4]`
(**note:** 1 must be added to the turn number, because `res` also stores the initial
particle states/coordinates). The values of `state` are 8-bit integers that correspond to
the following constants defined in `BeamTracking`:

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

Then we have a tensor `v` which contains each particle's phase space coordinates at each
saved turn. The indexing of this tensor is described in the output. For example, particle
1's phase space coordinates at turn 5 are indexed with `res.v[1,:,6]` (**again note:** 1
must be added to the turn number, because `res` also stores the initial particle
states/coordinates).

Finally we have `bunch`, which contains the full `Bunch` structure at the end of tracking.
If we'd like to continue tracking more turns with this bunch, we can simply provide this
bunch to `track`:

```{code-cell} julia
# track 5 more turns:
res2 = track(fodo, bunch=res.bunch, n_turns=5)
```

To do spin tracking, simply set `spin=true`. This will initialize all particles with their
own spin [quaternion](https://en.wikipedia.org/wiki/Quaternion), each of which will be
tracked along with the particle. A quaternion is essentially a rotation matrix - by tracking
quaternions instead of spin 3-vectors, we can track one particle once and then observe the
spin dynamics for any initial spin direction for that particle.

```{code-cell} julia
v0 = [1e-3 0 0 0 0 1e-3]
res = track(fodo, v0=v0, n_turns=10, spin=true)
```

Our `TrackingResult` now has an extra tensor `q`, which contains each particle's spin
quaternion from the beginning up to each saved turn. Similar to the phase space coordinates
indexing, particle 1's spin quaternion from the beginning up through turn 5 can be indexed
as `res.q[1,:,6]` (**again note:** 1 must be added to the turn number, because `res` also
stores the initial particle quaternion).

To compute a particle's spin 3-vector at each turn given some initial spin direction, we can
use the post-processing function `track_spin`:

```{code-cell} julia
s0 = [0, 1, 0] # initial vertical spin vector
s = track_spin(res.q, s0)
```

`s` is now a tensor that can be indexed in the same way as `v` and `q`, but now the second
dimension is the spin 3-vector index (between 1 and 3). If you'd like to initialize
different initial spin directions for each particle (perhaps along the invariant spin
field), give the initial spins as a matrix of size `n_particles x 3`.

```{docstring} track
```

```{docstring} TrackingResult
```

## Configuration Settings

While configuration settings can be set as keyword arguments at the level of `track`, a
`TrackingConfig` structure may be provided instead, to the keyword argument `config`. All of
these settings are described below in the docstring for `TrackingConfig`.

```{docstring} TrackingConfig
```

## The Bunch Struct

While initial particle phase space coordinates, quaternions, macroparticle weights, and more
can be set at the level of `track`, these are ultimately used to construct a `Bunch`, which
may alternatively be provided to `track` with the keyword argument `bunch`. A `Bunch`
contains all information about a particle bunch, including its species, current reference
time `t_ref` (used to evaluate [time-dependent parameters](timedependent.md)), and current
reference energy `p_over_q_ref`. All `Bunch` properties are described in the below
docstring.

```{docstring} Bunch
```

## Callbacks

The `Bunch` struct may also contain a tuple of functions `callbacks`. These functions are
executed **every integration step**, and allow users to inject their own code into the
tracking if desired. For example, `twiss` uses callbacks to compute the Twiss parameters at
every integration step, inside of elements.

Callbacks are required to have the function signature defined below

```julia
function mycallback(i, coords, cur_s, cur_t_ref, cur_beta_gamma_ref, last_ds_step, last_g, transforms_out!, transforms_in!)
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
- `cur_s`: Current s-position from the element start
- `cur_t_ref`: Current reference time of the bunch **from the beginning of tracking**
- `cur_beta_gamma_ref`: Current reference Lorentz $\beta\gamma$ of the bunch
- `last_ds_step`: Last integration step length before executing this callback
- `last_g`: Last integration coordinate system curvature as a tuple `(gx, gy)` in lab coordinates before executing this callback

The last two arguments, `transforms_out!` and `transforms_in!`, are special, in that they
are functions that can be applied to `coords` in order to transform the particle coordinates
out of the element integration body frame and back into the laboratory frame. This can be
useful if, for example, an element is misaligned, but you want to see the phase space
coordinates in the laboratory frame. The transformations may be called by simply executing
`transforms_out!(i, coords, cur_s, cur_t_ref)` or
`transforms_in!(i, coords, cur_s, cur_t_ref)`

Any of these quantities may be used in the callback. Also, as long as callbacks are written
to be GPU compatible, they will work on the GPU. For example, this callback prints the $x$
coordinate of each particle on the GPU in laboratory coordinates, each integration step:

```julia
using CUDA

function mycallback(i, coords, cur_s, cur_t_ref, cur_beta_gamma_ref, last_ds_step, last_g, transforms_out!, transforms_in!)
  transforms_out!(i, coords, cur_s, cur_t_ref) # Transform out of body integration coordinates
  CUDA.@cuprintln("Hi! I'm particle $i at $cur_t_ref and my x-coordinate is $(coords.v[i,1])")
  transforms_in!(i, coords, cur_s, cur_t_ref) # Transform back into body integration coordinates
  return
end
```

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

### Multiple GPUs

(Documentation in development, see
[CUDA: Multiple GPUs](https://cuda.juliagpu.org/stable/usage/multigpu/#Multiple-GPUs) in the
meantime)
