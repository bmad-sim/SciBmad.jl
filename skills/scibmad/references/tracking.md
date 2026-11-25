# Tracking, closed orbit, dynamic aperture, NAFF

Docs pages: `track.md`, `tracking-methods.md`, `collective.md`, `timedependent.md`,
`batch.md`, `co.md`, `dynamic-aperture.md`, `fma.md`, `sagancavity-tracking.md`.

## Contents
1. `track` basics and reading results
2. Saving at locations, starting elsewhere, reference time
3. Spin
4. Configuration, `Bunch`, continuing, other species
5. Performance: Float32, threads, GPU
6. Tracking methods, integration steps, radiation
7. Collective effects (IBS)
8. Time-dependent parameters and ramping
9. `BatchParam`: many lattices at once
10. Callbacks
11. Closed orbit
12. Dynamic aperture
13. Frequency analysis (`naff`)

## 1. `track` basics

```julia
v0 = [1e-3 0 0 0 0 0;        # one row per particle: x, px, y, py, z, pz
      1e-3 0 0 0 0 1e-3]
res = track(fodo, v0=v0, n_turns=10)
res.x[turn=10]               # x of all particles after turn 10 (turn 0 = initial)
res.x[turn=10, particle=1]   # one number
res.x[turn=:, particle=1]    # turns 0..10 of particle 1
res.v[turn=3, particle=1]    # all six coordinates
res.x[]                      # [particle, saved turn]; saved turn k is res.turns[k]
res.state[turn=10]           # 0x1 = alive
```

By default particles are saved at the end of the beamline after every turn. `turn` and
`particle` accept integers, ranges, vectors or `:` (default `:`). Parameters (including
deferred expressions) are evaluated once per `track` call.

`state` codes (`BeamTracking.STATE_*`): `0x0` PREBORN (location not reached yet), `0x1` ALIVE,
`0x2` LOST, `0x3`/`0x4` LOST_NEG_X/POS_X, `0x5`/`0x6` LOST_NEG_Y/POS_Y, `0x7` LOST_PZ,
`0x8` LOST_Z, `0x9` IMPLICIT_NONCONVERGENCE. Lost particles keep the coordinates where they
were lost.

Other `TrackResult` fields: `res.turns`, `res.start`, `res.s`, `res.index`, `res.name`,
`res.kind` (the saved locations), `res.config`, `res.elapsed_time`, the raw arrays
`res.data.v[particle, coordinate, location, saved turn]` etc., and `res.bunch`, the `Bunch` after
the last turn: `res.bunch.coords.v` (n×6), `.q` (n×4 or `nothing`), `.state`. Final values
come from there without saving any location.

## 2. Locations, start, reference

```julia
res = track(fodo, v0=v0, n_turns=10, at=[bpm1, bpm2])   # elements, indexes, names, s-ranges, :
res.x[bpm2, turn=10]                    # with several locations, the location comes first
res.x[[d], turn=10]                     # every placement: [particle, location]
res = track(fodo, v0=v0, n_turns=1, at=:)  # every integration step
res.s, res.x[:, turn=1, particle=1]     # orbit along s (e.g. lines(res.s, ...))
```

`at` takes elements (the entrance of each placement), beamline indexes, names, s-range tuples
and `:`. With `at`, only those locations are saved: add `length(bl.line) + 1` to also keep the
end of the beamline. An exit is not an `at` entry (`at=[[qd, :exit]]` is an error, "Unsupported
location"): save the next element or its beamline index instead; it is the same location.

Reading uses the indexing of `twiss` columns (`twiss.md`): an element (its entrance),
`[qd, :exit]`, `[qd, :]`, `[qd, k]`, a beamline index, `end`, names, s-ranges. Reading a
location that was not saved is an error.

`start=qd` starts the first turn at the entrance of `qd` (e.g. `n_turns=1` tracks the rest of
a linac); locations before `start` are `NaN`/PREBORN in turns 0 and 1.

`save_reference=true` also saves `res.t_ref` (s) and `res.beta_gamma_ref` per location and turn
(slightly slower).

## 3. Spin

```julia
res = track(fodo, v0=v0, n_turns=10, spin=true)  # each particle carries a spin quaternion
res.q[turn=5, particle=1]                         # rotation from the start through turn 5
s = track_spin(res.q, [0, 1, 0])                  # spin vectors for an initial spin (or n x 3)
s[turn=10, particle=1]                            # [sx, sy, sz]
s[turn=:, particle=1][2, :]                       # sy over all turns
```

Quaternions are tracked once and then applied to any initial spin direction. `RK4` does not
support spin. The final spins (end of the beamline, last turn) need no save point:
`track_spin(res.bunch.coords.q, [0, 0, 1])` gives them as an n×3 matrix. Convention check: an
electron spin starting along z through an `SBend` with positive `angle` (bending toward −x)
ends with s_x = −sin(aγθ), s_z = cos(aγθ).

## 4. Configuration, Bunch, species

Keywords of `track` (or a `TrackConfig` via `config=`): `n_turns` (1), `save_every_n_turns`
(1), `at`, `start`, `spin`, `weight`, `species`, `bunch`, `in_body_coordinates` (false),
`save_reference` (false), `rf_on` (true), `ramp_particle_energy_without_rf` (false),
`ramp_update_each_particle` (false), `batch_start` (1), `verbose` (false),
`use_cpu_multithreading` (false), `use_explicit_SIMD`, `groupsize`. Check `@doc track` for the
exact signature.

Continue tracking a bunch (tracked in place; earlier saved results are unchanged):

```julia
res = track(fodo, v0=v0, n_turns=5)
res2 = track(fodo, bunch=res.bunch, n_turns=5)
```

`track(fodo, v0=v0, species=Species("muon"))` tracks another species through the same fields
(same momentum → same orbit, different velocity → different `z`). A particle of opposite charge
bends the other way and is lost in a ring, as in reality.

Start a distribution on the closed orbit with the normalizing map (from `twiss.md`):

```julia
co = find_closed_orbit(ring).v0                  # 1x6
A = twiss(ring, cols=["a"], at=[1]).a[1]         # 6x6 matrix (normalized -> lab)
emit = [1e-9, 1e-12, 1e-8]                       # mode emittances [m]
xn = randn(n, 6) .* sqrt.(repeat(emit, inner=2))'
v0 = co .+ xn*A'
```

## 5. Performance

- CPU SIMD is automatic. Threads: start Julia with `--threads=auto` and pass
  `use_cpu_multithreading=true` (only pays off for many particles; test first).
- 32-bit: `v0 = zeros(Float32, n, 6)`; then *all* numbers in the tracking are Float32.
- GPU: pass a GPU array, e.g. `using CUDA; v0 = CUDA.rand(Float64, 100_000, 6) .* 1e-5`.
  Apple Metal is Float32 only: `using Metal; v0 = MtlArray(rand(Float32, n, 6) .* 1f-5)`.
  AMDGPU and oneAPI work the same way. The first GPU call compiles the kernels.
- First calls in a session compile; benchmark the second call.

## 6. Tracking methods, steps, radiation

Set per element: `ele.tracking_method = MatrixKick()` (or as a keyword).

| Method | Use |
|---|---|
| `DriftKick()` | exact drifts + multipole kicks; any straight element without solenoid field (incl. RF) |
| `MatrixKick()` | energy-dependent quadrupole + kicks; best for strong straight aligned quadrupoles |
| `SolenoidKick()` | exact solenoid + kicks; any straight element with or without solenoid (incl. RF) |
| `BendKick()` | exact bend + kicks; elements with `g_ref` |
| `Implicit(use_newton=false)` | arbitrary fields from a four-potential (`FourPotentialParams`) |
| `Exact()` | thin elements, patches, maps, markers; exact drift, pure solenoid, pure dipole |
| `RK4()` | non-symplectic, arbitrary fields `em_field` (`EMFieldParams`); no spin |
| `SaganCavity(n_cells, L_active)` | RF cavities with energy gain and transverse focusing (one step, no slices, no ramping) |

The symplectic integrators take `order` (2, 4, 6, ...) and are combined with the number of
steps: `DriftKick(order=6)`. An order 4 step costs 5 second-order steps, an order 6 step 13;
whether more order or more steps is better depends on the element (check convergence).

`UniversalParams` per element: `n_steps` or `ds_step`, `n_slices` (collective effects and
ramping are applied per slice; for accuracy alone increase `n_steps`, which is cheaper),
`fringe_at`, `rad_damping`, `rad_fluctuations`:

```julia
for ele in ring.line
  ele.rad_damping = true          # deterministic damping
  ele.rad_fluctuations = true     # quantum excitation
end
```

Patches, maps and `SaganCavity` elements are one step (`n_steps = 1`, no slicing).

## 7. Intra-beam scattering

```julia
for ele in fodo
  ele.ibs_model = IBSModel.Gaussian   # or GaussianDampingOnly, GaussianFluctuationsOnly, None
end
res = track(fodo, v0=v0, weight=1e10/n_particles, n_turns=100)  # weight: real per macro
```

Computed from the bunch covariance at the start of each slice. Weights may be a vector; lost
particles don't count.

## 8. Time dependence and ramping

```julia
t = Time()                                         # absolute time [s]
kick = VKicker(L=0.5, Ks0=1e-4*sin(1e6*t))          # any parameter, even offsets
fodo.E_ref = 18e9 + 4e11*Time()                     # energy ramp
res = track(fodo, v0=v0, n_turns=1000, ramp_particle_energy_without_rf=true)  # no RF in fodo
```

- Evaluated per particle at its own time, at the entrance of each element slice (constant
  through it): use `n_slices` for fast variations.
- With a ramp and no RF, `ramp_particle_energy_without_rf=true`, else `pz` falls as the
  reference rises.
- `L` and `g_ref`/`angle` cannot depend on time. `SaganCavity` does not support ramping.
- A `TimeFunction` can be called: `tf = 1e-4*sin(1e6*t); tf(0.0)`.

## 9. BatchParam

A `BatchParam` gives particle `i` the `i`-th value (cycling if shorter), on CPU and GPU:

```julia
fodo.context.K1 = BatchParam(collect(0.1:0.05:0.55))   # 10 lattices
v0 = repeat([1e-3 0 0 0 0 1e-3], 10, 1)                # one particle each
res = track(fodo, v0=v0)
# k particles per lattice: BatchParam(repeat(values, inner=k))
```

Limits: not for `n_steps`/`ds_step`/`n_slices`; one tracking method for all; a `BatchParam` may
appear inside a `Time()` expression, but not contain time functions (write
`BatchParam([1.,0.])*sin(a*t) + BatchParam([0.,1.])*cos(b*t)`). `twiss` uses one entry
(`batch_index`, default 1) and does not support batched lengths or reference energy.
`dynamic_aperture` errors on `BatchParam`s. With your own `Bunch` and batched lengths, give
`Bunch(v=v0, t_ref=BatchParam(zeros(n)))` (plus a batched `p_over_q_ref` for a batched energy).

## 10. Callbacks

`Bunch(...; callbacks=(f,))` runs `f(i, coords, p::TrackPoint)` at every integration step
(`n_slices*n_steps` per element; once for one-step elements). `coords.v[i, 1:6]`,
`coords.state[i]`, `coords.q`; `p.s`, `p.t_ref`, `p.step`, `p.beta_gamma_ref`,
`p.last_ds_step`, `p.last_g`. Inside misaligned elements, wrap with
`transforms_out!(i, coords, p)` … `transforms_in!(i, coords, p)` for lab coordinates. Keep
callbacks GPU-compatible if tracking on the GPU.

## 11. Closed orbit

```julia
co = find_closed_orbit(ring)          # (v0 = 1x6 orbit, coasting_beam, sol)
co.v0, co.coasting_beam, co.sol.retcode   # retcode 0x0 = converged
```

- `v0=` is the guess and **is overwritten with the result**.
- Without RF (coasting beam), δ stays fixed at the guess's `pz`: several rows of `v0` give the
  δ-dependent orbits in one batched solve (GPU-parallel with a `CuArray`).
- `BatchParam`s in the lattice: one orbit per row for each value.
- Keywords: `rf_on`, `reltol`, `abstol`, `maxiter`, `autodiff`, `coasting_beam`, `warn`.

## 12. Dynamic aperture

```julia
deltas = 0:0.002:0.01
x_da, y_da = dynamic_aperture(ring; n_turns=200, emit_1=2e-8, emit_2=1e-8, n_r=21, n_theta=9,
                              max_sig_x=30, max_sig_y=60, deltas=deltas, verbose=false)
# x_da[j, i], y_da[j, i]: first lost start along angle j for deltas[i], in sigmas (Inf: none lost)
```

Polar grid in units of the beam sizes at the start; `theta_lims=(0, pi)` by default; with
`delta_dependent_orbits` (default for coasting rings) each δ is centered on its closed orbit;
`backend=CUDABackend()` and `coordinates_number_type=Float32` for speed; other `TrackConfig`
keywords pass through.

## 13. NAFF

```julia
res = track(fodo, v0=v0, n_turns=1023)
tw = twiss(fodo)
β, α = tw.beta1[1], tw.alpha1[1]
q, amp = naff(res.x[turn=:] .- im .* (β .* res.px[turn=:] .+ α .* res.x[turn=:]))
```

One signal per row. The complex signal `x - i(βpx + αx)` gives the signed tune; a real signal
has ±q. Subtract a nonzero closed orbit first. `naff(data, n)` gives `n` frequencies. Runs
batched on a GPU array (not on Metal).
