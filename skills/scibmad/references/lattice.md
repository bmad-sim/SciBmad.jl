# Lattices: elements, beamlines, deferred expressions

Docs pages: `element.md`, `beamline.md`, `defexpr.md`, `def-tools.md`, `coordinates.md`,
`ring-design/coordinates.md`, `ring-design/control.md`.

## Contents
1. Elements and kinds
2. Parameter names (by parameter group)
3. Beamlines, reference particle, finding elements
4. Deferred expressions and contexts
5. Phase trombones and transport maps
6. Translating from Bmad / MAD-X

## 1. Elements and kinds

Every element is a `LatElement`. A "kind" constructor (`Quadrupole(...)`) only sets `kind` and a
default `tracking_method`; it takes any parameter as a keyword. Wrap definitions in `@elements`
so that each element's `name` is its variable name:

```julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  d  = Drift(L=1.2)
  b  = SBend(L=1.2, angle=pi/132)     # angle sets g_ref = angle/L and the field Kn0 = g_ref
end
qf.Kn1 = 0.37        # set any parameter afterwards; reflected in every placement
d.Ks21 = -200        # any element can carry any multipole
```

| Kind | Default tracking method | Kind | Default tracking method |
|---|---|---|---|
| `Drift` | `DriftKick()` | `Solenoid` | `SolenoidKick()` |
| `Quadrupole` | `MatrixKick()` | `Kicker`, `HKicker`, `VKicker` | `DriftKick()` |
| `Sextupole`, `Octupole`, `Multipole` | `DriftKick()` | `RFCavity` | `SolenoidKick()` |
| `SBend` | `BendKick()` | `CrabCavity` | `SolenoidKick()` (sets `is_crabcavity = true`) |
| `Marker`, `BPM`, `Patch`, `Map` | `Exact()` | `LatElement()` | none: set one before tracking |

Any element with `L = 0` uses `Exact()` (thin multipoles: `Multipole(Kn2L=0.01)`). Changing
`kind` afterwards does not change the tracking method. A tracking method that cannot handle the
element's fields errors at tracking time and names the one to use (e.g. a solenoid field in a
`Drift` needs `tracking_method = SolenoidKick()`). Details of the methods: `tracking.md`.

A parameter group is created when one of its properties is set; remove a whole group with
`ele.MapParams = nothing` (any group name). `propertynames(ele)` lists every settable name.

## 2. Parameter names

**Multipoles (`BMultipoleParams`)**, orders `n = 0 … 21`:
- `Kn<n>`, `Ks<n>`: normal/skew strength normalized by the reference rigidity (`Kn1` 1/m²,
  `Kn2` 1/m³, `Kn0` 1/m). `Kn<n>L`, `Ks<n>L`: integrated. `Bn<n>`, `Bs<n>` (and `...L`):
  unnormalized fields (T/m^n). Solenoid: `Ksol`, `KsolL`, `Bsol`, `BsolL`.
- `tilt<n>`: rotation of multipole `n`. Electric multipoles: `En<n>`, `Es<n>` (`EMultipoleParams`).
- Whichever form was *set last* for an order is stored; the others are computed. If `Kn1` is
  stored, `Bn1` follows the reference energy; if `Bn1` is stored, `Kn1` changes with it instead.
  Same order normal and skew share one form.

**Bends (`BendParams`)**: `g_ref` (curvature of the reference coordinate system, 1/m), `angle`
(write-only: sets `g_ref = angle/L` and `Kn0 = g_ref`; read `g_ref*L` instead), `tilt_ref`,
edge angles `e1`, `e2`, edge field integrals `edge1_int`, `edge2_int` (= Bmad `fint*hgap`).
The dipole field is `Kn0` (set it separately from `g_ref` for a bend whose field differs from
its geometry). `L` and `g_ref` cannot be time dependent.

**Alignment (`AlignmentParams`)**, relative to the element center: `x_offset`, `y_offset`,
`z_offset`, `x_rot`, `y_rot`, `tilt` (rotations applied in the order `tilt`, `x_rot`, `y_rot`).

**RF (`RFParams`)**: `voltage` (V), `rf_frequency` (Hz) *or* `harmon` (the one set last is
stored), `phi0` (rad), `zero_phase` (`PhaseRef.Accelerating` default, `PhaseRef.AboveTransition`,
`PhaseRef.BelowTransition`), `traveling_wave`, `is_crabcavity`. Energy gain of the reference
species arriving Δt late: `|q| V cos(phi0 + φ_zero + ω Δt)`, φ_zero = 0, +π/2, −π/2 for
Accelerating, AboveTransition, BelowTransition. For a storage ring with `phi0 = 0`, use the
`zero_phase` of the ring's side of transition: above transition when the slip factor
`twiss(ring, rf_on=false).etac` is positive (e.g. electron rings), below when negative (e.g. a
ring without bends). `dE_ref` on an element changes the reference energy there (must be
the first element of a `Beamline`; see the SaganCavity docs pages).

**Universal (`UniversalParams`)**: `name`, `kind`, `L`, `tracking_method`, `n_steps` or
`ds_step` (setting one sets the other), `n_slices`, `fringe_at` (`Fringe.BothEnds`,
`EntranceEnd`, `ExitEnd`, `NoEnd`), `rad_damping`, `rad_fluctuations`.

**Apertures (`ApertureParams`)**: `x1_limit`, `x2_limit`, `y1_limit`, `y2_limit` (m),
`aperture_shape` (`ApertureShape.Elliptical` default, `Rectangular`), `aperture_at`
(`ApertureAt.Entrance` default, `Exit`, `BothEnds`), `aperture_shifts_with_body`.

**Patches (`PatchParams`)**: `dx`, `dy`, `dz`, `dx_rot`, `dy_rot`, `dz_rot` (applied as
`dz_rot`, `dx_rot`, `dy_rot`), `dt`.

**Others**: `MapParams` (`transport_map(v, q, p)`, `transport_map_params`), `EMFieldParams`
(`em_field(x, y, s, t, p)` for `RK4`), `FourPotentialParams` (for `Implicit`), `IBSParams`
(`ibs_model`), `MetaParams` (`alias`, `label`, `description`).

**In a beamline only** (`BeamlineParams` of a placement): `s`, `s_downstream`,
`beamline_index`, `beamline`. Asking the parent element for them errors.

## 3. Beamlines

```julia
fodo = Beamline([qf, d, b, d], species_ref=Species("electron"), pc_ref=18e9)
```

Reference energy: one of `E_ref` (total energy, eV), `pc_ref` (eV), `p_over_q_ref` (signed
rigidity, T·m), or put them on the first element. `Species("electron")`, `"positron"`,
`"proton"`, `"muon"`, ... (AtomicAndPhysicalConstants). Other species can still be tracked
(`track(...; species=...)`): the magnets keep their fields.

Placing an element makes a child that inherits everything from the element (sets pass through
to the parent). `drifts[1] === drifts[2]` is false, but setting `d.L` changes both.
`ele.parent` gives the parent. A `Beamline` iterates like its element vector
(`for ele in fodo`, `fodo.line`, `length(fodo)`); a new beamline from the same elements can share
the context: `Beamline(elements; species_ref, pc_ref, context=old.context)`.
`deepcopy.(bl.line)` makes independent elements (e.g. for a ring with errors). The total
length is `bl[end].s_downstream`.

**Indexing a beamline** (the same rules as reading `twiss` and `track` results):

| Index | Gives |
|---|---|
| `fodo[qf]` | the single placement of `qf` (error if not placed exactly once) |
| `fodo[[d]]`, `fodo[[qf, 4]]` | every placement of `d`; a mix of elements and indexes |
| `fodo[4]` | the element at beamline index 4 |
| `fodo[x -> x.kind == "Quadrupole"]` | elements for which the function is true |
| `fodo["q*"]`, `fodo["?d"]` | names: `*` any characters, `%` or `?` one character |
| `fodo["Sextupole::*"]` | by kind |
| `fodo["d#2"]` | the 2nd match |
| `fodo["qf:qd"]`, `fodo["5:9"]` | a range (wraps around if the end comes first) |
| `fodo["qf, qd"]`, `fodo["Sextupole::* & qf:qd"]` | union, intersection |
| `fodo[r"(q\|s)d"]`, `fodo[r"Q.*"i]` | regex on the whole name |

Results are in beamline order, each element at most once. `findelements(fodo, "q*")` is the
same; on a `Lattice`, `"branch>>name"` restricts to a branch.

## 4. Deferred expressions and contexts

A `DefExpr` wraps a function of a context and is evaluated whenever the parameter is read:

```julia
fodo.context.kq = 0.36                 # each Beamline has a context
qf.Kn1 = DefExpr(c -> c.kq)
qd.Kn1 = DefExpr(c -> -c.kq)
fodo.context.kq = 0.37                 # both quadrupoles follow
fodo[qd].Kn1                           # -0.37 (read through the placement)
```

- Lookup: the beamline's context, then `DEFAULT_CONTEXT`. The element you defined (`qf`) is the
  parent of its placements and is in no beamline, so `qf.Kn1` evaluates with `DEFAULT_CONTEXT`
  and errors if `kq` is only in `fodo.context`: read `fodo[qf].Kn1`.
- Try values temporarily: `push!(ctx; kq=0.4)` … `pop!(ctx)`, or
  `withvalues(ctx; kq=0.4) do ... end` (restores even on error).
- Vector variables: `ctx.kicks = zeros(10)` and `DefExpr(c -> c.kicks[i])` per element; then
  `@knob ctx.kicks` varies every entry in a match.
- `rawget(ele, :Kn1)` returns the stored `DefExpr` (to build on it:
  `qf.Kn1 = 1.1*rawget(qf, :Kn1)`); `deval(expr, ctx)` evaluates it. `rawget` errors for a
  derived parameter (e.g. `Kn1L` when `Kn1` is stored).
- `Context(beta1=10.0, alpha1=0.0)` makes a standalone context (e.g. for `initial` in matching).
- Expressions are evaluated once per `track`/`twiss` call. Time dependence during tracking is a
  `Time()` expression instead (`tracking.md`).
- For matching, don't restrict a context or expression to a number type (`Context{Float64}`,
  `DefExpr{Float64}`): knobs put Taylor series through them.

## 5. Phase trombones and transport maps

`phase_trombone!(bl[marker]; phi1=0.1, phi2=-0.05)` turns an element into a linear map that
changes the phase advances (in units of 2π) without changing the lattice functions. The map is
built from the closed orbit and the linear optics at that moment (keyword `rf_on`): remove it
(`bl[marker].MapParams = nothing`) before changing the orbit or optics, then call it again.
To adjust it later, give the phases as context expressions:
`phase_trombone!(bl[tr]; phi1=DefExpr(c -> c.tr_phi1), phi2=DefExpr(c -> c.tr_phi2))`.

A general map: `LatElement(transport_map=f, transport_map_params=p)` with
`f(v, q, p) -> (v_new, q)`, `v` a 6-tuple, `q` `nothing` or a quaternion 4-tuple. Keep it type
stable and branchless for GPU use (e.g. a `StaticArrays` matrix in `p`).

## 6. Translating from Bmad / MAD-X

| Bmad / MAD-X | SciBmad |
|---|---|
| `k1`, `k2`, ... (Bmad, MAD-X) | `Kn1`, `Kn2` |
| `k1l`, MAD-X `knl[n]`, Bmad multipole `knl` | `Kn1L`, `Kn<n>L` |
| skew `k1s` / `ksl[n]` | `Ks1`, `Ks<n>L` |
| `ks` (solenoid) | `Ksol` |
| `g`, `angle`, `e1`, `e2`, `ref_tilt` | `g_ref`, `angle`, `e1`, `e2`, `tilt_ref` |
| `fint*hgap` | `edge1_int`, `edge2_int` |
| `x_offset`, `y_offset`, `tilt` | same names |
| Bmad `x_pitch` | `y_rot` |
| patch `x_offset`, `y_offset`, `z_offset`, `x_pitch`, `y_pitch`, `tilt` | `dx`, `dy`, `dz`, `dy_rot`, `dx_rot`, `dz_rot` |
| `num_steps`, `ds_step` | `n_steps`, `ds_step` |
| `rf_frequency`, `harmon`, `voltage` | same names |
| Bmad `phi0` (units of 2π) | `phi0` in **radians**, relative to `zero_phase` |
| `p0c`, `E_tot` | `pc_ref`, `E_ref` (on the `Beamline`) |

When comparing with Tao, converge Bmad's integration first: Bmad tracks a thick sextupole in
one step by default (`num_steps`), which shifted the chromaticities of a test cell by 3.5e-4
(0.297094 vs 0.296744); `sextupole[num_steps] = 20` in the Bmad file gives SciBmad's values.
SciBmad's defaults were converged there (1 to 64 steps changed nothing above 3e-14).

Bmad's lcavity and rfcavity phase conventions differ from each other; check the RF phase of a
translated cavity with the convention in section 2 (e.g. the energy gain of one particle with
`track`), and the sign conventions of rotations with a single tracked particle.
