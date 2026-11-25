# `twiss`: optics, chromatic and nonlinear quantities, spin

Docs pages: `twiss.md` (the model page), `parametric-nf.md`, `quickstart.md`.

## Contents
1. What `twiss` does
2. Summary values and default columns
3. Reading columns at locations
4. Rows (`at`) and columns (`cols`)
5. Truncation orders: chromatic and amplitude-dependent quantities
6. RF cavities, synchrotron motion, radiation
7. Spin
8. Open lines (`initial`, `a_initial`)
9. A part of a beamline (`start`, `stop`)
10. Resonance driving terms (`h` columns)
11. Parametric normal form by hand (GTPSA)
12. Keyword summary

## 1. What it does

For a periodic beamline, `twiss` finds the closed orbit, computes the one-turn map as a Taylor
series around it (automatic differentiation), computes its normal form (tunes, normalizing map
`a`), tracks the normalizing map through the beamline, and computes the lattice functions from
it at each row (Lie-series map methods; no integrals over the lattice are evaluated). The
tunes are those of the normal form; the lattice functions use the Edwards-Teng/Sagan-Rubin
definitions. Mode 1 is
horizontal-like, mode 2 vertical-like, mode 3 longitudinal (only with RF). For an open line the
start is given instead (section 8).

## 2. Summary values and default columns

```julia
tw = twiss(fodo)
tw.q1, tw.q2          # tunes in units of 2π, whole turns included
tw.etac, tw.alphac    # slip factor, momentum compaction (etac ≈ alphac − 1/γ²)
tw.df                 # the table as a DataFrame
```

| Summary value | Meaning |
|---|---|
| `q1`, `q2` | tunes (open line: phase advances) |
| `q3` | synchrotron tune (with RF; negative above transition) |
| `alphac`, `etac` | momentum compaction, slip factor |
| `qspin` | spin tune (`spin=true`) |
| `damp1`, `damp2`, `damp3` | damping decrements per pass (amplitude × e^(−damp)), with `rad_damping` |
| `dq1`, `dq2`, `dq1_2`, … | δ-derivatives of the tunes (need `chrom ≥ n+1`) |

Default columns: `s`, `index`, `name`, `kind`; `beta1`, `beta2`, `alpha1`, `alpha2`; `phi1`,
`phi2` (2π units, from the start of the table); orbit `x`, `px`, `y`, `py`, `z`, `pz`;
dispersion `dx`, `dpx`, `dy`, `dpy`; coupling `gammac`, `c11`, `c12`, `c21`, `c22`; `slip`
(−c × arrival-time change per δ from the start). With RF also `phi3`, `dampdec1..3`.

## 3. Reading columns at locations

```julia
tw.beta1[qd]          # beginning of qd (qd placed once)
tw.beta1[qd, :exit]   # end of qd (= beginning of the next element)
tw.beta1[qd, :]       # every integration step of qd, both ends included
tw.beta1[qd, 1]       # one step after the beginning of qd
tw.beta1[5]           # beginning of the element with beamline index 5 (NOT row 5)
tw.beta1[[d]]         # every placement of d
tw.beta1[begin], tw.beta1[end]   # beginning / end of the beamline (end-1 = last element)
tw.beta1["q*"]        # by name pattern
tw.beta1[[qf, qd]], tw.beta1[2:4]
tw.beta1[(2.0, 6.0)]  # every step in an s-range
tw.beta1[:]           # all rows
```

An element placed more than once must be indexed with `[[ele]]`. Names, vectors and ranges
always give a vector. A column used whole acts like the vector of rows: `maximum(tw.beta1)`,
`tw.dx ./ sqrt.(tw.beta1)`, `lines(tw.s, tw.beta1)` (Makie and Plots work), `collect(tw.beta1)`,
`tw.df.beta1[r]` for raw row `r`.

## 4. Rows and columns

```julia
tws = twiss(fodo, at=[qd], cols=["beta1", "dx"])  # only these rows and columns (faster)
twiss(fodo, at=[]).q1                             # summary values only (fastest)
```

`at`: elements (each placement's entrance), beamline indexes, names, s-range tuples; only those
locations can be read afterwards. A table of only the orbit (and, without RF, dispersion) is
cheaper.

Extra columns for `cols` (full list: `??twiss`):

| Columns | Meaning |
|---|---|
| `d<col>_<n>` (`dbeta1`, `dx_2`, `dc11`) | n-th δ-derivative of any numeric column (`_1` optional) |
| `w1`, `w2` (`w1a`, `w1b`, …) | Montague chromatic functions |
| `r11` … `r66` | linear transfer matrix from the start of the table to the row |
| `m11` … `m66` | one-turn matrix at the row (periodic only) |
| `zx`, `zpx`, `zy`, `zpy` | crab dispersion (with RF) |
| `n0x`, `n0y`, `n0z`, `nx`, `ny`, `nz` | spin closed-orbit direction, invariant spin field (`spin=true`) |
| `h<ijkl>`, `h<ijklmn>` | resonance driving terms / detuning (section 10) |
| `N`, `Vi`, `H1`…`H3`, `B1`…`B3`, `E1`…`E3`, `a`, `a0`, `a1`, `a2`, `as` | normal-form matrices and maps |

## 5. Orders

Two truncation orders: `chrom` for δ, `order` for all six coordinates (`chrom` defaults to
`order`). Orbit and dispersion are known to the order in δ; tunes, β, α, phases to one order
less.

| Quantity | Needs |
|---|---|
| `dq1` (`q1[delta=1]`), `dbeta1`, `dx_2`, `w1` | `chrom=2` |
| `dq1_2` (`q1[delta=2]`), `dbeta1_2`, `dx_3` | `chrom=3` |
| `q1[J1=1]`, `h2200` | `order=3` |
| `q1[delta=1, J1=1]` | `order=4` |
| `h<ijkl…>` | `order ≥ i+j+k+l+… − 1` |

General rule: n-th δ-derivative needs `chrom ≥ n+1` (orbit/dispersion: n); with action
derivatives, `order ≥ n+1+2·(number of J derivatives)`. Asking for an unresolved term errors
and names the order needed.

Without `order`/`chrom`, `twiss` picks the lowest orders the requested `cols` need
(`twiss(fodo, cols=["dx_2"])` uses `chrom=2`). Summary values are not columns: give `chrom=2`
for `dq1`.

At order ≥ 2 tunes are `AmplitudeDependentValue`s (Taylor series in δ, J₁, J₂):

```julia
tw3 = twiss(fodo, chrom=3)
tw3.q1[delta=2]          # ∂²q1/∂δ² (same as tw3.dq1_2)
getterm(tw3.q1, delta=2) # the printed coefficient = derivative / 2!
twiss(fodo, order=3).q1[J1=1]   # ∂q1/∂J1 (J: nonlinear normal-form action)
```

Chromatic expansions need a coasting beam: on a ring with RF give `rf_on=false`. Tunes at a
fixed δ on a coasting ring: `twiss(ring, delta0=1e-3)`.

## 6. RF, synchrotron motion, radiation

```julia
cav = RFCavity(L=0.1, voltage=5e6, harmon=15, zero_phase=PhaseRef.AboveTransition)
ring = Beamline([qf, d, b, d, qd, d, b, d, cav], species_ref=Species("electron"), pc_ref=18e9)
twr = twiss(ring)          # 6D: q3, phi3, dampdec columns
twr.q3                     # < 0 above transition
twiss(ring, chrom=2)               # ERROR: chromatic needs rf_on=false
twiss(ring, chrom=2, rf_on=false)  # OK
```

The dispersion d(closed orbit)/dδ is well defined only without synchrotron motion: compute it
with `rf_on=false`. With RF on there is one 6D closed orbit, and the `dx` … columns are an
approximation from the 6D normal form, valid when the synchrotron tune is much smaller than the whole
transverse tunes (1.5505 m against the exact 1.5374 m at a quadrupole of an 18 GeV FODO cell
with one cavity). Report the `rf_on=false` value as the dispersion, and call the other the
6D approximation.

Radiation damping: set `rad_damping=true` on the elements; then `damp1..3` appear. Use a
realistic voltage and a frequency that is a harmonic of the revolution frequency (`harmon`).
The `zero_phase` must match the ring's side of transition or the longitudinal motion is
unstable (closed orbit or normal form fails).

## 7. Spin

```julia
twsp = twiss(fodo, spin=true)
twsp.qspin               # spin tune (Taylor series in δ); horizontal bends: −aγθ/2π mod 1
twsp.n0y[qd]             # closed-orbit spin direction
tws2 = twiss(fodo, cols=["nx", "ny", "nz"], spin=true, as_taylor_series=true, order=2)
tws2.nx[1]               # invariant spin field as a Taylor series in (x, px, y, py, z, δ)
```

Read a Taylor series coefficient by monomial exponents: `t[[0,0,1,0,0,0]]` (∂/∂y), and its value
with `scalar(t)`.

## 8. Open lines

```julia
twl = twiss(line, initial=(beta1=10.0, alpha1=-1.0, beta2=5.0, alpha2=0.5, dx=0.2))
twl.beta1[end]
```

- `initial` names: `beta1`, `beta2` (required), `alpha1`, `alpha2`, `dx`, `dpx`, `dy`, `dpy`,
  `c11`, `c12`, `c21`, `c22`, `x`, `px`, `y`, `py`, `z`, `pz` (others zero). A `NamedTuple`, a
  `Dict` (also a Python dict), a `Context` (whose variables can be matching knobs), or a table
  row: `initial=twcell.df[1, :]` (continue a cell's periodic functions into a line).
- `initial` is without synchrotron motion: RF cavities with voltage need `rf_on=false`.
- For 6D, spin or nonlinear starts use `a_initial`, the normalizing map: the `a` column of a
  periodic `twiss` with `as_taylor_series=true`:
  `a = twiss(cell, cols=["a"], as_taylor_series=true, at=[1]).a[1]`, then
  `twiss(line, a_initial=a)`.

## 9. A part of a beamline

```julia
twp = twiss(cells, start=9)                 # periodic optics of the part from element 9 to the end
two = twiss(cells, start=9, stop=13, initial=(beta1=10.0, beta2=5.0))  # open line part
```

The part runs from the entrance of `start` to the entrance of `stop`. `[begin]`/`[end]` are its
ends; `s` and `index` stay those of the whole beamline; phases count from `start`; an element
must be placed once in the part.

## 10. `h` columns

`h<ijkl>` (coasting) and `h<ijklmn>` (with RF) are coefficients of the effective Hamiltonian h,
with one pass from the row back to it = exp(:h:) followed by the rotation, in the complex
variables z₁ = (x̃₁+ip̃₁)/√2 = √J₁ e^{−iφ₁}, z₂ = conjugate, z₃, z₄ for mode 2, … Terms with
equal exponents of each mode's pair (`h2200`, `h1111`) are detuning; the others drive
resonances. Thin-element first-order checks (β at the multipole, φ the phase advance in radians
from the row to it):
- sextupole: h3000 = K2L β₁^{3/2} e^{−3iφ₁}/(12√2), h2100 = 3·K2L β₁^{3/2} e^{−iφ₁}/(12√2),
  h1020 = −K2L β₁^{1/2} β₂ e^{−i(φ₁+2φ₂)}/(4√2)
- octupole: h2200 = K3L β₁²/16, h4000 = K3L β₁² e^{−4iφ₁}/96

The values are complex; exact drifts alone give detuning.

## 11. Parametric normal form with GTPSA

For derivatives in lattice parameters, prefer `TwissExpr` + `SciBmad.value_and_jacobian`
(`optimize.md`): it handles orders and cleanup. By hand:

```julia
dnf = Descriptor([1, 1, 1, 1, 1, 1], 2, [1, 1], 1)  # 6 vars to order 1, 2 params, total 2
dk = params(dnf)
qf.Kn1 = 0.36 + dk[1]
qd.Kn1 = -0.36 + dk[2]
twp = twiss(fodo, GTPSA_descriptor=dnf)   # results are Taylor series in dk
grad(twp.q1)                              # [∂q1/∂dk1, ∂q1/∂dk2]
grad(twp.beta1[qd])
scalarize!(fodo)                          # REQUIRED before any other twiss/track
```

Tunes and β come from first-order terms in the variables, so a first-order dependence on a
parameter needs total order 2 (variable × parameter). Mixing Descriptors aborts Julia (e.g.
`twiss(fodo)` before `scalarize!`).

## 12. Keywords

`at`, `cols`, `base_cols`, `start`, `stop`, `order`, `chrom`, `rf_on`, `spin`,
`as_taylor_series`, `initial`, `a_initial`, `delta0` (δ of a coasting orbit / 6D guess), `v0`
(closed-orbit guess), `GTPSA_descriptor`, `in_body_coordinates`, `batch_index`, `damping`,
`coast_tol`, `symplectic_tol`.
