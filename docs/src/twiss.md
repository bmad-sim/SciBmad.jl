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

(twiss)=
# Twiss

```{code-cell} julia
:tags: [remove-cell]
using SciBmad
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

`twiss` is a general, all-encompassing function for doing analysis of any beamline, open or
closed, via perturbation theory around the (closed) orbit. It computes the linear and nonlinear optics (e.g. amplitude-dependent tunes and resonance driving terms) and, with `spin=true`,
the invariant spin field and the amplitude-dependent spin tune. The dependence of all
of these on the relative momentum deviation δ (the coordinate `pz`) and on parameters of the
lattice (magnet strengths, lengths, etc) can also be computed.

## Basics

All `twiss` needs is a beamline:

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  d = Drift(L=1.0)
  qd = Quadrupole(Kn1=-0.36, L=0.5, n_steps=2)
  b = SBend(L=2.0, angle=pi/20)
end
fodo = Beamline([qf, d, b, d, qd, d, b, d], species_ref=Species("electron"), pc_ref=18e9)
tw = twiss(fodo)
```

The output is a `Twiss` structure. Shown first are the summary values, which describe the motion
through the whole beamline: in this case, the tunes `q1` and `q2`, the slip factor `etac` and the
momentum compaction `alphac`. Then follows a table of the s-dependent quantities, with one row per
location: by default, every integration step through every element.

If no initial lattice functions are provided (see [Open lines](#twiss.open)), `twiss` finds the
closed orbit and computes the periodic solution. The map once through the beamline is computed as a
Taylor series in the deviations of the six phase space coordinates from the closed orbit, using
high-order automatic differentiation, and is then brought to normal form. The normalizing map is the 
change of coordinates in which the motion is a rotation in each mode (a damped one with radiation 
damping): mode 1 (horizontal-like), mode 2 (vertical-like) and, with RF cavities (synchrotron motion), 
mode 3 (longitudinal-like). The normalizing map is then tracked through the beamline, and the lattice 
functions at each location are computed from it.

To first order, the default, the beta and alpha functions (Twiss parameters), the phase advances, the dispersion, the coupling along the beamline, and the tunes are computed. Summary values and columns carry the number of their mode, e.g. `q1` the horizontal-like tune and `beta2` the vertical-like beta function. Coupled motion is described in the Sagan-Rubin/Edwards-Teng formalism by default.

The summary values are properties of `Twiss`:

```{code-cell} julia
tw.q1, tw.q2 # (0.1486522740866844, 0.1274197024658691)
```

| Summary value | Meaning |
|---|---|
| `q1`, `q2` | The tunes of modes 1 and 2 in units of 2π, whole oscillations included (for an open line, the phase advances through it) |
| `q3` | The synchrotron tune (with RF cavities, see [Synchrotron motion and radiation](#twiss.rf)) |
| `alphac` | The momentum compaction: the relative change of the length of the orbit per unit δ |
| `etac` | The slip factor: the relative change of the time the particles take through the beamline per unit δ (to first order `alphac` − 1/γ², with γ the Lorentz factor) |
| `qspin` | The spin tune (with `spin=true`, see [Spin](#twiss.spin)) |
| `damp1`, `damp2`, `damp3` | The damping decrements of the three modes (with radiation damping, see [Synchrotron motion and radiation](#twiss.rf)): in one pass through the beamline, the amplitude of each mode decreases by the factor $e^{-\mathrm{damp}}$ |

The columns of the table are `TwissColumn`s, and are read at locations of the beamline in the same
way as `TrackColumn`s can be read:

```{code-cell} julia
tw.beta1[qd]    # 6.695 at the beginning of qd
tw.beta1[qd, :] # [6.695, 6.538, 6.695] at every integration step of qd, both ends included
tw.beta1[[d]]   # [14.56, 8.177, 6.695, 12.14] at every placement of d (four)
tw.beta1[end]   # 14.56 at the end of the beamline
```

All the ways to give a location:

| Location | Meaning |
|---|---|
| `tw.beta1[qd]`, `tw.beta1[5]` | At the beginning of an element, or of the element with that beamline index |
| `tw.beta1[[d]]` | At the beginning of every placement of an element |
| `tw.beta1[qd, :exit]` | At the end of the element (the beginning of the next one) |
| `tw.beta1[qd, :]` | At every integration step of the element, both ends included |
| `tw.beta1[qd, 1]` | One integration step after the beginning of the element |
| `tw.beta1[begin]`, `tw.beta1[end]` | At the beginning and the end of the beamline (`end` is the beamline index of the end: `end-1` is the last element) |
| `tw.beta1["q*"]` | At the elements found by name, see [Searching by Name](#search:name) |
| `tw.beta1[[qf, qd]]`, `tw.beta1[2:4]` | At each of these elements |
| `tw.beta1[(2.0, 6.0)]`, `tw.beta1[:]` | At every integration step in an s-range, or of the whole beamline |

If an element is placed more than once in the beamline, `tw.beta1[d]` is an error, because it
is not clear which placement is meant. `tw.beta1[[d]]` gives the values at all of them. Names,
vectors and ranges always give a vector, even if they match only one location.

Expressions for [matching](optimize.md) and indexing a `Beamline` use the same rules as above.

A column can also be used as a whole, as the vector of all rows of the table. For example:

```julia
using CairoMakie
lines(tw.s, tw.beta1)       # the beta function along the beamline
maximum(tw.beta1)           # 14.56
tw.dx ./ sqrt.(tw.beta1)    # [1.059, 1.059, ...] at every row
```
The underlying vector of a `TwissColumn` can be obtained with `collect(tw.beta1)`. The whole table, here `tw.df` is a [`DataFrame`](https://dataframes.juliadata.org/stable/). 

The default columns computed by `twiss` are:

| Columns | Meaning | Units |
|---|---|---|
| `s`, `index`, `name`, `kind` | The position along the beamline, and the beamline index, name and kind of the element | m |
| `beta1`, `beta2`, `alpha1`, `alpha2` | The Twiss parameters β and α of modes 1 and 2 | m, 1 |
| `phi1`, `phi2` | The phase advances from the start of the table (at its end, `q1` and `q2`) | 2π |
| `x`, `px`, `y`, `py`, `z`, `pz` | The orbit (`pz` is δ) | m, 1 |
| `dx`, `dpx`, `dy`, `dpy` | The dispersion: the derivatives of the orbit with respect to δ | m, 1 |
| `gammac`, `c11`, `c12`, `c21`, `c22` | The coupling: the factor γ and the coupling matrix C of the Sagan-Rubin formalism (all zero, and γ = 1, without coupling) | |
| `slip` | The change of the arrival time per unit δ from the start of the table, times −c (at its end, −c T `etac`, with T the time through the beamline; with synchrotron motion an approximation) | m |

(twiss.rows)=
## Choosing rows and columns

`at` chooses the rows and `cols` the columns:

```{code-cell} julia
tws = twiss(fodo, at=[qd], cols=["beta1", "dx"])
tws.beta1[qd] # 6.695
```

`at` is a vector of elements (the beginning of every placement), beamline indexes and names (the
beginning of the element), and s-ranges (every integration step in the range). Only these
locations can then be read, and computing fewer rows is faster. With `at=[]` only the summary
values are computed:

```{code-cell} julia
twiss(fodo, at=[]).q1 # 0.1486522740866844
```

A table of only the orbit (and, without RF cavities, the dispersion) is faster too. Besides the
default columns, `cols` can ask for these (all are listed in the `twiss` docstring at the end of
this page):

| Columns | Meaning |
|---|---|
| `d<column>_<n>`, e.g. `dbeta1`, `dx_2` | The n-th derivative of any numeric column with respect to δ (`_1` can be left out), see [Chromatic and amplitude-dependent quantities](#twiss.orders) |
| `w1`, `w2` | The Montague chromatic functions $W = \sqrt{a^2 + b^2}$, with $b = \frac{1}{\beta}\frac{\partial\beta}{\partial\delta}$ (`w1b`, `w2b`) and $a = \frac{\partial\alpha}{\partial\delta} - \frac{\alpha}{\beta}\frac{\partial\beta}{\partial\delta}$ (`w1a`, `w2a`) |
| `r11` … `r66` | The linear transfer matrix from the start of the table to the row |
| `m11` … `m66` | The transfer matrix once through a periodic beamline, from the row back to it |
| `zx`, `zpx`, `zy`, `zpy` | The crab dispersion (with synchrotron motion) |
| `n0x`, `n0y`, `n0z`, `nx`, `ny`, `nz` | The spin direction on the closed orbit, and the invariant spin field (with `spin=true`) |
| `h<ijkl>`, `h<ijklmn>` | Resonance driving terms and detuning, see [below](#twiss.h) |
| `N`, `Vi`, `H1` … `E3`, `a`, `a0`, `a1`, `a2`, `as` | Matrices and maps of the normal form |

(twiss.orders)=
## Chromatic and amplitude-dependent quantities

Beyond first order, the Taylor series give the dependence of the optics on δ (chromatic
quantities) and the nonlinear motion: amplitude-dependent tunes and resonance driving terms. For
example the chromaticities, the first derivatives of the tunes with respect to δ, need the
expansion to second order in δ, which the keyword argument `chrom` sets:

```{code-cell} julia
twc = twiss(fodo, chrom=2)
twc.dq1, twc.dq2 # (-0.10506231945773534, -0.1180234082730797)
```

`dq1` is the first derivative of `q1` with respect to δ. This naming works for any quantity:
`d<quantity>_<n>` is its n-th derivative with respect to δ, and `_1` can be left out. The
dispersion `dx` is itself the derivative of the orbit `x`. Derivatives of columns are asked for in
`cols`, e.g. the second derivative of `x` (the second-order dispersion) and the first derivative
of `beta1`:

```{code-cell} julia
twd = twiss(fodo, chrom=2, cols=["dx_2", "dbeta1"])
twd.dx_2[qd], twd.dbeta1[qd] # (4.366, 0.7423)
```

How far the expansion goes is set by two truncation orders: `chrom` for δ alone, and `order`
for all six phase space coordinates (`chrom` defaults to `order`).

The orbit and the dispersion are known to the order of the expansion in δ. The other quantities
(tunes, β, α, phase advances, ...) come from the first-order terms of the map, so they are known
to one order less in δ. With `chrom=2`, for example, the dispersion is known to second order in δ
(`dx_2`), but the tunes only to first order (the chromaticities).

The expansion in δ needs a beam without synchrotron motion, in which δ stays constant. For a
beamline with RF cavities, also give `rf_on=false`, which turns the cavities off (see
[Synchrotron motion and radiation](#twiss.rf)).

With `chrom` or `order` of 2 or more, the tunes are not numbers but Taylor series, and indexing
them gives their derivatives:

```{code-cell} julia
tw3 = twiss(fodo, chrom=3)
tw3.q1
```

```{code-cell} julia
tw3.q1[delta=2] # 3.304
```

These `AmplitudeDependentValue`s are Taylor series in δ and in the actions `J₁` and `J₂`, which
measure the amplitudes of modes 1 and 2. Indexing gives a derivative: `tw3.q1[delta=2]` is
$\partial^2 q_1/\partial\delta^2$, which can also be written `tw3.dq1_2`. For the coefficient of
δ² as it is printed, i.e. the derivative divided by 2!, use `getterm(tw3.q1, delta=2)`.

Derivatives with respect to the actions need `order`:

```{code-cell} julia
twiss(fodo, order=3).q1[J1=1] # 0.6817, ∂q1/∂J1
```

The truncation orders that some quantities need:

| Quantity | Needs |
|---|---|
| `dq1` (`q1[delta=1]`), `dbeta1`, `dx_2` | `chrom=2` |
| `dq1_2` (`q1[delta=2]`), `dbeta1_2`, `dx_3` | `chrom=3` |
| `q1[J1=1]`, `h2200` | `order=3` |
| `q1[delta=1, J1=1]` | `order=4` |

In general, an n-th derivative with respect to δ needs `chrom` of at least n+1 (for the orbit and
the dispersion, n). With derivatives with respect to the actions too, it needs `order` of at least
n+1, plus 2 for each of them. Asking for a derivative that the truncation does not resolve is an
error that says which order is needed.

Without `order` and `chrom`, `twiss` uses the lowest orders that the requested columns need:
`twiss(fodo, cols=["dx_2"])` uses `chrom=2`, and `twiss(fodo, cols=["h3000"])` `order=2`.
Summary values are not columns, so for the chromaticities alone give `chrom`, as above.

The maps can also be expanded in deviations of parameters of the lattice, e.g. a magnet
strength: all of these quantities are then Taylor series in the parameters too, see
[Parametric Normal Form](#pnf).

(twiss.rf)=
## Synchrotron motion and radiation

With RF cavities, the longitudinal motion oscillates too, and `twiss` computes it as mode 3:

```{code-cell} julia
@elements begin
  cav = RFCavity(L=0.1, voltage=5e6, harmon=15, zero_phase=PhaseRef.AboveTransition)
end
fodorf = Beamline([qf, d, b, d, qd, d, b, d, cav], species_ref=Species("electron"),
                  pc_ref=18e9)
twr = twiss(fodorf)
twr.q3 # -0.008711
```

The synchrotron tune `q3` is negative above transition, as here, and positive below it, because
the longitudinal motion turns the other way in the `z`-`pz` plane.

The table now also has the longitudinal phase advance `phi3` and the damping decrements
`dampdec1`, `dampdec2`, `dampdec3` from the start of the table. These are zero here, as there is
no radiation damping. Radiation damping is turned on per element with `rad_damping=true` (see
[Integration Steps and Slices](#track.steps)), and then also gives the summary values `damp1`,
`damp2`, `damp3`.

As the chromatic quantities need a beam without synchrotron motion, `twiss(fodorf, chrom=2)` is an
error. `twiss(fodorf, chrom=2, rf_on=false)` computes them with the RF cavities turned off.

(twiss.spin)=
## Spin

With `spin=true`, `twiss` also computes the spin motion:

```{code-cell} julia
twsp = twiss(fodo, spin=true)
twsp.qspin
```

The spin tune `qspin` is a Taylor series in δ here. With only horizontal bends, it is
$-a\gamma\,\theta/2\pi$ up to a whole number, where θ = π/10 is the total bending angle and
aγ = 40.85 for 18 GeV electrons ($a$ is the anomalous magnetic moment). It changes with δ as aγ
does.

The table now also has `n0x`, `n0y`, `n0z`: the spin direction on the closed orbit that repeats
itself after each pass (here vertical: `twsp.n0y[qd]` is −1). The invariant spin field extends
this direction to the phase space around the closed orbit: a spin along the field stays along the
field, at the particle's new position, after each pass. The columns `dnx`, `dny`, `dnz` are its
derivatives with respect to δ.

(twiss.open)=
## Open lines

An open line, such as a transfer line, has no closed orbit. Instead, the orbit and the lattice
functions at its start are given with `initial`, and the normalizing map at the start is built
from them:

```{code-cell} julia
@elements begin
  qa = Quadrupole(Kn1=0.36, L=0.5)
  qb = Quadrupole(Kn1=-0.36, L=0.5)
  qc = Quadrupole(Kn1=0.2, L=0.5)
end
line = Beamline([qa, d, qb, d, qc, d], species_ref=Species("electron"), pc_ref=18e9)
twl = twiss(line, initial=(beta1=10.0, alpha1=-1.0, beta2=5.0, alpha2=0.5, dx=0.2))
twl.beta1[end] # 12.30
```

The values are given by the names of the columns: the Twiss parameters `beta1` and `beta2`
(required), `alpha1` and `alpha2`, the dispersion `dx`, `dpx`, `dy`, `dpy`, the coupling `c11`,
`c12`, `c21`, `c22`, and the orbit `x`, `px`, `y`, `py`, `z`, `pz` (the others are zero if not
given). `initial` can also be a `Dict` (for example a Python dictionary), or a row of another
table, e.g. to continue the periodic lattice functions of a cell into a line that starts with it:

```{code-cell} julia
cell = Beamline([qa, d, qb, d], species_ref=Species("electron"), pc_ref=18e9)
twcell = twiss(cell)
twiss(line, initial=twcell.df[1, :], cols=["beta1", "alpha1", "beta2", "alpha2"])
```

Through the first four elements these are the periodic lattice functions of the cell. The
values may also be Taylor series in parameters, to see how the lattice functions along the line
change with them (see [Parametric Normal Form](#pnf)), and they can be knobs of a match (see
[Matching and Optimization](optimize.md)).

`initial` describes a beam without synchrotron motion, so a line with RF cavities needs
`rf_on=false`. For a start with synchrotron motion, spin, or nonlinear terms, give instead the
normalizing map at the start as `a_initial`. The normalizing map of a periodic beamline is the
`a` column of its `twiss` table (the map with `as_taylor_series=true`, else only its matrix):

```{code-cell} julia
a = twiss(cell, cols=["a"], as_taylor_series=true, at=[1]).a[1]
twiss(line, a_initial=a, cols=["beta1", "alpha1", "beta2", "alpha2"])
```

(twiss.part)=
## A part of a beamline

The keyword arguments `start` and `stop` restrict the analysis to a part of the beamline, e.g.
a single cell of a line of several cells, as a periodic beamline by itself:

```{code-cell} julia
cells = Beamline([qf, d, b, d, qd, d, b, d, qf, d, b, d, qd, d, b, d],
                 species_ref=Species("electron"), pc_ref=18e9)
twp = twiss(cells, start=9) # the second cell
twp.q1, twp.beta1[begin]    # (0.1487, 14.56), as for fodo
```

The part goes from the beginning of the element `start` to the beginning of `stop` (by default
the end of the beamline). Its table is read like that of the whole beamline, except that:

- `twp.beta1[begin]` and `twp.beta1[end]` are the beginning and the end of the part;
- locations with several values (`:`, an s-range, ...) give only those in the part;
- an element must be placed once in the part.

`s` and the beamline indexes stay those of the whole beamline, and the phases are counted from
`start`.

With `initial` or `a_initial`, the part is an open line that starts at `start`. For example, from
lattice functions given at the beginning of the second cell to the beginning of its `qd`
(beamline index 13):

```{code-cell} julia
two = twiss(cells, start=9, stop=13, initial=(beta1=10.0, beta2=5.0))
two.beta1[end] # 2.345
```

(twiss.h)=
## Resonance driving terms and detuning (`h` columns)

For a periodic beamline, the columns `h<ijkl>` (without synchrotron motion) and `h<ijklmn>`
(with it) are the coefficients of the nonlinear effective Hamiltonian $h$, e.g. in a cell of
thin quadrupoles with a thin sextupole:

```{code-cell} julia
tqf = Multipole(Kn1L=0.8)
tqd = Multipole(Kn1L=-0.7)
sx = Multipole(Kn2L=0.01)
thincell = Beamline([tqf, d, sx, tqd, d], species_ref=Species("electron"), pc_ref=1e9)
twh = twiss(thincell, cols=["beta1", "phi1", "h3000", "h2100"])
twh.h3000[1] # 0.0002407 - 0.001088im at the start
```

In normalized coordinates, one pass through the beamline from the row back to it is the motion
under $h$ for one unit of time, followed by the rotation by the tunes. In the operator notation
of the [Quickstart](quickstart.md),

$$
\mathcal{M} = \mathcal{A}^{-1}\exp{(:h:)}\,\mathcal{R}\,\mathcal{A}
$$

where $\mathcal{M}$ is the map around the closed orbit once through the beamline from the row
back to it, $\mathcal{A}$ the linear (Courant-Snyder) normalizing map at the row and
$\mathcal{R}$ the rotation by the tunes (with their dependence on δ and on parameters). $h$ is a
polynomial in the complex variables

$$
z_{2k-1} = \frac{\tilde{x}_k + i\tilde{p}_k}{\sqrt{2}} = \sqrt{J_k}\,e^{-i\phi_k}, \qquad
z_{2k} = \frac{\tilde{x}_k - i\tilde{p}_k}{\sqrt{2}} = \sqrt{J_k}\,e^{i\phi_k},
$$

where $\tilde{x}_k$, $\tilde{p}_k$ are the normalized coordinates of mode $k$, $J_k$ its action
and $\phi_k$ its angle. ($\mathcal{A}$ takes the normalized coordinates to the laboratory
coordinates: without coupling and dispersion, $\tilde{x}_1 = \sqrt{2J_1}\cos\phi_1$ and
$x = \sqrt{2\beta_1 J_1}\cos\phi_1$.) In these variables,

$$
h = \sum h_{ijklmn}\, z_1^i z_2^j z_3^k z_4^l z_5^m z_6^n ,
$$

and $\exp(:h:)$ is the motion given by Hamilton's equations
$d\tilde{x}_k/dt = \partial h/\partial\tilde{p}_k$,
$d\tilde{p}_k/dt = -\partial h/\partial\tilde{x}_k$.

The terms with equal exponents of $z_{2k-1}$ and $z_{2k}$ for every mode (e.g. `h2200`) change the
tunes with the amplitudes (detuning). The others drive resonances. The column `h<ijklmn>` needs
`order` of at least $i+j+k+l+m+n-1$.

To first order in their strengths, thin multipoles add their Hamiltonians to $h$. For a normal
multipole of order $n$ (`Kn<n>L`) this is $K_nL\,\mathrm{Re}\,(x+iy)^{n+1}/(n+1)!$, with
$x = \sqrt{\beta_1/2}\,(z_1e^{-i\varphi_1} + z_2e^{i\varphi_1})$ and
$y = \sqrt{\beta_2/2}\,(z_3e^{-i\varphi_2} + z_4e^{i\varphi_2})$. Here $\beta_k$ are the beta
functions at the multipole and $\varphi_k$ the phase advances (in radians) from the row to the
multipole, for uncoupled motion without dispersion at the multipole. For example:

- a thin sextupole gives $h_{3000} = K_2L\,\beta_1^{3/2}e^{-3i\varphi_1}/(12\sqrt{2})$,
  $h_{2100} = 3K_2L\,\beta_1^{3/2}e^{-i\varphi_1}/(12\sqrt{2})$ and
  $h_{1020} = -K_2L\,\beta_1^{1/2}\beta_2\,e^{-i(\varphi_1+2\varphi_2)}/(4\sqrt{2})$;
- a thin octupole gives $h_{2200} = K_3L\,\beta_1^2/16$ and
  $h_{4000} = K_3L\,\beta_1^2e^{-4i\varphi_1}/96$.

For the sextupole above:

```{code-cell} julia
β = twh.beta1[sx]
φ = 2pi*twh.phi1[sx]
twh.h3000[1], 0.01*β^1.5/(12*sqrt(2))*cis(-3φ)   # the column, and the formula
twh.h2100[1], 3*0.01*β^1.5/(12*sqrt(2))*cis(-φ)  # the same for h2100
```

Other nonlinear elements contribute too: the drifts, with their exact Hamiltonian, give
detuning without any multipole (in this cell, $h_{2200} = 0.27$ without the sextupole).

## Reference

```{docstring} twiss
```

```{docstring} TwissColumn
```
