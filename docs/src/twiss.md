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

`twiss` computes the Twiss parameters of a beamline: values for the whole ring, such as the tunes,
and a table of the lattice functions (beta functions, phase advances, dispersion, closed orbit,
...) along the beamline. By default these are the periodic functions of a ring, at every
integration step:

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

## Reading the results

The values for the whole ring are properties of the result, for example the tunes:

```{code-cell} julia
tw.q1, tw.q2 # (0.14865227408668, 0.1274197024658691)
```

Every column of the table is a property too, and is read at locations of the beamline:

```{code-cell} julia
tw.beta1[qd]        # 6.695 at the beginning of qd
tw.beta1[5]         # 6.695 at the beginning of the element with beamline index 5 (qd)
tw.beta1[qd, :exit] # 6.695 at its end (the beginning of the next element)
tw.beta1[qd, :]     # [6.695, 6.538, 6.695] at every integration step of qd, both ends included
tw.beta1[d]         # [14.56, 8.177, 6.695, 12.14] d is in the beamline four times
tw.beta1["q*"]      # [14.56, 6.695] at the elements whose names start with q
tw.beta1[end]       # 14.56 at the end of the beamline
```

All the ways to give a location:

| Location | Meaning |
|---|---|
| `tw.beta1[qd]`, `tw.beta1[5]` | At the beginning of an element, or of the element with that beamline index |
| `tw.beta1[qd, :exit]` | At the end of the element (the beginning of the next one) |
| `tw.beta1[qd, :]` | At every integration step of the element, both ends included |
| `tw.beta1[qd, 1]` | One integration step after the beginning of the element |
| `tw.beta1[begin]`, `tw.beta1[end]` | At the beginning and the end of the beamline |
| `tw.beta1["q*"]` | At the elements found by name, see [Searching by Name](beamline.md) |
| `tw.beta1[[qf, qd]]`, `tw.beta1[2:4]` | At each of these elements |
| `tw.beta1[(2.0, 6.0)]`, `tw.beta1[:]` | At every integration step in an s-range, or of the whole beamline |

An element that is in the beamline several times gives one value per placement, in beamline
order, and names, vectors and ranges always give a vector. These locations are the same
everywhere in SciBmad: the expressions used for matching are written in exactly the same way
(see [](#optimize)).

A column used as a whole is the column of the table, so it can be plotted or computed with
directly:

```julia
using CairoMakie
lines(tw.s, tw.beta1)       # the beta function along the beamline
maximum(tw.beta1)           # 14.56
tw.dx ./ sqrt.(tw.beta1)    # [1.059, 1.059, ...] at every row
```

The whole table is the `DataFrame` `tw.df`, with the beamline index, name, kind and `s` of each
row.

The chromaticities are `tw.dq1` and `tw.dq2`. They need `chrom` (or `order`) of at least 2, and
no synchrotron motion (`rf_on=false` if there are RF cavities):

```{code-cell} julia
twc = twiss(fodo, chrom=2)
twc.dq1, twc.dq2 # (-0.10506231945773534, -0.1180234082730797)
```

With `chrom` or `order` of 2 or more, the tunes are Taylor series in δ and in the amplitudes
(see the [Quickstart](quickstart.md)), and indexing gives their derivatives: `tw.q1[delta=2]` is
$\partial^2 q_1/\partial\delta^2$ (also written `tw.dq1_2`), `tw.q1[J1=1]` is
$\partial q_1/\partial J_1$ and `tw.q1[delta=1, J1=1]` is
$\partial^2 q_1/\partial\delta\partial J_1$. `getterm(tw.q1, delta=2)` is instead the
coefficient of δ² in the Taylor series as it is printed: the derivative divided by 2!. A
derivative of order n in δ needs `chrom` of at least n+1, and one in J₁ or J₂ `order` of at least
3; asking for one that the truncation does not resolve is an error:

```{code-cell} julia
twiss(fodo, chrom=3).q1[delta=2] # 3.304
```

```{code-cell} julia
twiss(fodo, order=3).q1[J1=1] # 0.6817
```

## Choosing rows and columns

By default `twiss` computes a standard set of columns at every integration step. `cols` chooses
the columns (all of them are listed in the reference at the end of this page), and `at` the rows:
elements (or their beamline indexes) for their beginnings, and s-ranges for every integration step
in them. Only these locations can then be read, and computing fewer rows is faster. Asking only for
the orbit (`x`, `px`, `y`, `py`, `pz`; with RF cavities also `z`, without them the dispersion) is
faster too, since then none of the optics is computed at the rows. With `at=[]` only the values for
the whole ring are computed:

```{code-cell} julia
twiss(fodo, at=[]).q1                    # 0.14865227408668
twiss(fodo, at=[qd], cols=["beta1"]).beta1[qd] # 6.695
```

The truncation orders are chosen from the columns: `twiss(fodo, cols=["dx_2"])` uses `chrom=2`
and `twiss(fodo, cols=["h3000"])` uses `order=2`. `order` and `chrom` (or a `GTPSA_descriptor`)
can still be given, for example for the chromaticities alone, which no column asks for:
`twiss(fodo, chrom=2, at=[]).dq1`.

`start` and `stop` compute only a part of the beamline, from the beginning of `start` to the
beginning of `stop` (for example one cell of a ring); `tw.beta1[begin]` and `tw.beta1[end]` are
then those of the part.

## Open lattices

By default `twiss` computes the periodic Twiss functions of a ring. For a line that is not 
periodic, such as a transfer line, give the Twiss parameters at its start with `initial`, by the
names of the columns: `beta1` and `beta2` (required), `alpha1`, `alpha2`, the dispersion `dx`,
`dpx`, `dy`, `dpy`, the coupling `c11`, `c12`, `c21`, `c22`, and the orbit `x`, `px`, `y`, `py`,
`z`, `pz` (all zero if not given):

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  d = Drift(L=1.0)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  q3 = Quadrupole(Kn1=0.2, L=0.5)
end
line = Beamline([qf, d, qd, d, q3, d], species_ref=Species("electron"), pc_ref=18e9)
twl = twiss(line, initial=(beta1=10.0, alpha1=-1.0, beta2=5.0, alpha2=0.5, dx=0.2))
twl.beta1[end] # 12.30
```

`initial` can also be a `Dict` (for example a Python dictionary), or a row of another table, to
continue the Twiss functions of a ring into a line that starts with one of its cells:

```{code-cell} julia
cell = Beamline([qf, d, qd, d], species_ref=Species("electron"), pc_ref=18e9)
twc = twiss(cell)
twiss(line, initial=twc.df[1, :], cols=["beta1", "alpha1", "beta2", "alpha2"])
```

Through the first four elements these are the periodic Twiss functions of the cell. The values
may also be Taylor series in GTPSA parameters, to see how the Twiss functions along the line change
with them (see [](#pnf)). `initial` describes a beam without synchrotron motion, so a line
with RF cavities needs `rf_on=false`. For a start with synchrotron motion, spin, or nonlinear
terms, give instead the map `a` from normal form coordinates to laboratory coordinates as
`a_initial`. The `a` of a periodic lattice is the `a` column of its `twiss` table:

```{code-cell} julia
a = twiss(cell, cols=["a"], as_taylor_series=true, at=[1]).a[1]
twiss(line, a_initial=a, cols=["beta1", "alpha1", "beta2", "alpha2"])
```

The same works for matching: the Twiss functions at the end of the line can be matched to a target
as in [](#optimize), and the Twiss parameters at the start can be knobs themselves (see there).

## Resonance driving terms and detuning (`h` columns)

The columns `h<ijkl>` (coasting beam) and `h<ijklmn>` (with synchrotron motion) are the
coefficients of the nonlinear effective Hamiltonian $h$ at the row: going once around the ring
from the row, the particles first move as $h$ makes them move in one unit of time, and then
follow the linear one-turn motion. In the operator notation of the
[Quickstart](quickstart.md),

$$
\mathcal{M} = \mathcal{A}^{-1}\exp{(:h:)}\,\mathcal{R}\,\mathcal{A}
$$

where $\mathcal{M}$ is the one-turn map around the closed orbit, $\mathcal{A}$ the linear
(Courant-Snyder) normalizing map at the row and $\mathcal{R}$ the rotation by the tunes (with
their dependence on δ and on GTPSA parameters). $h$ is a polynomial in the complex variables

$$
z_{2k-1} = \frac{\tilde{x}_k + i\tilde{p}_k}{\sqrt{2}} = \sqrt{J_k}\,e^{-i\phi_k}, \qquad
z_{2k} = \frac{\tilde{x}_k - i\tilde{p}_k}{\sqrt{2}} = \sqrt{J_k}\,e^{i\phi_k},
$$

with $\tilde{x}_k$, $\tilde{p}_k$ the normalized coordinates of mode $k$ ($\mathcal{A}$ applied to
them gives the laboratory coordinates; $\tilde{x}_1 = \sqrt{2J_1}\cos\phi_1$ and
$x = \sqrt{2\beta_1 J_1}\cos\phi_1$ without coupling and dispersion), $J_k$ its action and
$\phi_k$ its angle:

$$
h = \sum h_{ijklmn}\, z_1^i z_2^j z_3^k z_4^l z_5^m z_6^n ,
$$

and $\exp(:h:)$ is the motion given by Hamilton's equations
$d\tilde{x}_k/dt = \partial h/\partial\tilde{p}_k$,
$d\tilde{p}_k/dt = -\partial h/\partial\tilde{x}_k$.
The terms with equal exponents of $z_{2k-1}$ and $z_{2k}$ for every mode (e.g. `h2200`) change
the tunes with the amplitudes (detuning); the others drive resonances.

To first order in their strengths, thin multipoles add their Hamiltonians to $h$. For a normal
multipole of order $n$ (`Kn<n>L`) this is $K_nL\,\mathrm{Re}\,(x+iy)^{n+1}/(n+1)!$, with
$x = \sqrt{\beta_1/2}\,(z_1e^{-i\varphi_1} + z_2e^{i\varphi_1})$ and
$y = \sqrt{\beta_2/2}\,(z_3e^{-i\varphi_2} + z_4e^{i\varphi_2})$, where $\beta_k$ are the beta
functions at the multipole and $\varphi_k$ the phase advances (in radians) from the row to the
multipole (for uncoupled motion without dispersion at the multipole). For example, a thin
sextupole gives $h_{3000} = K_2L\,\beta_1^{3/2}e^{-3i\varphi_1}/(12\sqrt{2})$,
$h_{2100} = 3K_2L\,\beta_1^{3/2}e^{-i\varphi_1}/(12\sqrt{2})$ and
$h_{1020} = -K_2L\,\beta_1^{1/2}\beta_2\,e^{-i(\varphi_1+2\varphi_2)}/(4\sqrt{2})$, and a thin
octupole $h_{2200} = K_3L\,\beta_1^2/16$ and $h_{4000} = K_3L\,\beta_1^2e^{-4i\varphi_1}/96$.
Other nonlinear elements contribute too (the exact drifts give a small $h_{2200}$ without any
multipole). With a thin sextupole in a cell:

```{code-cell} julia
qf = Multipole(Kn1L=0.8)
qd = Multipole(Kn1L=-0.7)
d = Drift(L=1.0)
sx = Multipole(Kn2L=0.01)
ring = Beamline([qf, d, sx, qd, d], species_ref=Species("electron"), pc_ref=1e9)
tw = twiss(ring, rf_on=false, order=2, cols=["beta1", "phi1", "h3000"])
β = tw.beta1[sx]
φ = 2pi*tw.phi1[sx]
tw.h3000[1], 0.01*β^1.5/(12*sqrt(2))*cis(-3φ)
```

The column `h<ijklmn>` needs `order` of at least $i+j+k+l+m+n-1$.

## Reference

```{docstring} twiss
```

```{docstring} TwissColumn
```
