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

```{docstring} twiss
```

## Open lattices

By default `twiss` computes the periodic Twiss functions of a ring. For a line that is not 
periodic, such as a transfer line, give the Twiss functions at its start with `a_initial`: the 
map `a` from normal form coordinates to laboratory coordinates. The `a` of a periodic lattice is 
the `a` column of its `twiss` table. Here the periodic Twiss functions of a FODO cell continue 
into a line that starts with the cell:

```{code-cell} julia
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  d = Drift(L=1.0)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  q3 = Quadrupole(Kn1=0.2, L=0.5)
end
cell = Beamline([qf, d, qd, d], species_ref=Species("electron"), pc_ref=18e9)
a = twiss(cell, cols=["a"], as_taylor_series=true, at=[1]).a[1]

line = Beamline([qf, d, qd, d, q3, d], species_ref=Species("electron"), pc_ref=18e9)
twiss(line, a_initial=a, cols=["beta1", "alpha1", "beta2", "alpha2"])
```

Through the first four elements these are the periodic Twiss functions of the cell. The same 
works for matching: the Twiss functions at the end of the line can be matched to a target as in 
[](#optimize), with `a_initial` given to `twiss`.

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
