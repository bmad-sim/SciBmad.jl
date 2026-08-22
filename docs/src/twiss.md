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

`twiss` computes the (parametric and nonlinear) Twiss parameters of a `Beamline`. It returns
a `Twiss` struct containing the (amplitude-dependent) (spin) tunes/slip, and a
[`DataFrame`](https://dataframes.juliadata.org/stable/) of the Twiss parameters at each
specified integration step. By default, lattice functions of the Sagan-Rubin/Edwards-Teng
coupling formalism are computed at *every* integration step.

We will use the FODO cell from the [Quickstart](quickstart.md) throughout this page:

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

tw = twiss(fodo)
```

## Choosing what is computed

If only a subset of quantities is needed, the `cols` keyword restricts the computation (and
the columns returned) to exactly what you ask for, which can save computation time:

```{code-cell} julia
#=
  Compute only specific periodic Twiss parameters
  "beta1" = horizontal-like beta
  "phi2" = vertical-like phase advance,
  "dx" = horizontal dispersion
=#
tw = twiss(fodo, cols=["beta1", "phi2", "dx"])
```

The columns always present are set by `base_cols`, which defaults to
`["index", "name", "kind", "s"]`. Commonly requested `cols` include:

`beta1`, `beta2`, `alpha1`, `alpha2`
: Horizontal-like and vertical-like beta/alpha functions of the Sagan-Rubin/Edwards-Teng
  coupling formalism.

`phi1`, `phi2`, `phi3`
: Phase advances in units of $2\pi$.

`x`, `px`, `y`, `py`, `z`, `pz`
: The closed orbit canonical coordinates.

`c11`, `c12`, `c21`, `c22`, `gammac`, `N`, `Vi`
: The Sagan-Rubin coupling matrix, coupling factor, and normalizing matrices.

`w1`, `w2` (and `w1a`/`w1b`/`w2a`/`w2b`)
: Montague functions; require `chrom > 1` and/or `order > 1`.

`nx`, `ny`, `nz`, `n0x`, `n0y`, `n0z`
: Invariant spin field components and the closed-orbit periodic spin direction; require
  `spin=true`.

`h<ijkl>` / `h<ijklmn>`
: Resonance driving terms / detune coefficients (Bengtsson monomials).

For the complete list — including the De Moivre-Ripken matrices and the normalizing maps —
see the extended help of the docstring, available in a Julia session with `??twiss`, or the
{external:doc}`API Reference <index>`.

To compute the Twiss parameters only at certain places (or nowhere, when only the tunes are
wanted), use the `at` keyword argument, which accepts `LineElement`s, beamline indices,
and/or tuples giving s-ranges:

```{code-cell} julia
twiss(fodo, at=[qf, qd], cols=["beta1", "beta2"])
```

## Nonlinear Twiss

Beyond the purely linear optics, `twiss` can also compute higher-order and nonlinear
quantities by increasing the order of the underlying truncated power series (DA) map used
internally. All quantities are computed using SciBmad's Lie algebraic normal form analysis
package [`NonlinearNormalForm`](https://github.com/bmad-sim/NonlinearNormalForm.jl). `twiss`
exposes two independent "orders" that can be set:

- `chrom`, the order to which the energy deviation δ is truncated, for computing higher order
  chromatic quantities
- `order`, the truncated order of all individual phase space variables, for computing
  amplitude-dependent tunes and resonance driving terms

For example, setting `chrom=2` will compute the chromaticities, and if requested in `cols`
the Montague `w1`/`w2` functions, second-order dispersions `dx_2`/`dy_2`, chromatic beta beat
`dbeta1`/`dbeta2` (equivalent to writing `dbeta1_1`/`dbeta2_1`), etc. In fact, we can request
a chromatic derivative of *any* scalar-valued quantity, using the notation
`d<quantity>_<order>`. By omitting the `_<order>`, order is assumed to be one. For example,
`dc11` is the first derivative w.r.t. δ of the coupling matrix component `c11`, and `dc11_2`
is the second derivative w.r.t. δ.

```{code-cell} julia
#=
  Compute higher order chromatic quantities such as
  the chromaticity, Montague W functions ("w1", "w2"),
  and 2nd order dispersions ("dx_2", "dy_2") by setting
  the order of δ (chrom) equal to 2
=#
tw = twiss(fodo, cols=["w1", "w2", "dx_2", "dy_2"], chrom=2)
```

Amplitude- and energy-dependent quantities, such as the tunes `q1`/`q2`, are returned not as
plain numbers but as `AmplitudeDependentValue`s — Taylor series in the action-angle variables
`J₁`, `J₂` and the energy deviation δ:

```{code-cell} julia
print(tw.q2)
```

Individual terms of an `AmplitudeDependentValue` can be extracted with `getterm`, by
specifying the power of each variable you want. For example, the linear chromaticity in `y`
is the coefficient of `δ¹`:

```{code-cell} julia
# Get the y-chromaticity
chromy = getterm(tw.q2, delta=1)
```

The same mechanism extends naturally to purely amplitude-dependent tune shifts (once `order`
is raised high enough to resolve them), using the `J1` and `J2` keyword arguments to extract
coefficients of `J₁`/`J₂`.

Using the operator notation, the Bengtsson polynomial is defined as the polynomial $h$ in

$$
\mathcal{M} = \mathcal{A}_{cs}^{-1}\exp{(: h : )} \mathcal{R} \mathcal{A}_{cs}
$$

where $\mathcal{M}$ is the compositional operator representing the one turn map and
$\mathcal{A}_{cs}$ is the compositional operator representing only a linear (Courant-Snyder)
normalizing transformation. Monomials of $h$ are sometimes referred to as **resonance driving
terms** or **detune coefficients** depending on if they drive resonances or tune shifts with
amplitude.

We can extract any Bengtsson monomial by simply setting the `order` of `twiss` appropriately
(must be at least one less than the total order of the monomial). And, if the beam is
coasting, we can take chromatic derivatives as described before.

```{code-cell} julia
# Resonance driving terms h3000, h2100, require order=2
tw = twiss(fodo, cols=["h3000", "h2100"], order=2)
```

## Spin

`twiss` can additionally analyze the spin dynamics by setting `spin=true`. This enables
computation of the invariant spin field (as a Taylor series in the phase space coordinates)
and the amplitude-dependent spin tune:

```{code-cell} julia
#=
  Spin analysis (invariant spin field as Taylor series,
  amplitude-dependent spin tune).
=#
tw = twiss(fodo, cols=["nx", "ny", "nz"], spin=true,
      as_taylor_series=true, order=2)
n = [tw.nx, tw.ny, tw.nz] # ISF
```

With `as_taylor_series=true`, the components of the ISF are returned as full Taylor series in
the phase space variables $(x, p_x, y, p_y, z, p_z)$, rather than just returning
$\hat{n}_0$. The spin tune, accessible as `tw.qspin`, is likewise an
`AmplitudeDependentValue`:

```{code-cell} julia
print(tw.qspin)
```

As with the orbital tunes, individual terms — such as the spin tune's linear dependence on
energy, or on its amplitude `J₂` — can be pulled out with `getterm`.

## Parameter dependence

By passing a GTPSA `Descriptor` that includes parameters to the `GTPSA_descriptor` keyword
argument, every quantity `twiss` computes becomes a Taylor series in those parameters as
well, so that e.g. $\partial q_1 / \partial k_{qf}$ is available exactly. See
[Parametric Normal Form](parametric-nf.md), and [Optimization with Autodiff](optimize.md) for
using those derivatives in an optimizer.

:::{seealso}
The full `twiss` docstring is in the {external:doc}`API Reference <index>`. The
[Nonlinear Twiss](examples/julia/nonlinear-twiss.ipynb) notebook works through a full ring.
:::
