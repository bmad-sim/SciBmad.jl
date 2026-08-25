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

# Quickstart Guide

The following is a tour of SciBmad in one block - build a FODO cell, retune it, compute linear, 
nonlinear, chromatic and spin Twiss parameters, and track particles with spin. 
The rest of this page walks through it step by step.

```{code-cell} julia
:tags: [remove-cell]
# Hidden setup: keep printed tables narrow enough for the page.
ENV["COLUMNS"] = 100
ENV["LINES"] = 30
```

## A Simple Copy-Paste Example

```julia
using SciBmad

# Define lattice elements using @elements
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-1.2, L=0.2)
end

# Construct a Beamline and define the reference species + momentum
fodo = Beamline([qf, sf, b, d, qd, sd, b, d],
        species_ref=Species("electron"), pc_ref=18e9)

# Change the quadrupole strengths
qf.Kn1 = 0.37
qd.Kn1 = -0.37

# Find the drifts `d` in `fodo`
ds = fodo[d];

# Get the first element of `fodo`
first_ele = fodo[1];

# Gets all quadrupoles
quadrupoles = fodo[x -> x.kind == "Quadrupole"];

# Compute the periodic Twiss parameters
tw = twiss(fodo)

#=
  Compute only specific periodic Twiss parameters
  "beta1" = horizontal-like beta
  "phi2" = vertical-like phase advance,
  "dx" = horizontal dispersion
=#
tw = twiss(fodo, cols=["beta1", "phi2", "dx"])

#=
  Compute higher order chromatic quantities such as
  the chromaticity, Montague W functions ("w1", "w2"),
  and 2nd order dispersions ("dx_2", "dy_2") by setting
  the order of δ (chrom) equal to 2
=#
tw = twiss(fodo, cols=["w1", "w2", "dx_2", "dy_2"], chrom=2)
print(tw.q2)
#= Prints:
  AmplitudeDependentValue:
   0.05253999423230923 - 0.00471066037657978 δ
=#
# Get the y-chromaticity
chromy = getterm(tw.q2, delta=1)

# Resonance driving terms h3000, h2100, require order=2
tw = twiss(fodo, cols=["h3000", "h2100"], order=2)

#=
  Spin analysis (invariant spin field as Taylor series,
  amplitude-dependent spin tune).
=#
tw = twiss(fodo, cols=["nx", "ny", "nz"], spin=true,
      as_taylor_series=true, order=2)
print(tw.qspin)
#= Prints:
  AmplitudeDependentValue:
   -0.3094612779434053 - 0.309461277627552 δ - 7.243608429370996e-10 J₁
    + 27.379264609438568 J₂ - 4.343783185854744e-10 δ²
=#

#=
  Track a particle with [x, px, y, py, z, pz]
  = [1e-3 0. 0. 0. 0. 1e-3] for n_turns=10
=#
res = track(fodo, v0=[1e-3 0. 0. 0. 0. 1e-3], n_turns=10)

# Also do spin quaternion tracking:
res = track(fodo, v0=[1e-3 0. 0. 0. 0. 1e-3], n_turns=10, spin=true)
# Apply spin quaternion to horizontal initial spin direction (FAST):
s = track_spin(res.q, [1, 0, 0])

# Track a bunch, give matrix of size n_particles x 6:
n_particles = 10
res = track(fodo, v0=rand(n_particles, 6).*1e-5, n_turns=10, spin=true)
```

## Step-By-Step Description

### Constructing a Beamline

Let's start by constructing a simple FODO cell `Beamline`.

To do this, we will define six `LineElement`s corresponding to each of these objects. The
lengths of each object are specified by `L` (in meters), and the quadrupole strengths can be
set using the property `Kn1`, where `n` means the "normal" multipole (`s` would be "skew")
and `1` means 1st order multipole (quadrupole). The normal sextupole strength is thus `Kn2`.
For the bend, the `angle` parameter sets both the curvature of the coordinate system and the
corresponding magnetic field so that a "reference particle" will follow the coordinate
system curvature exactly. The coordinate system curvature and normalized field strength can
be set independently using `g_ref` and `Kn0` respectively if desired.

```{code-cell} julia
using SciBmad

# Define lattice elements using @elements
@elements begin
  qf = Quadrupole(Kn1=0.36, L=0.5)
  sf = Sextupole(Kn2=1.2, L=0.2)
  d = Drift(L=0.1)
  b = SBend(L=1.2, angle=pi/132)
  qd = Quadrupole(Kn1=-0.36, L=0.5)
  sd = Sextupole(Kn2=-1.2, L=0.2)
end

# Construct a Beamline and define the reference species + momentum
fodo = Beamline([qf, sf, b, d, qd, sd, b, d],
        species_ref=Species("electron"), pc_ref=18e9)
```

Note that, under the hood, all element "kinds" (`Sextupole`, `Quadrupole`, etc.) are one
single type `LineElement`. Therefore, there is nothing preventing you from writing e.g.
`Drift(L=1.2, Ks8=123)`. This structure provides maximal flexibility for defining and
modifying elements (see [Defining a LineElement](element.md) for more).

The [`AtomicAndPhysicalConstants`](https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl)
package is used for specifying particle species, and so any species defined by that package
may be provided. Also, instead of specifying the reference momentum `pc_ref`, we
alternatively could have specified the total reference energy `E_ref`, or the _signed_
magnetic rigidity `p_over_q_ref`.

We can find instances of elements in a `Beamline`, by "indexing" it in the following manner:

```{code-cell} julia
# Find the drifts `d` in `fodo`
ds = fodo[d];

# Get the first element of `fodo`
first_ele = fodo[1];

# Gets all quadrupoles
quadrupoles = fodo[x -> x.kind == "Quadrupole"];
```

Note that `LineElement`s in a beamline simply inherit all properties from the original
element definition. For example, retuning the quadrupole strengths

```{code-cell} julia
# Change the quadrupole strengths
qf.Kn1 = 0.37
qd.Kn1 = -0.37
```

is reflected everywhere that element has been placed in a `Beamline`, without needing to
rebuild `fodo` or search-and-replace through its element list (see
[Defining a Beamline](beamline.md) for more).

### Twiss

Once a `Beamline` is defined, we can compute the Twiss parameters with `twiss`:

```{code-cell} julia
# Compute the periodic Twiss parameters
tw = twiss(fodo)
```

By default this computes the tunes and, in the Sagan-Rubin/Edwards-Teng coupling formalism,
the beta functions, alpha functions, coupling matrix, phase advances/slip, closed orbit and
(crab) dispersions at _all integration steps_ in a
[`DataFrame`](https://dataframes.juliadata.org/stable/) struct.

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

To see all available columns that can provided to `cols`, see the [`twiss`](twiss.md)
docstring.

### Nonlinear Twiss

Beyond the purely linear optics, `twiss` can also compute higher-order and nonlinear
quantities by increasing the order of the underlying truncated power series (DA) map used
internally. All quantities are computed using SciBmad's Lie algebraic normal form analysis
package [`NonlinearNormalForm`](https://github.com/bmad-sim/NonlinearNormalForm.jl). `twiss`
exposes two independent "orders" that can be set:

- `chrom`, the order to which the energy deviation `δ` is truncated, for computing higher
  order chromatic quantities
- `order`, the truncated order of all individual phase space variables, for computing
  amplitude-dependent tunes and resonance driving terms

For example, setting `chrom=2` will compute the chromaticities, and if requested in `cols`
the Montague `w1`/`w2` functions, second-order dispersions `dx_2`/`dy_2`, chromatic beta
beat `dbeta1`/`dbeta2` (equivalent to writing `dbeta1_1`/`dbeta2_1`), etc. In fact, we can
request a chromatic derivative of _any_ scalar-valued quantity, using the notation
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
plain numbers but as `AmplitudeDependentValue`s - Taylor series in the action-angle
variables `J₁`, `J₂` and the energy deviation `δ`:

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
$\mathcal{A}_{cs}$ is the compositional operator representing only a linear (Courant Snyder)
normalizing transformation. Monomials of $h$ are sometimes referred to as **resonance
driving terms** or **detune coefficients** depending on if they drive resonances or tune
shifts with amplitude.

We can extract any Bengtsson monomial by simply setting the `order` of `twiss` appropriately
(must be at least one less than the total order of the monomial). And, if the beam is
coasting, we can take chromatic derivatives as described before.

```{code-cell} julia
# Resonance driving terms h3000, h2100, require order=2
tw = twiss(fodo, cols=["h3000", "h2100"], order=2)
```

### Spin

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
n = [tw.nx[1], tw.ny[1], tw.nz[1]] # ISF at first element
```

With `as_taylor_series=true`, the components of the ISF are returned as full Taylor series in
the phase space variables $(x,p_x,y,p_y,z,p_z)$, rather than just returning $\hat{n}_0$. The
spin tune, accessible as `tw.qspin`, is likewise an `AmplitudeDependentValue`:

```{code-cell} julia
print(tw.qspin)
```

As with the orbital tunes, individual terms, such as the spin tune's linear dependence on
energy, or on its amplitude `J₂`, can be pulled out with `getterm`.

### Track

To do particle tracking, a single function `track` is provided, which accepts initial phase
space coordinates as an `n_particle x 6` matrix:

```{code-cell} julia
#=
  Track a particle with [x, px, y, py, z, pz]
  = [1e-3 0. 0. 0. 0. 1e-3] for n_turns=10
=#
res = track(fodo, v0=[1e-3 0. 0. 0. 0. 1e-3], n_turns=10)

# Also do spin quaternion tracking:
res = track(fodo, v0=[1e-3 0. 0. 0. 0. 1e-3], n_turns=10, spin=true)
# Apply spin quaternion to horizontal initial spin direction (FAST):
s = track_spin(res.q, [1, 0, 0])

# Track a bunch, give matrix of size n_particles x 6:
n_particles = 10
res = track(fodo, v0=rand(n_particles, 6).*1e-5, n_turns=10, spin=true)
```

On the CPU, all tracking will automatically be parallelized using single instruction,
multiple data (SIMD) if your hardware supports it. CPU multithreading can also be enabled by
starting Julia with threads, and setting `use_cpu_multithreading=true`. For more details,
see the [CPU Parallelization](#cpuparallel) section of the manual.

For GPU parallelized tracking, simply initialize your initial particle coordinates as a GPU
array. For example:

```julia
using CUDA
n_particles = 100000 # 100,000 particles
v0 = CUDA.rand(Float64, n_particles, 6) .* 1e-5
res = track(fodo, v0=v0, n_turns=10, spin=true)
```

For more details, see the [GPU Parallelization](#gpuparallel) section of the manual.
