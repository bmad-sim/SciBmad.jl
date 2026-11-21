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

The following is an introduction to SciBmad in one copy-and-pastable block: build a FODO cell, compute linear, nonlinear, chromatic and spin Twiss parameters, adjust the sextupoles to set the chromaticities, and track particles with spin. The rest of this page walks through it step by step. To easily copy a block of code anywhere in the documentation, move your cursor to the top right of the code block and select to copy.

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

# Set number of integration steps in qd to 3
qd.n_steps = 3

# Find the single instance of `qf` in fodo:
qf_in_fodo = fodo[qf];

# Find the drifts `d` in `fodo`
ds = fodo[[d]];

# Get the first element of `fodo`
first_ele = fodo[1];

# Gets all quadrupoles
quads = fodo[x -> x.kind == "Quadrupole"]; # Lambda function
quads = fodo["Quadrupole::*"];             # Classic Bmad syntax
quads = fodo[r"q.*"];                      # Regex match on name

# Compute the periodic Twiss parameters
tw = twiss(fodo)

#= 
  Print the tunes, phase slip factor (etac), momentum compaction (alphac)
=#
@show tw.q1     # Horizontal-like tune
@show tw.q2     # Vertical-like tune
@show tw.etac   # Phase slip factor
@show tw.alphac # Momentum compaction

#=
  Print the horizontal-like beta function at various locations
  including inside elements
=#
@show tw.beta1[qd]        # Beginning of element qd
@show tw.beta1[qd, :exit] # End of qd
@show tw.beta1[qd, :]     # Every integration step inside qd
@show tw.beta1[qd, 2]     # 2nd integration step inside qd
@show tw.beta1[4, :]      # Every integration step of element 4
@show tw.beta1[(0, 1.23)] # Every integration step between s=0, s=1.23

#=
  Compute only specific Twiss parameters
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
@show tw.q2
#= Prints:
  tw.q2 = AmplitudeDependentValue:
    0.05401330001000863 - 0.008253153240321323 δ
=#
# Get the y-chromaticity
chromy = tw.q2[delta=1]
chromy = tw.dq2 # or equivalently tw.dq2_1 

# Resonance driving terms h3000, h2100
tw = twiss(fodo, cols=["h3000", "h2100"])

#=
  Spin analysis (invariant spin field as Taylor series,
  amplitude-dependent spin tune).
=#
tw = twiss(fodo, cols=["nx", "ny", "nz"], spin=true,
      as_taylor_series=true, order=2)
@show tw.qspin
#= Prints:
  tw.qspin = AmplitudeDependentValue:
    -0.3094612779434053 - 0.3094612776559473 δ - 4.0543183384536337e-10 J₁ 
    + 28.127501690102246 J₂ - 2.950933476050364e-10 δ²
=#

#= 
  Adjust the sextupoles to make chromaticities to be +1 
  using SciBmad's autodiff + parametric normal form
=#
te = TwissExpr(fodo)
sol = match!(vary = [@knob(sf.Kn2), @knob(sd.Kn2)],
             targets = [te.dq1 == 1, te.dq2 == 1])

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

To do this, we will define six `LatElement`s corresponding to each of these objects. The
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

All element "kinds" (`Sextupole`, `Quadrupole`, etc.) are one single type `LatElement` 
under the hood. Therefore, there is nothing preventing you from writing e.g.
`Drift(L=1.2, Ks8=123)`. This structure provides maximal flexibility for defining and
modifying elements (see [Defining a LatElement](element.md) for more). A default tracking 
method is chosen given an element kind, e.g. `MatrixKick` for a finite-length `Quadrupole`, 
which you can change at any time (see [Tracking Methods](tracking-methods.md)).

`LatElement`s in a beamline simply inherit all properties from the original
element definition. For example, retuning the quadrupole strengths or number of integration 
steps

```{code-cell} julia
# Change the quadrupole strengths
qf.Kn1 = 0.37
qd.Kn1 = -0.37

# Set number of integration steps in qd to 3
qd.n_steps = 3
```

is reflected everywhere that element has been placed in a `Beamline`, without needing to
rebuild `fodo` or search-and-replace through its element list (see
[Defining a Beamline](beamline.md) for more). 

The [`AtomicAndPhysicalConstants`](https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl)
package is used for specifying particle species, and so any species defined by that package
may be provided. Also, instead of specifying the reference momentum `pc_ref`, we
alternatively could have specified the total reference energy `E_ref`, or the _signed_
magnetic rigidity `p_over_q_ref`.

We can find instances of elements in a `Beamline`, by searching in the following manners:

```{code-cell} julia
# Find the *single* instance of `qf` in fodo:
qf_in_fodo = fodo[qf];

# Find *all* the drifts `d` in `fodo` (note the extra [])
ds = fodo[[d]];

# Get the first element of `fodo`
first_ele = fodo[1];

# Gets all quadrupoles
quads = fodo[x -> x.kind == "Quadrupole"]; # Lambda function
quads = fodo["Quadrupole::*"];             # Classic Bmad syntax
quads = fodo[r"q.*"];                      # Regex match on name
```

To see all the ways you can search for elements, see [Defining a Beamline](beamline.md).

### Twiss

Once a `Beamline` is defined, we can compute the Twiss parameters with `twiss`:

```{code-cell} julia
# Compute the periodic Twiss parameters
tw = twiss(fodo)

# Compute the open Twiss parameters given initial
two = twiss(fodo, initial=(beta1=13, alpha1=-1, beta2=10, alpha2=1))
```

By default this computes the closed orbit, tunes/phase advances and, in the Sagan-Rubin/Edwards-Teng coupling formalism, the beta functions, alpha functions, coupling matrix, phase advances/slip, and (crab) dispersions at _all integration steps_ in a [`DataFrame`](https://dataframes.juliadata.org/stable/) struct.

The values for the whole ring, such as the tunes, are properties of `tw`:
```{code-cell} julia
@show tw.q1     # Horizontal-like tune
@show tw.q2     # Vertical-like tune
@show tw.etac   # Phase slip factor
@show tw.alphac; # Momentum compaction
```

In Julia, the `@` sign specifies a macro, which modifies the code it acts on. In this case, `@show tw.q1` is syntactic sugar that replaces the line of code with `println("tw.q1 = ", tw.q1)`.

The columns of the Twiss table contain the s-dependent values, and can be read at any location, for example the beginning/ends of elements, or every integration step over an s-range.

```{code-cell} julia
# Betas at various locations:
@show tw.beta1[qd]        # Beginning of element qd
@show tw.beta1[qd, :exit] # End of qd
@show tw.beta1[qd, :]     # Every integration step inside qd
@show tw.beta1[qd, 2]     # 2nd integration step inside qd
@show tw.beta1[4, :]      # Every integration step of element 4
@show tw.beta1[(0, 1.23)]; # Every integration step between s=0, s=1.23
```
All of the ways to specify a location are described in [Twiss](twiss.md).

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

Beyond the purely linear optics, `twiss` can also compute higher order quantities, simply by requesting the associated column and/or increasing the order of the underlying truncated power series (DA) map used internally. All quantities are computed using SciBmad's Lie series normal form analysis package [`NonlinearNormalForm`](https://github.com/bmad-sim/NonlinearNormalForm.jl). 

`twiss`exposes two independent "orders" that can be set:

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
@show tw.q2
```

Individual terms of an `AmplitudeDependentValue` can be extracted by indexing the quantity and providing the power of each variable you want. For example, the linear chromaticity in `y` is the coefficient of `δ¹`:

```{code-cell} julia
# Get the y-chromaticity
chromy = tw.q2[delta=1]
```

For chromatic derivatives only, you can also use the shorthand `d<quantity>_<order>` notation:

```{code-cell} julia
chromy = tw.dq2 # or equivalently tw.dq2_1 
```

Likewise, `tw.q2[delta=2]` (or `tw.dq2_2`) is $\partial^2 q_2/\partial\delta^2$. The same mechanism extends naturally to purely amplitude-dependent tune shifts (once `order`
is raised high enough to resolve them), using the `J1` and `J2` keyword arguments to extract tune derivatives w.r.t. the action variables.

We can also extract any so-called resonance driving term/detune coefficient - monomials of the nonlinear effective Hamiltonian in linear action-angle variables. And, if the beam is
coasting, we can even take chromatic derivatives of these quantities as described before. These quantities are defined in [Resonance driving terms and detuning](twiss.md).

```{code-cell} julia
# Resonance driving terms h3000, h2100
tw = twiss(fodo, cols=["h3000", "h2100"])
```

To compute these quantities, technically `order=2` is required. However, `twiss` will automatically resolve the internal orders given the requested columns if not explicitly provided.

### Spin Analysis

`twiss` can additionally analyze the spin dynamics by setting `spin=true`. This enables
computation of the invariant spin field (as a Taylor series in the phase space coordinates if requested) and the amplitude-dependent spin tune:

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
the phase space variables $(x,p_x,y,p_y,z,\delta)$, rather than just returning $\hat{n}_0$. The spin tune, accessible as `tw.qspin`, is likewise an `AmplitudeDependentValue`:

```{code-cell} julia
@show tw.qspin
```

As with the orbital tunes, individual derivatives, such as the spin tune's linear dependence on energy, or on its amplitude `J₂`, can be indexed.

### Optimization using Autodiff
Let's check the chromaticities of our FODO cell:

```{code-cell} julia
tw = twiss(fodo, chrom=2)
@show tw.dq1
@show tw.dq2
```

We want to adjust the sextupoles `sf` and `sd` so that these values are both `+1`. To do this, we first form a `TrissExpr` of our Beamline - this allows us to form mathematical expressions for _any_ output of regular `twiss`, which we can then use in optimizations.

```{code-cell} julia
te = TwissExpr(fodo)
```

Because `TwissExpr` allows expressions for any output from `twiss`, the `cols` keyword argument does not need to, and cannot, be provided.

We can now form the two target expressions using our `TwissExpr`, specifying that we want the chromaticities `dq1` and `dq2` to be `+1`.

```{code-cell} julia
[te.dq1 == 1, te.dq2 == 1]
```

Lastly, we need to specify our "knobs" - the things we want to change to achieve our targets. This is done using `@knob` with the normal sextupole strengths `Kn2` of `sf` and `sd`:

```{code-cell} julia
[@knob(sf.Kn2), @knob(sd.Kn2)]
```

Putting this all together into SciBmad's `match!` function, we can now run the optimization:

```{code-cell} julia
te = TwissExpr(fodo)
sol = match!(vary = [@knob(sf.Kn2), @knob(sd.Kn2)],
             targets = [te.dq1 == 1, te.dq2 == 1])
```

We see that the match has been achieved, and can verify it with a regular `twiss` call:

```{code-cell} julia
twiss(fodo, chrom=2)
```

Under the hood, `match!` uses SciBmad's automatic differentiation and parametric normal form to achieve an exact, and fast, match.

`TwissExpr` enables powerful accelerator lattice design and optimization workflows. Outputs of `TwissExpr` (such as the optics functions) can be passed into other `TwissExpr`s as initial conditions. For example, after matching arc FODO cells, one can pass the output values easily into another `TwissExpr` that is then used to match the arc FODO cells to a straight section. Derivatives of various quantities including w.r.t. knobs can also be matched through the same interface, for example as done in the Linear Optics from Closed Orbits (LOCO) algorithm. For all details, see the [Matching and Optimization](optimize.md) section of the documentation.

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

The particles are saved at the end of the beamline after every turn by default (turn 0 holds the initial coordinates), and are read by turn and particle:

```{code-cell} julia
res.x[turn=10]               # x of all particles after turn 10
res.x[turn=:, particle=1]    # x of particle 1 at turns 0 to 10
```

To save them at other places, give the elements (or positions inside elements) with `at`, and read them with the element first, as for the `twiss` results, e.g. `res.x[qd, turn=10]`. See the [Track](track.md) page for more details.

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

Apple Metal GPUs only support 32-bit floats, so on a Mac use `Float32` coordinates; all of
the tracking will then be done in 32-bit:

```julia
using Metal
v0 = MtlArray(rand(Float32, n_particles, 6) .* 1f-5)
res = track(fodo, v0=v0, n_turns=10, spin=true)
```

For more details, see the [GPU Parallelization](#gpuparallel) section of the manual.

## Where to Go Next

This guide only scratched the surface. The rest of the documentation covers:

- [Matching and Optimization](optimize.md): optimize tunes, beta functions, chromaticities,
  resonance driving terms and more, at any location, by varying any element parameter, with
  exact derivatives; or get the values and derivatives of any Twiss quantity for your own
  optimizer
- [Twiss](twiss.md): all the Twiss columns, reading them at elements and inside elements, Twiss
  of a part of a beamline and of open lines, resonance driving terms and detuning coefficients
- [Parametric Normal Form](parametric-nf.md): Twiss quantities and tunes as Taylor series in
  magnet strengths, lengths or any other parameter
- [Track](track.md): saving particles anywhere, other species, 32-bit coordinates, CPU and GPU
  parallelization; with [Tracking Methods](tracking-methods.md) per element,
  [Collective Effects](collective.md), [time-dependent parameters](timedependent.md) and [batches](batch.md) of lattice parameters
- The [closed orbit](co.md), [dynamic aperture](dynamic-aperture.md) and
  [frequency map analysis](fma.md), all of which run on GPUs
- Lattices: [elements](element.md), [beamlines](beamline.md), [deferred
  expressions](defexpr.md) for interdependent parameters, and [definition tools](def-tools.md)
  such as phase trombones

To get the most use out of SciBmad, we recommend going through the documentation sequentially, starting with the next page [Defining a LatElement](element.md).