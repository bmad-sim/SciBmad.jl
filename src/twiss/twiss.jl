const TENG_EDWARDS_COAST = [   
  "beta1"  ,
  "phi1"   ,
  "dx"     ,
  "x"      ,
  "beta2"  ,
  "phi2"   ,
  "dy"     ,
  "y"      ,
  "slip"   ,
  "alpha1" ,
  "alpha2" ,
  "px"     ,
  "py"     ,
  "z"      ,
  "pz"     ,
  "dpx"    ,
  "dpy"    ,
  "gammac" ,
  "c11"    ,
  "c12"    ,
  "c21"    ,
  "c22"    ,
]

const TENG_EDWARDS = [   
  "beta1"  ,
  "phi1"   ,
  "dx"     ,
  "x"      ,
  "beta2"  ,
  "phi2"   ,
  "dy"     ,
  "y"      ,
  "slip"   ,
  "phi3"   ,
  "alpha1" ,
  "alpha2" ,
  "px"     ,
  "py"     ,
  "z"      ,
  "pz"     ,
  "dpx"    ,
  "dpy"    ,
  "gammac" ,
  "c11"    ,
  "c12"    ,
  "c21"    ,
  "c22"    ,
  "dampdec1",
  "dampdec2",
  "dampdec3",
]

base_cols::Vector{String} = ["index", "name", "kind", "s"]

"""
    twiss(bl::Beamline; kwargs...)

General function to compute the (parametric and nonlinear) optics of a `Beamline`. 
Returns a `Twiss` struct that contains the (amplitude-dependent) (spin) tunes/slip, and a 
dataframe of the lattice functions at each specified integration step. By default, lattice 
functions of the Sagan-Rubin/Edwards-Teng coupling formalism are computed in the dataframe 
at every integration step. The columns are read at locations of the beamline, e.g. 
`tw.beta1[qd]` at the beginning of the element `qd` (see [`TwissColumn`](@ref)), and the 
whole table is `tw.df`.

The `cols` keyword argument can be set to customize what quantities will be computed and 
stored as columns in the returned dataframe. E.g., 
```julia
tw = twiss(bl; cols=["beta1"])
```
will compute only the horizontal-like beta function. We can also specify so-called 
"resonance driving terms"/"detune coefficients" of the nonlinear effective Hamiltonian,
```julia
tw = twiss(bl; cols=["beta1", "h3000"], order=2)
```
where the total order of the `twiss` call has also been set appropriately to compute `h3000` 
(without `order` and `chrom`, the orders that the columns need are chosen).

An arbitrary order chromatic derivative of any scalar-valued column can be taken using the 
syntax `d<column>_<order>`. If `order==1`, then the `_1` can be omitted. E.g., the linear and 
2nd order dispersions, as well as ∂β₁/∂δ can be computed with
```julia
tw = twiss(bl; cols=["dx", "dx_2","dbeta1"], chrom=2)
```
where the `chrom` keyword argument, which specifies solely the truncation order of δ, has been 
set appropriately. 

Chromatic derivatives need a beam without synchrotron motion: `rf_on=false` if there are RF
cavities.

To customize the output, such as computing higher order (chromatic) lattice functions, spin 
lattice functions/tune, amplitude-dependent tunes, and parametric quantities, see the 
keyword arguments:

## Keyword arguments
- `at::Union{Colon, AbstractVector}`: Where to compute the lattice functions (the rows of the
    table): a vector of locations, each as one index of a column (see [`TwissColumn`](@ref)):
    elements (at the beginning of each placement), beamline indexes, names (as for
    `findelements`), and s-ranges (tuples). Default `:`, every integration step and the end of
    the beamline. An entry that is not in the beamline is an error; in a part of the beamline
    (`start`, `stop`) only the locations in the part are computed.
- `start`, `stop`: Elements (or their beamline indexes) to compute the optics of only
    a part of the beamline: from the beginning of `start` (default: the first element) to the
    beginning of `stop` (default: the end of the beamline). Without `initial` or `a_initial`
    these are the periodic optics of that part by itself (e.g. of one cell of a ring),
    with them those of an open line that starts at `start`. `s` and `index` in the table are
    those of the whole beamline, phases are counted from `start`, and the last row is the
    beginning of `stop`.
- `in_body_coordinates::Bool`: If `true` the lattice functions inside misaligned elements are not
    transformed back to the reference curve coordinates. Default is `false`
- `rf_on::Bool`: If `false`, then any `RFParams` in any `LatElement`s are ignored for the 
    entirety of the Twiss tracking and calculation. Default is `true`
- `order::Integer`: TPSA truncation order for all phase space coordinates used in the Twiss 
    calculation. If neither `order` nor `chrom` is given, both are chosen from `cols` (as 
    [`TwissExpr`](@ref) does): 1 for the default columns, higher for chromatic derivatives 
    (`chrom`) and resonance driving terms (`order`). Default is 1 if only `chrom` is given
- `chrom::Integer`: TPSA truncation order for only δ, default is `order` if `order` is given
- `GTPSA_descriptor::Union{Descriptor,Nothing}`: Custom GTPSA `Descriptor` to use for the Twiss 
    calculation, required to provide to do parametric normal form and lattice function 
    calculations. Default is `Descriptor([order, order, order, order, order, chrom], max(order,chrom))`.
- `initial`: The lattice functions at the start of an "open" line (e.g. a transfer line), by name:
    `beta1`, `beta2` (required), `alpha1`, `alpha2`, the coupling `c11`, `c12`, `c21`, `c22`, the
    dispersion `dx`, `dpx`, `dy`, `dpy`, and the orbit `x`, `px`, `y`, `py`, `z`, `pz` (all 0 if
    not given; `pz` is δ, else `delta0`), with the meanings of the columns of the same names.
    A `NamedTuple`, a `Dict` (with `Symbol` or `String` keys), a `Context` (whose variables can
    then be knobs of a match), or a row of another table (`tw.df[end, :]`; its other columns are
    ignored). The values may be Taylor series in GTPSA parameters, not in the phase space
    variables or δ. The beam has no synchrotron motion: RF cavities with voltage need
    `rf_on=false` (or a 6D `a_initial`).
    Default is `nothing`, to compute the periodic ("closed") lattice functions
- `a_initial::Union{DAMap,Nothing}`: A `DAMap` of the initial transformation from normal form 
    coordinates to laboratory coordinates, to compute the lattice functions of an "open" line
    (instead of `initial`). A coasting `a_initial` (without synchrotron motion) also needs
    `rf_on=false` if the line has RF cavities with voltage. Default is `nothing`
- `damping::Union{Bool,Nothing}`: Specifies if radiation damping is included. Default is `nothing`,
    which will auto-detect from `a_initial` or the periodic `a`.
- `delta0`: If the beam is coasting (no longitudinal oscillations), then the optics will 
    be computed around this delta-dependent orbit. Else, this will be an initial guess for the 
    6D closed orbit calculation. Default is `0.`. If `a_initial` is provided then this will be ignored, 
    UNLESS `a_initial` is coasting, in which case this will set the value for δ. With `initial`, it
    is δ if `initial` has no `pz`.
- `v0`: Initial guess for the closed orbit finder if `a_initial` is not provided, else setting this 
    will override the orbit expansion origin coordinates in `a_initial.v0`. Not with `initial`,
    which gives the orbit.
- `spin::Bool`: If `true`, spin is included in the Twiss calculation. Default is `false`, or `true` if 
    `a_initial` includes spin.
- `base_cols`: A vector of "base columns", that will always be included in the Twiss dataframe regardless 
    of what is specified in `cols`. Defaults to the global variable `SciBmad.base_cols`, which defaults 
    to `["index", "name", "kind", "s"]`.
- `cols`: A vector of columns to include in the Twiss dataframe. See the extended help for all possible 
    columns.
- `as_taylor_series::Union{Bool,Nothing}`: If `true`, the lattice functions in the dataframe are 
    returned as Taylor series types - this will be necessary if doing parametric normal form. Default 
    is `nothing`, which autoselects to `false` if there are no parameters in the GTPSA, and `true` if 
    there are parameters.
- `coast_tol`: Tolerance for the first-order delta dependences on each of the phase space variables below 
    which coasting beam is assumed, passed to `find_closed_orbit`. Default is `1e-14`.
- `symplectic_tol::Float64`: Tolerance of symplectic condition violation, above which radiation damping 
    is assumed. Default is `1e-8`.
- `batch_index::Int`: If any element parameters are `BatchParam`, sets which index in the batch the `twiss`
    should be evaluated at. Default is `1`. Batched element lengths and a batched reference
    energy are not supported in `twiss` yet.

To see a description of all quantities available to include in `cols`, see the extended help 
section using `??twiss`

# Extended help

Columns to include in `cols` are detailed below. As a reminder, an arbitrary order chromatic derivative 
of *any* scalar-valued output can be computed using the syntax "d<column>_<order>" (e.g. "dx_1" is the 
linear dispersion).

- `index`  : Beamline index of the containing `LatElement`
- `name`   : Name of the containing `LatElement`
- `kind`   : Kind of the containing `LatElement`
- `s`      : s-position of the row (may be inside of `LatElement`)
- `beta1`  : Horizontal-like beta function of the Sagan-Rubin/Edwards-Teng coupling formalism
- `beta2`  : Vertical-like beta function of the Sagan-Rubin/Edwards-Teng coupling formalism
- `alpha1` : Horizontal-like alpha function of the Sagan-Rubin/Edwards-Teng coupling formalism
- `alpha2` : Vertical-like alpha function of the Sagan-Rubin/Edwards-Teng coupling formalism
- `phi1`   : Horizontal-like phase advance in units of [2π]
- `phi2`   : Vertical-like phase advance in units of [2π]
- `phi3`   : Longitudinal-like phase advance in units of [2π]
- `dampdec1`: Horizontal-like damping decrement
- `dampdec2`: Vertical-like damping decrement
- `dampdec3`: Longitudinal-like damping decrement
- `slip`   : Derivative of z/β₀ with δ from the start of the table: -c times the change of the
    arrival time per unit δ [m]
- `x`      : Orbit canonical x coordinate
- `px`     : Orbit canonical px coordinate
- `y`      : Orbit canonical y coordinate
- `py`     : Orbit canonical py coordinate
- `z`      : Orbit canonical z coordinate
- `pz`     : Orbit canonical pz coordinate
- `zx`     : Adiabatic dependence of x on initial longitudinal position/time (AKA "crab dispersion") 
- `zpx`    : Adiabatic dependence of px on initial longitudinal position/time (AKA "crab dispersion")
- `zy`     : Adiabatic dependence of y on initial longitudinal position/time (AKA "crab dispersion")
- `zpy`    : Adiabatic dependence of py on initial longitudinal position/time (AKA "crab dispersion")
- `nx`     : Invariant spin field x-component (recommended to use with `as_taylor_series=true`)
- `ny`     : Invariant spin field y-component (recommended to use with `as_taylor_series=true`)
- `nz`     : Invariant spin field z-component (recommended to use with `as_taylor_series=true`)
- `n0x`    : Closed orbit periodic spin direction x-component
- `n0y`    : Closed orbit periodic spin direction y-component
- `n0z`    : Closed orbit periodic spin direction z-component
- `N`      : N matrix of the Sagan-Rubin coupling formalism
- `Vi`     : V⁻¹ matrix of the Sagan-Rubin coupling formalism
- `c11`    : [1,1] component of the Sagan-Rubin coupling matrix C
- `c12`    : [1,2] component of the Sagan-Rubin coupling matrix C
- `c21`    : [2,1] component of the Sagan-Rubin coupling matrix C
- `c22`    : [2,2] component of the Sagan-Rubin coupling matrix C
- `gammac` : Coupling factor of the Sagan-Rubin coupling formalism
- `r<ij>`  : Component [i,j] (i, j from 1 to 6) of the linear transfer matrix from the start of
    the table (the beginning of the beamline, `start`, or the start of an open line) to the
    row: ∂zᵢ/∂zⱼ, with z = (x, px, y, py, z, pz) at the row and zⱼ at the start, around the orbit
- `m<ij>`  : Component [i,j] of the periodic matrix at the row (periodic tables only: from the row
    once through the beamline, or the part from `start` to `stop`, back to the row)
- `w1a`    : Horizontal-like Montague function "a" component, requires `chrom > 1` and/or `order > 1`
- `w2a`    : Vertical-like Montague function "a" component, requires `chrom > 1` and/or `order > 1`
- `w1b`    : Horizontal-like Montague function "b" component, requires `chrom > 1` and/or `order > 1`
- `w2b`    : Vertical-like Montague function "b" component, requires `chrom > 1` and/or `order > 1`
- `w1`     : Horizontal-like Montague function, requires `chrom > 1` and/or `order > 1`
- `w2`     : Vertical-like Montague function, requires `chrom > 1` and/or `order > 1`
- `H1`     : De Moivre-Ripken H¹ matrix (see E. Forest, _From Tracking Code to Analysis_)
- `H2`     : De Moivre-Ripken H² matrix (see E. Forest, _From Tracking Code to Analysis_)
- `H3`     : De Moivre-Ripken H³ matrix (see E. Forest, _From Tracking Code to Analysis_)
- `B1`     : De Moivre-Ripken B¹ matrix (see E. Forest, _From Tracking Code to Analysis_)
- `B2`     : De Moivre-Ripken B² matrix (see E. Forest, _From Tracking Code to Analysis_)
- `B3`     : De Moivre-Ripken B³ matrix (see E. Forest, _From Tracking Code to Analysis_)
- `E1`     : De Moivre-Ripken E¹ matrix (see E. Forest, _From Tracking Code to Analysis_)
- `E2`     : De Moivre-Ripken E² matrix (see E. Forest, _From Tracking Code to Analysis_)
- `E3`     : De Moivre-Ripken E³ matrix (see E. Forest, _From Tracking Code to Analysis_)
- `a`      : (Nonlinear) map to transform from Floquet variables to laboratory variables
- `a0`     : (Nonlinear) map to transform to the parameter-dependent fixed point
- `a1`     : Linear normalizing map around the parameter-dependent fixed point, including nonlinear parameter dependence
- `a2`     : Nonlinear part of the normalizing map, around the parameter-dependent fixed point
- `as`     : (Nonlinear) spin normalizing map
- `h<ijkl>`   : The "ijkl" resonance driving term/detune coefficient/nonlinear effective Hamiltonian monomial for the coasting beam case
- `h<ijklmn>` : The "ijklmn" resonance driving term/detune coefficient/nonlinear effective Hamiltonian monomial including longitudinal oscillations
"""
function twiss(
  bl::Beamline; 

  at::Union{Colon, AbstractVector, _Locations} = :,
  start::Union{Nothing,Integer,LatElement} = nothing,
  stop::Union{Nothing,Integer,LatElement} = nothing,
  in_body_coordinates::Bool = false, 
  rf_on::Bool = true,

  # GTPSA truncation order sets:
  order::Union{Nothing,Integer} = nothing,
  chrom::Union{Nothing,Integer} = nothing,
  GTPSA_descriptor::Union{Descriptor,Nothing} = nothing,

  initial                           = nothing,
  a_initial::Union{Nothing,DAMap}   = nothing, 
  damping::Union{Nothing,Bool} = nothing, # nothing = auto-detect from a_initial

  # Initial input, CO guess if periodic, initial orbit if not periodic
  delta0::Union{Nothing,Number} = nothing, # (0 unless initial has pz)
  v0::Union{Nothing,Matrix} = nothing, # (from delta0 or a_initial, below)

  # The lattice functions to compute
  spin::Bool      = isnothing(a_initial) ? false : !isnothing(a_initial.q),
  base_cols       = SciBmad.base_cols,
  cols            = nothing, # (de_moivre ? DE_MOIVRE : TENG_EDWARDS)..., (spin ? SPIN : Function[])...],\
  as_taylor_series::Union{Nothing,Bool} = nothing, # nothing = auto-select (if nn > 6, true, else false)

  coast_tol = 1e-14, 
  symplectic_tol = 1e-8, # Tolerance below which to include damping
  batch_index::Int=1,
  )

  if !isnothing(start) || !isnothing(stop)
    # A part of the beamline: the twiss of a Beamline of these elements (inheriting from those
    # of bl, with the same reference and context) at the locations of `at` in the part, with the
    # s and the indexes of the whole one
    n_ele = length(bl.line)
    i_start = isnothing(start) ? 1 : _element_index(bl, start, "start")
    i_stop = isnothing(stop) ? n_ele + 1 : _element_index(bl, stop, "stop")
    1 <= i_start < i_stop <= n_ele + 1 ||
      error("`start` must be before `stop` in the beamline (indexes $i_start and $i_stop)")
    part = Beamline(bl.line[i_start:i_stop-1]; species_ref=bl.species_ref,
                    p_over_q_ref=bl.p_over_q_ref, context=bl.context)
    ls = _line_steps(bl)
    locs = _at_locations(bl, (i_start, i_stop), at, ls)
    at_part = _Locations([i == -1 || i == i_stop ? (-1, 0) : (i - i_start + 1, k)
                          for (i, k) in locs])
    tw = twiss(part; at=at_part, in_body_coordinates, rf_on, order, chrom, GTPSA_descriptor,
               initial, a_initial, damping, delta0, v0, spin, base_cols, cols, as_taylor_series,
               coast_tol, symplectic_tol, batch_index)
    return _twiss_part_table(tw, bl, i_start, i_stop, locs, ls)
  end

  if !isnothing(initial)
    # The start of an open line by its Twiss parameters: a_initial is made from them below (with
    # the Descriptor), the orbit is v0
    isnothing(a_initial) || error("Give either `initial` or `a_initial`, not both")
    isnothing(v0) || error("`initial` gives the initial orbit (`x`, `px`, ...): `v0` is not used")
    spin && error("`initial` has no spin: for spin use `a_initial`")
    v0, desc_initial = _initial_v0(initial, delta0)
    if isnothing(GTPSA_descriptor)
      GTPSA_descriptor = desc_initial
    elseif !isnothing(desc_initial) && desc_initial != GTPSA_descriptor
      error("Specified `GTPSA_descriptor` disagrees with that of the values of `initial`")
    end
    v0_and_coast = (v0, true)
  else
    delta0 = something(delta0, 0.0)
    if isnothing(v0)
      v0 = [0. 0. 0. 0. 0. delta0]
      if !isnothing(a_initial)
        v0[1:length(a_initial.v0)] .= a_initial.v0
      end
    end
    if isnothing(a_initial)
      v0_and_coast = co_and_coast(bl, v0, rf_on, coast_tol, batch_index)
    else
      v0_and_coast = (v0, isodd(NNF.nvars(a_initial)))
    end
  end
  # An open line started without synchrotron motion (initial, a coasting a_initial): its maps
  # can't hold the energy change of a cavity, which would be lost silently
  if (!isnothing(initial) || !isnothing(a_initial)) && v0_and_coast[2] && rf_on &&
     any(ele -> !isnothing(ele.RFParams) && !(ele.voltage isa Number && iszero(ele.voltage)),
         bl.line)
    error("`initial` and a coasting `a_initial` describe a beam without synchrotron motion, " *
          "and this line has RF cavities with voltage: use `rf_on=false`, or a 6D `a_initial`")
  end

  coast = v0_and_coast[2]
  # The δ of the orbit of a coasting beam (delta0). In the maps δ is a parameter without a value:
  # it is added to the tracked coordinates and taken off the tracked maps again
  delta_orbit = coast ? Float64(v0_and_coast[1][6]) : 0.0

  if isnothing(cols)
    if coast
      cols = TENG_EDWARDS_COAST
    else
      cols = TENG_EDWARDS
    end
  end

  if spin
    spinregex = r"^(?:nx|ny|nz|n0x|n0y|n0z|d(?:nx|ny|nz|n0x|n0y|n0z)(?:_[1-9])?)$"
    if !any(x->occursin(spinregex, x), cols)
      # Then add default cols:
      cols = vcat(cols, ["n0x", "n0y", "n0z", "dnx", "dny", "dnz"])
    end
  end

  # Only the orbit asked for (for a coasting beam also its derivatives in δ, as the dispersion):
  # the rows only need the translation to the fixed point (see _twiss_push_a)
  orbit_only = !spin && all(c -> _orbit_col(c, coast), Iterators.flatten((base_cols, cols)))
  # The transfer matrix from the start and the one-turn matrix (r and m columns) need the map
  # from the start to every row
  transfer = any(c -> occursin(r"^d?[rm][1-6][1-6](?:_[1-9])?$", c),
                 Iterators.flatten((base_cols, cols)))

  # Without `order` and `chrom`, the truncation orders are those the columns need (as TwissExpr
  # chooses them); with synchrotron motion both are the same
  explicit_orders = !isnothing(order) || !isnothing(chrom)
  if explicit_orders
    order = something(order, 1)
    chrom = something(chrom, order)
  else
    order, chrom = _twiss_auto_orders(cols)
    if !coast
      order = chrom = max(order, chrom)
    end
  end

  if !isnothing(a_initial)
    if !isnothing(GTPSA_descriptor) && GTPSA.getdesc(first(a_initial.v)) != GTPSA_descriptor
      error("Specified `GTPSA_descriptor` disagrees with that of `a_initial`")
    end
    if spin && isnothing(a_initial.q)
      error("Unable to propagate spin: `a_initial` does not include spin")
    elseif !spin && !isnothing(a_initial)
      a_initial = DAMap(v0=a_initial.v0, v=a_initial.v, nv=NNF.nvars(a_initial), np=NNF.nparams(a_initial), s=a_initial.s)
    end
    GTPSA_descriptor = GTPSA.getdesc(first(a_initial.v))
  end

  if isnothing(GTPSA_descriptor)
    storedesc = GTPSA.desc_current
    GTPSA_descriptor = Descriptor([order, order, order, order, order, chrom], max(order,chrom))
    GTPSA.desc_current = storedesc # Don't reset the global
  elseif explicit_orders
    @info "`GTPSA_descriptor` has been explicitly provided: ignoring `order`/`chrom` inputs"
  end

  if !coast && chrom != order
    error("""
    You specified `chrom`, but this beamline has synchrotron motion. Please turn off RF 
    cavities to get delta-dependent lattice functions.
    """)
  end

  init = TI.InitGTPSA{GTPSA.Dynamic,Descriptor}(; dynamic_descriptor=GTPSA_descriptor)
  if !isnothing(initial)
    a_initial = _twiss_initial_map(initial, init)
  end

  # Check if output are TPSA in parameters (delta excluded)
  if isnothing(as_taylor_series)
    if TI.ndiffs(init) > 6
      as_taylor_series = true
    else
      as_taylor_series = false
    end
  end

  # The rows of the table are the locations of `at`. The beginning and the end of the beamline
  # are always tracked: the first and the last location
  ls = _line_steps(bl)
  rows = _at_locations(bl, (1, length(bl.line) + 1), at, ls)
  include_start = !isempty(rows) && first(rows) == (1, 0)
  include_end = !isempty(rows) && last(rows) == (-1, 0)
  locs = [include_start ? Tuple{Int,Int}[] : [(1, 0)]; rows;
          include_end ? Tuple{Int,Int}[] : [(-1, 0)]]
  s, idxs, names, kinds, step_save, ele_steps = _location_info(bl, locs, ls)
  beta_gamma_ref = Vector{Float64}(undef, length(s)) # Store the reference energy at each step
  t_ref = Vector{eltype(s)}(undef, length(s)) # Store the reference time at each step (type of the lengths, e.g. TPS for parametric lengths)

  # The maps are tracked with track, which saves them at the rows (the steps step_save) with the
  # reference there. If the GTPSA truncation order is uniform, the identity is tracked once: the
  # maps from the start to the rows, the last one the one-turn map, give a and its value at every
  # row (a composed with them). Else a is computed from the one-turn map, and the tracking of its
  # fixed point a0 (below) gives a at the rows too. With a_initial, a_initial is tracked.
  locations = (s, idxs, names, kinds, ele_steps)
  maps = nothing
  r_and_tunes = nothing
  tr = nothing
  # The phases of the rows are known modulo 1 (see _store_twiss!). If the rows are not all
  # integration steps they can be more than half a turn apart, and whole turns come from the
  # linear maps to every step (_twiss_turns). A linear twiss without parameters saves its own
  # tracking at every step for that (tr_steps); else the identity is tracked once more at order 1
  n_steps_total = last(step_save)
  sparse = length(step_save) < n_steps_total + 1
  no_locations = (similar(s, 0), Int[], String[], String[], Int[])
  tr_steps = nothing
  a0_rows = nothing
  if isnothing(a_initial)
    eye = _twiss_make_identity(v0_and_coast[1], init, Val{coast}(), Val{spin}())
    if _check_cachable(GTPSA_descriptor)
      if sparse && !spin && TI.ndiffs(init) == 6 && unsafe_load(GTPSA_descriptor.desc).mo == 1
        tr = _twiss_track(eye, bl, rf_on, batch_index, zero(s[1]), no_locations,
                          collect(0:n_steps_total), in_body_coordinates, delta_orbit)
        tr_steps = tr
        maps = _twiss_maps(eye, tr, step_save .+ 1)
      else
        tr = _twiss_track(eye, bl, rf_on, batch_index, zero(s[1]), locations, step_save,
                          in_body_coordinates, delta_orbit)
        maps = _twiss_maps(eye, tr, axes(tr.data.v, 3))
      end
      m_turn = maps[end]
    else
      tr0 = _twiss_track(eye, bl, rf_on, batch_index, zero(s[1]), no_locations, Int[],
                         in_body_coordinates, delta_orbit)
      m_turn = _twiss_map(eye, view(tr0.bunch.v, 1, :), _view_q(tr0.bunch.q))
    end
    a, r, tunes = _a_r_tunes(m_turn)
    if isnothing(maps)
      # With phase space variables below the maximum order, the one-turn map around the closed
      # orbit lacks terms (e.g. x^2) that the fixed point's dependence on δ and on parameters
      # turns into terms of lower order in x, so a is only right to first order in them. Tracking
      # the fixed point itself (a0, a translation) loses nothing: the one-turn map around it
      # gives a again, with the error of the fixed point squared each time (a Newton iteration).
      # The last tracking saves a0 at the rows: composed with inv(a0) ∘ a, which has no constant
      # part (so nothing is lost either), it gives a there
      mo = unsafe_load(GTPSA_descriptor.desc).mo
      n_refine = ceil(Int, log2(mo))
      for k in 1:n_refine
        a0 = factorise(a; compute_r=Val{false}()).a0
        if k < n_refine
          tr0 = _twiss_track(a0, bl, rf_on, batch_index, zero(s[1]), no_locations, Int[],
                             in_body_coordinates, delta_orbit)
        else
          tr = _twiss_track(a0, bl, rf_on, batch_index, zero(s[1]), locations, step_save,
                            in_body_coordinates, delta_orbit)
          maps = _twiss_maps(a0, tr, axes(tr.data.v, 3))
          a0_rows = a0
          tr0 = tr
        end
        m0 = inv(a0) ∘ _twiss_map(a0, view(tr0.bunch.v, 1, :), _view_q(tr0.bunch.q))
        b, r, tunes = _a_r_tunes(m0)
        a = a0 ∘ b
      end
    end
    a_initial, r_and_tunes = a, (r, tunes)
  end

  if isnothing(damping)
    damping = norm(NNF.checksymp(NNF.jacobian(a_initial))) > symplectic_tol
  end

  # Determine canonization level
  canonise, phase, damp = canonise_phase_damp(GTPSA_descriptor, coast, damping)

  a_initial = factorise(a_initial; canonise=canonise, damping=isnothing(damp) ? false : true,
                        compute_r=Val{false}()).a

  turns = nothing
  if sparse || orbit_only
    if isnothing(tr_steps) && !sparse && !isnothing(maps) && isnothing(a0_rows)
      tr_steps = tr # (the identity, tracked to every step)
    elseif isnothing(tr_steps)
      storedesc = GTPSA.desc_current
      init1 = TI.InitGTPSA{GTPSA.Dynamic,Descriptor}(; dynamic_descriptor=Descriptor(6, 1))
      GTPSA.desc_current = storedesc
      eye1 = _twiss_make_identity(v0_and_coast[1], init1, Val{coast}(), Val{false}())
      # (with the numbers of TPS parameters of the lattice, and a time of that type)
      tr_steps = _twiss_track(eye1, bl, rf_on, batch_index, 0.0, no_locations,
                              collect(0:n_steps_total), in_body_coordinates, delta_orbit;
                              scalar_params=true)
    end
    turns = _twiss_turns(tr_steps, a_initial, step_save)
  end

  # a at every row: the maps to the rows (from the start, or the tracked a0) composed with a (or
  # inv(a0) ∘ a), or a itself tracked
  if isnothing(maps)
    tr = _twiss_track(a_initial, bl, rf_on, batch_index, zero(s[1]), locations, step_save,
                      in_body_coordinates, delta_orbit)
    fac, a0s, transfers, phi1, phi2, phi3_or_slip, damp1, damp2, damp3 =
      _twiss_push_a(tr, a_initial, canonise, phase, damp, coast, turns, orbit_only, transfer)
  else
    a_right = isnothing(a0_rows) ? a_initial : inv(a0_rows) ∘ a_initial
    fac, a0s, transfers, phi1, phi2, phi3_or_slip, damp1, damp2, damp3 =
      _twiss_push_a(maps, a_right, canonise, phase, damp, coast, turns, orbit_only, transfer)
  end
  _twiss_reference!(beta_gamma_ref, t_ref, tr,
                    tr === tr_steps ? step_save .+ 1 : axes(tr.data.v, 3))

  twi = TwissInternal(s, names, kinds, idxs, beta_gamma_ref, t_ref, fac, a0s, transfers, phi1,
                      phi2, phi3_or_slip, damp1, damp2, damp3, r_and_tunes, delta_orbit)

  # Finally, construct the summ and the dataframe (with provided columns)
  # And post-process with the provided columns
  # Need to do one row first then can construct the DataFrame
  df, cache = _twiss_df(vcat(base_cols, cols), twi, include_start, include_end, Val{as_taylor_series}())

  # q1, q2, slip factor eta_c, momentum compaction alpha_c, [q3, qspin]
  # only thing is with 3d motion, need to compute 
  summ = _twiss_summ(twi, cache)
  # The row of each location (beamline index, step in the element) in the table
  return Twiss(summ, df, bl, 1, nothing,
               Dict{Tuple{Int,Int},Int}(l => r for (r, l) in enumerate(rows)))
end

# The table of the part with the indexes and s of the whole beamline bl (the last row, the end of
# the part, is the beginning of the element i_stop), its rows at the locations locs of bl (ls:
# _line_steps of bl)
function _twiss_part_table(tw, bl, i_start, i_stop, locs, ls)
  df = tw.df
  ends = hasproperty(df, :index) ? findall(==(-1), df.index) : Int[]
  if hasproperty(df, :index)
    df.index .+= i_start - 1
  end
  if i_stop <= length(bl.line)
    up = bl.line[i_stop].UniversalParams::UniversalParams
    for r in ends
      df.index[r] = i_stop
      hasproperty(df, :name) && (df.name[r] = up.name)
      hasproperty(df, :kind) && (df.kind[r] = up.kind)
    end
  elseif hasproperty(df, :index)
    df.index[ends] .= -1
  end
  if hasproperty(df, :s)
    s = first(_location_info(bl, locs, ls))
    df.s .= eltype(df.s) <: TPS ? s : scalar.(s)
  end
  stop = i_stop <= length(bl.line) ? i_stop : nothing
  rows = Dict{Tuple{Int,Int},Int}(l => r for (r, l) in enumerate(locs))
  return Twiss(tw.summ, df, bl, i_start, stop, rows)
end

function co_and_coast(bl, v0, rf_on, coast_tol, batch_index)
  co_sol = find_closed_orbit(bl; v0=copy(v0), batch=Val{false}(), rf_on, batch_start=batch_index, coast_tol) # find_closed_orbit mutates v0
  if co_sol.sol.retcode != RETCODE_SUCCESS
    error("Closed orbit finder did not converge.")
  end
  return (co_sol.v0, co_sol.coasting_beam)
end

# The truncation orders (order, chrom) that the columns need, as TwissExpr chooses them
# (_col_order): driving terms (and their δ-derivatives) need all variables to that order, the other
# columns (chromatic derivatives, Montague functions) only δ
# A column that only needs the orbit (see orbit_only in twiss). For a coasting beam z and the map
# a0 are not: factorise canonises a first, which changes the dependence of their time-like row on δ
_orbit_col(col, coast) = col in ("index", "name", "kind", "s", "x", "px", "y", "py", "pz") ||
                         (coast ? occursin(r"^d(?:x|px|y|py)(?:_[1-9])?$", col) :
                                  col in ("z", "a0"))

function _twiss_auto_orders(cols)
  order = 1
  chrom = 1
  for col in cols
    n = try
      _col_order(col)
    catch # not a column of TwissExpr (e.g. a matrix): order 1, or twiss reports it
      1
    end
    if occursin(r"^d?h[0-9]{4,6}(?:_[1-9])?$", col)
      order = max(order, n)
    else
      chrom = max(chrom, n)
    end
  end
  return order, max(order, chrom)
end

function _check_cachable(GTPSA_descriptor)
  # check if we can cache_and_concat: the maps from the start composed with a lose no term if
  # all phase space variables have the maximum order (a has no constant part, so a term of total
  # order n of the composition only needs terms of the maps up to order n). Parameters may have
  # lower orders
  desc = unsafe_load(GTPSA_descriptor.desc)
  mo = desc.mo
  if all(x->x == mo, unsafe_wrap(Vector{UInt8}, desc.no, desc.nv))
    return true
  else
    return false
  end
end

function _twiss_make_identity(v0, init, ::Val{coast}, ::Val{spin}) where {coast,spin}
  nn = TI.ndiffs(init)
  nv = 6
  np = nn-nv
  if coast
    nv -= 1
    np += 1
  end
  return DAMap(init=init, nv=nv, np=np, v0=view(v0, :, 1:nv), v_matrix=I, q=(spin ? I : nothing))
end

# The Twiss parameters that `initial` (see twiss) can give
const _INITIAL_KEYS = (:beta1, :alpha1, :beta2, :alpha2, :c11, :c12, :c21, :c22, :dx, :dpx, :dy,
                       :dpy, :x, :px, :y, :py, :z, :pz)

# A value of initial (a NamedTuple, a Dict with Symbol or String keys, e.g. from Python, a Context,
# a table row), or default
function _initial_get(initial, name::Symbol, default)
  if initial isa AbstractDict
    haskey(initial, name) && return initial[name]
    haskey(initial, String(name)) && return initial[String(name)]
    return default
  end
  return name in propertynames(initial) ? getproperty(initial, name) : default
end

# The initial orbit of `initial` (numbers: the values of Taylor series, whose parameter dependence
# goes into the map), and the Descriptor of its Taylor series (nothing if none)
function _initial_v0(initial, delta0)
  names = initial isa AbstractDict ? Symbol.(keys(initial)) : propertynames(initial)
  for k in names # (other columns of a table row are ignored, e.g. name and phi1; typos are not)
    k in _INITIAL_KEYS || haskey(_TWISS_FCN_MAP, String(k)) ||
      (try _col_order(String(k)); true catch; false end) || # (e.g. dbeta1, dx_2, h3000)
      error("`initial` has no parameter `$k`")
  end
  for k in (:beta1, :beta2)
    isnothing(_initial_get(initial, k, nothing)) && error("`initial` needs `beta1` and `beta2`")
  end
  pz = _initial_get(initial, :pz, nothing)
  !isnothing(pz) && !isnothing(delta0) && error("Give δ either as `pz` of `initial` or `delta0`")
  orbit = (_initial_get(initial, :x, 0.0), _initial_get(initial, :px, 0.0),
           _initial_get(initial, :y, 0.0), _initial_get(initial, :py, 0.0),
           _initial_get(initial, :z, 0.0), something(pz, delta0, 0.0))
  v0 = Float64[scalar(o) for _ in 1:1, o in orbit]
  desc = nothing
  for k in _INITIAL_KEYS
    v = _initial_get(initial, k, 0.0)
    v isa TPS || continue
    isnothing(desc) && (desc = GTPSA.getdesc(v))
    GTPSA.getdesc(v) == desc || error("The values of `initial` have different GTPSA Descriptors")
  end
  return v0, desc
end

# The map a from normal form coordinates to the coordinates at the start of an open line with the
# Twiss parameters of `initial`, without synchrotron motion (δ a parameter). As the a of a periodic
# twiss, it starts at the origin of the normal form coordinates and the orbit is its constant part.
# a = a0 ∘ a1: a0 the translation to the orbit, linear in δ by the dispersion (with the time-like
# correction of factorise), a1 linear in x, px, y, py: A = V N, with N the normalizing
# Courant-Snyder matrices (N11 = sqrt(beta), N21 = -alpha/sqrt(beta), N12 = 0, N22 = 1/N11) and V
# the inverse of the Vi of the Sagan-Rubin coupling matrix C (see _Vi), [g I C; -C⁺ g I] with
# g = sqrt(1 - det C), as twiss computes these columns from a
function _twiss_initial_map(initial, init)
  id = _twiss_make_identity(zeros(1, 6), init, Val{true}(), Val{false}())
  xi = id.v
  vals = Dict(k => _initial_get(initial, k, 0.0) for k in _INITIAL_KEYS)
  vals[:pz] = 0.0 # (δ: v0)
  for (k, v) in vals
    if v isa TPS
      GTPSA.getdesc(v) == GTPSA.getdesc(xi[1]) ||
        error("`initial`: `$k` is a Taylor series of another Descriptor than the twiss")
      any(i -> !iszero(TI.deriv(v, i)), 1:6) && error("`initial`: `$k` may depend on GTPSA " *
        "parameters, not on the phase space variables or δ (give the dispersion as `dx`, ...)")
      vals[k] = zero(xi[1]) # (the type of the twiss, e.g. from a TPS with a static Descriptor)
      TI.copy_tps!(vals[k], v)
    end
  end
  c11, c12, c21, c22 = vals[:c11], vals[:c12], vals[:c21], vals[:c22]
  detc = c11*c22 - c12*c21
  scalar(detc) < 1 || error("`initial`: the coupling matrix has a determinant >= 1")
  g = sqrt(1 - detc)
  # (Any: a literal with Taylor series would convert the zeros with GTPSA.desc_current)
  V = Any[g 0 c11 c12; 0 g c21 c22; -c22 c12 g 0; c21 -c11 0 g]
  b1, b2 = sqrt(vals[:beta1]), sqrt(vals[:beta2])
  N = Any[b1 0 0 0; -vals[:alpha1]/b1 1/b1 0 0; 0 0 b2 0; 0 0 -vals[:alpha2]/b2 1/b2]

  a0 = one(id)
  a1 = one(id)
  orbit = (:x, :px, :y, :py, :z)
  for k in 1:4
    o = vals[orbit[k]]
    eta = vals[(:dx, :dpx, :dy, :dpy)[k]]
    TI.copy_tps!(a0.v[k], xi[k] + o + eta*xi[6])
    TI.copy_tps!(a1.v[k], sum(sum(V[k, l]*N[l, m] for l in 1:4)*xi[m] for m in 1:4))
  end
  TI.copy_tps!(a0.v[5], xi[5] + vals[:z])
  return NNF.factorise_a0(a0) ∘ a1
end

# The map with the coordinates v (and quaternion q, or nothing) of a map like template, e.g.
# saved by track
function _twiss_map(template, v, q)
  map = zero(template)
  NNF.setray!(map.v, v=v)
  if NNF.nvars(map) == 5 # coasting: δ is a parameter of the map, without the value of the orbit
    TI.seti!(map.v[6], 0, 0)
  end
  if !isnothing(map.q)
    NNF.setquat!(map.q, q=q)
  end
  return map
end
_view_q(q, j) = isnothing(q) ? nothing : view(q, 1, :, j, 2)
# The maps saved by track at its locations js (with m: from the start)
_twiss_maps(m, tr, js) = [_twiss_map(m, view(tr.data.v, 1, :, j, 2), _view_q(tr.data.q, j))
                          for j in js]

# The reference at the rows (the locations js of track), saved by track
function _twiss_reference!(beta_gamma_ref, t_ref, tr, js)
  beta_gamma_ref .= view(tr.data.beta_gamma_ref, js, 2)
  t_ref .= view(tr.data.t_ref, js, 2)
  return nothing
end

# The phase advances of the two transverse modes from the start to the rows (the steps step_save),
# in turns, from the linear maps to every integration step (tr_steps: the identity tracked at
# order 1 and saved at the steps 0, 1, ...) and the linear part of a. The phase of mode k at a
# step is that of the canonical form of M*A (as factorise: its element [2k-1, 2k] vanishes); it
# is followed from step to step, which are much less than half a turn apart. Only the whole
# turns of the result are used (_store_twiss!)
function _twiss_turns(tr_steps, a, step_save)
  A = NNF.jacobian(a, NNF.VARS)
  nv = size(A, 1)
  turns = (zeros(length(step_save)), zeros(length(step_save)))
  for k in 1:2
    advance = 0.0
    previous = atan(A[2k-1,2k], A[2k-1,2k-1])/(2*pi)
    row = 2 # (the first row is the start)
    for step in 1:last(step_save)
      x = tr_steps.data.v[1, 2k-1, step+1, 2]
      a1 = 0.0
      a2 = 0.0
      for m in 1:nv
        c = TI.geti(x, m)
        a1 += c*A[m,2k-1]
        a2 += c*A[m,2k]
      end
      phi = atan(a2, a1)/(2*pi)
      advance += phi - previous - round(phi - previous)
      previous = phi
      if step == step_save[row]
        turns[k][row] = advance
        row += 1
      end
    end
  end
  return turns
end

# The functions at every row from a at the start: composed with the maps to the rows, or a itself
# tracked (saved at the rows by track). With only the orbit asked for (orbit_only), the rows only
# need a0 (the translation to the fixed point, as factorise gives it), and only the first and the
# last are factorised (for the summary; the last one's phases are unwrapped with turns, and the
# longitudinal one against the first row). With transfer, also the map from the start to every
# row around the fixed point at the start (for the r columns): a (before factorise canonises it)
# composed with the inverse of a at the first row without its translation (which has no constant
# part, so that nothing is lost when the variables have lower orders than the parameters)
function _twiss_push_a(maps_or_tr, a_initial, canonise, phase, damp, coast, turns, orbit_only,
                       transfer)
  tracked = maps_or_tr isa TrackResult
  n = tracked ? size(maps_or_tr.data.v, 3) : length(maps_or_tr)
  fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3 = _twiss_make_base_columns(n, a_initial,
                                                                               phase, damp)
  a0s = Vector{typeof(a_initial)}(undef, n)
  transfers = Vector{typeof(a_initial)}(undef, transfer ? n : 0)
  a_start_inv = nothing
  for j in 1:n
    a = tracked ? _twiss_map(a_initial, view(maps_or_tr.data.v, 1, :, j, 2),
                             _view_q(maps_or_tr.data.q, j)) :
                  maps_or_tr[j] ∘ a_initial
    if transfer
      if j == 1
        shift = one(a)
        for p in 1:NNF.nvars(a)
          t = _linear_coefficient(a.v[p], 0, NNF.nvars(a))
          shift.v[p] += t - scalar(t)
        end
        a_start_inv = inv(inv(shift) ∘ a)
      end
      transfers[j] = a ∘ a_start_inv
    end
    if orbit_only && 1 < j < n
      a0s[j] = factorise_a0(a)
    else
      _store_twiss!(fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3, a, canonise, phase, damp,
                    j, coast, turns; jprev=orbit_only ? 1 : j - 1)
      a0s[j] = fac[j].a0
    end
  end
  return fac, a0s, transfers, phi1, phi2, phi3_or_slip, damp1, damp2, damp3
end
_view_q(q) = isnothing(q) ? nothing : view(q, 1, :)

# Tracks the map m (its coordinates: the orbit plus the map) through bl with track, saving it at the
# rows (locations, at the steps step_save) with the reference there. (Always with the reference,
# so that all of twiss's tracking compiles the same kernels)
function _twiss_track(m, bl, rf_on, batch_index, t_ref0, locations, step_save, in_body_coordinates,
                      delta_orbit; scalar_params=false)
  if NNF.nvars(m) == 5
    v = reshape([(i < 5 ? m.v0[i]+copy(m.v[i]) : (i == 6 ? delta_orbit+m.v[i] : copy(m.v[i])))
                 for i in 1:6], 1, 6)
  else
    v = reshape([m.v0[i]+copy(m.v[i]) for i in 1:6], 1, 6)
  end
  q = isnothing(m.q) ? nothing : [copy(m.q[1]) copy(m.q[2]) copy(m.q[3]) copy(m.q[4])]
  # t_ref must hold the type of the lengths
  p_over_q_ref = bl.p_over_q_ref isa TimeDependentParam ? bl.p_over_q_ref(0) : bl.p_over_q_ref
  b0 = Bunch(v=v, q=q, t_ref=t_ref0, species=bl.species_ref, p_over_q_ref=p_over_q_ref)
  config = TrackConfig(; in_body_coordinates, rf_on, batch_start=batch_index,
                          save_reference=true, use_KA=false, scalar_params)
  n_steps_turn = isempty(step_save) ? 0 : last(step_save)
  return _track(bl, b0, config, nothing, 0, locations, step_save, n_steps_turn, Val(true))
end

function _a_r_tunes(m::DAMap)
  mo = NNF.maxord(m)
  a = normal(m)
  c = c_map(m) # Transform to phasor basis
  r = inv(c) ∘ inv(a) ∘ m ∘ a ∘ c
  # Need to cut highest order
  Q_x = -cutord(angle(NNF.factor_out(r.v[1], 1))/(2*pi), mo)
  Q_y = -cutord(angle(NNF.factor_out(r.v[3], 3))/(2*pi), mo)
  if NNF.nvars(m) == 5
    Q_s = real(r.v[5])
    TI.seti!(Q_s, 0, 5) # subtract time identity
  else
    Q_s = -cutord(angle(NNF.factor_out(r.v[5], 5))/(2*pi), mo)
  end
  if isnothing(m.q)
    return a, r, SA[Q_x, Q_y, Q_s]
  else
    Q_spin = -atan(real(r.q.q2), real(r.q.q0))/pi # not two pi bc quaternion
    return a, r, SA[Q_x, Q_y, Q_s, Q_spin]
  end
end

function canonise_phase_damp(GTPSA_descriptor, coast, damping)
  desc = unsafe_load(GTPSA_descriptor.desc)
  mo = desc.mo
  no4 = unsafe_wrap(Vector{UInt8}, desc.no, 4)
  if mo == 1
    canonise = 0
    phase = MVector{3,Float64}(0,0,0)
    damp = damping ? MVector{3,Float64}(0,0,0) : nothing
  else
    if coast && no4 == SA[1,1,1,1]
      canonise = 1
    else
      canonise = 2
    end
    zer = TPS64(use=GTPSA_descriptor)
    phase = MVector{3,typeof(zer)}(zer, zero(zer), zero(zer))
    damp = damping ? MVector{3,typeof(zer)}(zero(zer), zero(zer), zero(zer)) : nothing
  end
  return canonise, phase, damp
end

# The functions at row j from a, the map from the start: the phases (and damping) are those from
# the start, the oscillation phases (all but the slip of a coasting beam) mod 1. Whole turns are
# added so that the phase is next to that of the previous row jprev (which differs by less than
# half a turn if the rows are all integration steps), or for the transverse modes next to the phase
# advance turns[k][j] followed through every step (if given: rows that may be further apart)
function _store_twiss!(fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3, a, canonise, phase, damp, j,
                       coast, turns; jprev=j-1)
  damping = !isnothing(damp)
  _clear!(phase)
  damping && _clear!(damp)
  facj = factorise(a; canonise=canonise, phase=phase, damp=damp, damping=damping,
                   compute_r=Val{false}())
  fac[j] = facj
  for (k, phi) in enumerate((phi1, phi2, phi3_or_slip))
    phi[j] = j != 1 ? copy(phase[k]) : zero(phase[k])
    if j != 1 && k <= 2 && !isnothing(turns)
      phi[j] += round(turns[k][j] - TI.scalar(phi[j]))
    elseif j != 1 && (k <= 2 || !coast)
      phi[j] += round(TI.scalar(phi[jprev]) - TI.scalar(phi[j]))
    end
  end
  if damping
    damp1[j] = j != 1 ? copy(damp[1]) : zero(damp[1])
    damp2[j] = j != 1 ? copy(damp[2]) : zero(damp[2])
    damp3[j] = j != 1 ? copy(damp[3]) : zero(damp[3])
  end
  return
end

# (in place: the TPS of an MVector can't be replaced)
function _clear!(x)
  for k in eachindex(x)
    if isbits(x[k])
      x[k] = zero(x[k])
    else
      TI.clear!(x[k])
    end
  end
  return x
end

function _twiss_make_base_columns(n, a::T, phase, damp) where {T}
  if !isnothing(a.q)
    fac = Vector{@NamedTuple{as::T, a0::T, a1::T, a2::T, a::T}}(undef, n)
  else
    fac = Vector{@NamedTuple{a0::T, a1::T, a2::T, a::T}}(undef, n)
  end
  phi1 = Vector{eltype(phase)}(undef, n)
  phi2 = Vector{eltype(phase)}(undef, n)
  phi3_or_slip = Vector{eltype(phase)}(undef, n) 
  if !isnothing(damp)
    damp1 = Vector{eltype(damp)}(undef, n) 
    damp2 = Vector{eltype(damp)}(undef, n) 
    damp3 = Vector{eltype(damp)}(undef, n) 
  else
    damp1 = nothing
    damp2 = nothing
    damp3 = nothing
  end
  return fac, phi1, phi2, phi3_or_slip, damp1, damp2, damp3
end

function _twiss_summ(twi, cache)
  j = length(twi.s)
  # The caches hold the last row of the table, which is not the end if the table does not include it
  empty!(cache.map)
  empty!(cache.tps)
  empty!(cache.float)
  empty!(cache.matrix)
  empty!(cache.vf)
  q1 = _phi1(j, twi, cache, Val{true}())
  coast = iscoasting(twi)
  oper = x -> (TI.is_tps_type(typeof(x)) isa TI.IsTPSType ? AmplitudeDependentValue(x, coast) : x)
  summ = LittleDict{Symbol,Union{Float64,AmplitudeDependentValue}}()
  summ[:q1] = oper(q1)
  summ[:q2] = oper(_phi2(j, twi, cache, Val{true}()))

  if !iscoasting(twi)
    summ[:q3] = oper(_phi3(j, twi, cache, Val{true}()))
  end

  etac = -_slip(j, twi, cache, Val{true}()) / (C_LIGHT * twi.t_ref[end])
  alphac = -_z_slip(j, twi, cache, Val{true}()) / twi.s[end] +
            _velocity_slip(j, twi, cache, Val{true}())
  if coast && TI.is_tps_type(typeof(etac)) isa TI.IsTPSType
    etac = TI.deriv(etac, 6)
  end
  if coast && TI.is_tps_type(typeof(alphac)) isa TI.IsTPSType
    alphac = TI.deriv(alphac, 6)
  end
  summ[:etac] = oper(etac)
  summ[:alphac] = oper(alphac)

  if !isnothing(twi.r_and_tunes) && length(twi.r_and_tunes[2]) == 4
    qspin = twi.r_and_tunes[2][end]
    if !(TI.is_tps_type(typeof(q1)) isa TI.IsTPSType) && !coast
      qspin = scalar(qspin)
    else
      qspin = AmplitudeDependentValue(qspin, coast, 0)
    end
    summ[:qspin] = qspin
  end
  

  if !isnothing(twi.damp1)
    summ[:damp1] = oper(twi.damp1[end])
    summ[:damp2] = oper(twi.damp2[end])
    summ[:damp3] = oper(twi.damp3[end])
  end
  return TwissSummary(summ)
end
