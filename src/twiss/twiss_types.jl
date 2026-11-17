struct TwissSummary
  sdict::LittleDict{Symbol,Union{Float64,AmplitudeDependentValue}}
end

Base.propertynames(summ::TwissSummary) = [keys(summ.sdict)...]
function Base.getproperty(summ::TwissSummary, s::Symbol)
  if s == :sdict
    return getfield(summ, :sdict)
  else
    getfield(summ, :sdict)[s]
  end
end
function _show_summ(summ::TwissSummary)
  io = IOBuffer()
  width = length(" alphac") # longest string 
  for (k,v) in summ.sdict
    println(io, rpad(" " * string(k), width), " = ", v isa AmplitudeDependentValue ? _show_adv(v) : repr(v))
  end
  count = length(keys(summ.sdict))
  return String(take!(io)), count
end

Base.show(io::IO, summ::TwissSummary) = print(io, "TwissSummary:\n", _show_summ(summ)[1])

struct Twiss
  summ::TwissSummary
  df::DataFrame         # s-dependent quantities
  bl::Union{Beamline,Nothing} # the Beamline, to find the rows of its elements
  start::Int                  # the beamline index of its beginning (of `start`)
  stop::Union{Int,Nothing}    # the beamline index of `stop` (nothing: the end of the beamline)
  rows::Dict{Tuple{Int,Int},Int} # the row of each location (beamline index, step in the element)
end
Twiss(summ, df) = Twiss(summ, df, nothing, 1, nothing, Dict{Tuple{Int,Int},Int}())

function Base.getproperty(tw::Twiss, s::Symbol)
  if s == :df
    return getfield(tw, :df)
  elseif s == :summ
    return getfield(tw, :summ)
  elseif s == :bl
    return getfield(tw, :bl)
  else
    summ = getfield(tw, :summ)
    df = getfield(tw, :df)
    if haskey(summ.sdict, s)
      return summ.sdict[s]
    elseif hasproperty(df, s)
      return TwissColumn(getproperty(df, s), tw)
    elseif (m = match(r"^dq([12])(?:_([1-9]))?$", String(s))) !== nothing
      # The chromaticities dq1, dq2 and their derivatives dq1_n = ∂ⁿq1/∂δⁿ
      q = summ.sdict[m.captures[1] == "1" ? :q1 : :q2]
      n = isnothing(m.captures[2]) ? 1 : parse(Int, m.captures[2])
      if !(q isa AmplitudeDependentValue) || !q.coast
        error("The chromaticity `$s` needs `chrom` (or `order`) of at least $(n+1) and a " *
              "coasting beam (`rf_on=false` if there are RF cavities)")
      end
      return q[delta=n]
    else
      error("Twiss does not have $s")
    end
  end
end

Base.propertynames(tw::Twiss) = vcat(propertynames(tw.summ), propertynames(tw.df))

"""
    TwissColumn

A column of a `Twiss` table (e.g. `tw.beta1`), indexed with locations of the beamline exactly as
the columns of a [`TwissExpr`](@ref):

- `tw.beta1[qd]`, `tw.beta1[5]`: at the beginning of the element `qd`, or of the element with
  beamline index 5. An element that is in the beamline several times gives a vector with one
  entry per placement, in beamline order
- `tw.beta1[begin]`, `tw.beta1[end]`: at the beginning and the end of the beamline (or of the part
  of it given with `start` and `stop`)
- `tw.beta1[qd, :exit]`: at the end of the element (the beginning of the next one)
- `tw.beta1[qd, :]`: at every integration step of the element, both ends included
- `tw.beta1[qd, 2]`: two integration steps after the beginning of the element
- `tw.beta1["q*"]`, `tw.beta1[r"..."]`, `tw.beta1[[qf, qd]]`, `tw.beta1[2:5]`: at the beginning
  of the elements found by name (as for `findelements`), or of each element given (a vector)
- `tw.beta1[(2.0, 10.0)]`, `tw.beta1[:]`: at every integration step in an s-range (and the end
  of the beamline if the range contains it), or of the whole beamline

The table must have rows at these locations (it has with the default `at`). The column also
iterates over all rows of the table, in order, and broadcasts (`tw.dbeta1 ./ tw.beta1`) as a
vector of them; `collect(tw.beta1)` (or `tw.df.beta1`) is that vector.
"""
struct TwissColumn{T,V<:AbstractVector{T}}
  v::V
  tw::Twiss
end
Base.length(c::TwissColumn) = length(c.v)
Base.size(c::TwissColumn) = size(c.v)
Base.eltype(::Type{<:TwissColumn{T}}) where {T} = T
Base.iterate(c::TwissColumn, state...) = iterate(c.v, state...)
Base.broadcastable(c::TwissColumn) = c.v
Base.collect(c::TwissColumn) = collect(c.v)
# (comparisons are of all rows, with another column or a vector)
Base.:(==)(a::TwissColumn, b::TwissColumn) = a.v == b.v
Base.:(==)(a::TwissColumn, b::AbstractVector) = a.v == b
Base.:(==)(a::AbstractVector, b::TwissColumn) = a == b.v
Base.isapprox(a::TwissColumn, b::TwissColumn; kw...) = isapprox(a.v, b.v; kw...)
Base.isapprox(a::TwissColumn, b::AbstractVector; kw...) = isapprox(a.v, b; kw...)
Base.isapprox(a::AbstractVector, b::TwissColumn; kw...) = isapprox(a, b.v; kw...)
# (with more than one index, e.g. beta1[end, :exit], `end` is lastindex(c, 1))
Base.firstindex(c::TwissColumn, d...) = getfield(c.tw, :start)
Base.lastindex(::TwissColumn, d...) = _End()

function Base.getindex(c::TwissColumn, args...)
  tw = c.tw
  bl = getfield(tw, :bl)
  isnothing(bl) && error("This Twiss table does not know its Beamline")
  locs, one = _column_locations(bl, getfield(tw, :stop), args...)
  vals = [c.v[_twiss_row(tw, l)] for l in locs]
  return one ? only(vals) : vals
end

# The row of the table at location (i, k): k integration steps after the beginning of the element
# with beamline index i (-1: the end of the beamline)
function _twiss_row(tw::Twiss, (i, k))
  r = get(getfield(tw, :rows), (i, k), 0)
  r == 0 && error("The Twiss table has no row ", i == -1 ? "at the end of the beamline" :
                  k == 0 ? "at the beginning of element $i" : "at step $k of element $i",
                  ": include it with `at`")
  return r
end

function Base.show(io::IO, c::TwissColumn)
  print(io, "TwissColumn(")
  show(io, c.v)
  print(io, ")")
end
function Base.show(io::IO, m::MIME"text/plain", c::TwissColumn)
  println(io, length(c.v), "-row column of a Twiss table (index it with locations of the ",
          "beamline; collect it for a Vector of all rows):")
  Base.print_array(io, c.v)
end

function Base.show(io::IO, tw::Twiss)
  # Note: copy-pasted
  summ = tw.summ
  df = tw.df
  println(io, "Twiss:")
  str, count = _show_summ(summ)
  println(io, "summ:")
  print(io, str)
  println(io, "\ndf:")
  units = [something(DataFrames.colmetadata(df, col, "unit", nothing), "") for col in names(df)]
  show(io, df; eltypes=true, column_labels = [names(df), units], reserved_display_lines = 2+count+4)
end

struct TwissInternal{T,F,P,D,R}
  s::Vector{T}
  name::Vector{String}
  kind::Vector{String}
  index::Vector{Int}
  beta_gamma_ref::Vector{Float64}
  t_ref::Vector{T}
  fac::F
  phi1::P
  phi2::P
  phi3_or_slip::P
  damp1::D
  damp2::D
  damp3::D
  r_and_tunes::R
  delta0::Float64 # δ of the orbit of a coasting beam
end

ndiffs(twi::TwissInternal) = NNF.ndiffs(twi.fac[1].a)
nparams(twi::TwissInternal) = NNF.nparams(twi.fac[1].a)
nvars(twi::TwissInternal) = NNF.nvars(twi.fac[1].a)
nhvars(twi::TwissInternal) = NNF.nhvars(twi.fac[1].a)
iscoasting(twi::TwissInternal) = NNF.iscoasting(twi.fac[1].a)
maxord(twi::TwissInternal) = NNF.maxord(twi.fac[1].a)
noi(twi::TwissInternal, i) = unsafe_load(unsafe_load(GTPSA.getdesc(first(twi.fac[1].a.v)).desc).no, i)

struct TwissCache{M,T,F,A,VF,CM}
 map::M
 tps::T
 float::F
 matrix::A
 vf::VF
 persistent_matrix::A
 persistent_map::M
 persistent_cmap::CM
end

function build_cache(::Type{T}, n=10) where {T}
  keys = Vector{Symbol}(undef, 0)
  vals = Vector{T}(undef, 0)
  sizehint!(keys, n)
  sizehint!(vals, n)
  return LittleDict(keys, vals)
end


