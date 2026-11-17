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
end
Twiss(summ, df) = Twiss(summ, df, nothing)

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
      return TwissColumnVector(getproperty(df, s), tw)
    elseif s == :dq1 || s == :dq2 # the chromaticities
      q = summ.sdict[s == :dq1 ? :q1 : :q2]
      if !(q isa AmplitudeDependentValue) || !q.coast
        error("The chromaticity `$s` needs `chrom` (or `order`) of at least 2 and a coasting " *
              "beam (`rf_on=false` if there are RF cavities)")
      end
      return getterm(q, delta=1)
    else
      error("Twiss does not have $s")
    end
  end
end

Base.propertynames(tw::Twiss) = vcat(propertynames(tw.summ), propertynames(tw.df))

"""
    TwissColumnVector

A column of a `Twiss` table (e.g. `tw.beta1`): a vector with one entry per row of the table,
which can also be indexed with elements of the beamline:

- `tw.beta1[i]`, `tw.beta1[end]`, `tw.beta1[2:5]`: rows of the table, as for any vector
- `tw.beta1[ele]`: at the beginning of the element `ele`. An element that is in the beamline
  several times gives a vector with one entry per placement, in beamline order
- `tw.beta1[ele, :exit]`: at the end of the element (the beginning of the next one)
- `tw.beta1[ele, :]`: at every integration step of the element, both ends included
- `tw.beta1["q*"]`, `tw.beta1[r"..."]`: at the beginning of the elements found by name, as for
  `findelements` (always a vector)

The rows must be in the table (they are with the default `at`). The same locations are
used by [`TwissExpr`](@ref), where however an integer is a beamline index (here it is a row).
"""
struct TwissColumnVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
  v::V
  tw::Twiss
end
Base.size(c::TwissColumnVector) = size(c.v)
Base.IndexStyle(::Type{<:TwissColumnVector}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(c::TwissColumnVector, i::Int) = c.v[i]
Base.@propagate_inbounds Base.setindex!(c::TwissColumnVector, x, i::Int) = (c.v[i] = x; c)

# The rows of the table at the beginning of the element with beamline index i (-1: the end of
# the beamline), in the order of its integration steps
function _twiss_rows(c::TwissColumnVector, i)
  df = c.tw.df
  hasproperty(df, :index) || error("The Twiss table has no `index` column (see `base_cols`)")
  rows = findall(==(i), df.index)
  isempty(rows) && error("The Twiss table has no row ", i == -1 ? "at the end of the beamline" :
                         "of element $i", ": include it with `at`")
  return rows
end

function _twiss_element_indexes(c::TwissColumnVector, sel)
  bl = c.tw.bl
  isnothing(bl) && error("This Twiss table does not know its Beamline")
  return _element_indexes(bl, sel)
end

function Base.getindex(c::TwissColumnVector, sel::LatElement)
  vals = [c.v[first(_twiss_rows(c, i))] for i in _twiss_element_indexes(c, sel)]
  return length(vals) == 1 ? only(vals) : vals
end
Base.getindex(c::TwissColumnVector, sel::Union{AbstractString,Regex}) =
  [c.v[first(_twiss_rows(c, i))] for i in _twiss_element_indexes(c, sel)]
function Base.getindex(c::TwissColumnVector, sel::Union{LatElement,AbstractString,Regex},
                       pos::Union{Symbol,AbstractString,Colon})
  idxs = _twiss_element_indexes(c, sel)
  n = length(c.tw.bl.line)
  pos isa AbstractString && (pos = Symbol(pos))
  exit_row(i) = first(_twiss_rows(c, i == n ? -1 : i + 1))
  if pos isa Colon || pos == :all
    return reduce(vcat, [c.v[[_twiss_rows(c, i); exit_row(i)]] for i in idxs])
  elseif pos == :exit
    vals = [c.v[exit_row(i)] for i in idxs]
  elseif pos == :entrance
    vals = [c.v[first(_twiss_rows(c, i))] for i in idxs]
  else
    error("Unknown position $(repr(pos)): use :entrance, :exit or :")
  end
  return sel isa LatElement && length(vals) == 1 ? only(vals) : vals
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


