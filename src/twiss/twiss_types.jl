struct Twiss
  summ::LittleDict{Symbol,Union{Float64,AmplitudeDependentValue}}
  df::DataFrame         # s-dependent quantities
end

function Base.getproperty(tw::Twiss, s::Symbol)
  if s == :df
    return getfield(tw, :df)
  elseif s == :summ
    return getfield(tw, :summ)
  else
    summ = getfield(tw, :summ)
    df = getfield(tw, :df)
    if haskey(summ, s)
      return summ[s]
    elseif hasproperty(df, s)
      return getproperty(df, s)
    else
      error("Twiss summ does not have $s")
    end
  end
end

Base.propertynames(tw::Twiss) = vcat(keys(tw.summ)..., propertynames(tw.df))

function Base.show(io::IO, tw::Twiss)
  # Note: copy-pasted
  summ = tw.summ
  df = tw.df
  println(io, "Twiss:")
  width = length(" alphac") # longest string 
  println(io, "summ:")
  for (k,v) in summ
    println(io, rpad(" " * string(k), width), " = ", v isa AmplitudeDependentValue ? _show_adv(v) : repr(v))
  end
  count = length(keys(summ))
  println(io, "\ndf:")
  units = [something(DataFrames.colmetadata(df, col, "unit", nothing), "") for col in names(df)]
  show(io, df; eltypes=true, column_labels = [names(df), units], reserved_display_lines = 2+count+4)
end

struct TwissInternal{F,P,D,R}
  s::Vector{Float64}
  name::Vector{String}
  kind::Vector{String}
  index::Vector{Int}
  beta_gamma_ref::Vector{Float64}
  t_ref::Vector{Float64}
  fac::F
  phi1::P
  phi2::P
  phi3_or_slip::P
  damp1::D
  damp2::D
  damp3::D
  r_and_tunes::R
end

ndiffs(twi::TwissInternal) = NNF.ndiffs(twi.fac[1].a)
nparams(twi::TwissInternal) = NNF.nparams(twi.fac[1].a)
nvars(twi::TwissInternal) = NNF.nvars(twi.fac[1].a)
nhvars(twi::TwissInternal) = NNF.nhvars(twi.fac[1].a)
iscoasting(twi::TwissInternal) = NNF.iscoasting(twi.fac[1].a)
maxord(twi::TwissInternal) = NNF.maxord(twi.fac[1].a)
noi(twi::TwissInternal, i) = unsafe_load(unsafe_load(GTPSA.getdesc(first(twi.fac[1].a.v)).desc).no, i)

struct TwissCache{M,T,F,SM4,SM6,VF,CM}
 map::M
 tps::T
 float::F
 smatrix4::SM4
 smatrix6::SM6
 vf::VF
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


