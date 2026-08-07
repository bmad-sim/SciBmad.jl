struct Twiss{T}
  summary::Dict{Symbol,T}
  table::DataFrame  # s-dependent quantities
end

function Base.getproperty(tw::Twiss, s::Symbol)
  if s == :table
    return getfield(tw, :table)
  elseif s == :summary
    return getfield(tw, :summary)
  else
    if haskey(summary, s)
      return summary[s]
    else
      error("Twiss summary does not have $s")
    end
  end
end

struct TwissInternal{F,P,D,R}
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

struct TwissCache{M,T,F,SM,SV}
 map::M
 tps::T
 float::F
 smatrix4::SM
 smatrix6::SV
end

function build_cache(::Type{T}, n=10) where {T}
  keys = Vector{Symbol}(undef, 0)
  vals = Vector{T}(undef, 0)
  sizehint!(keys, n)
  sizehint!(vals, n)
  return LittleDict(keys, vals)
end