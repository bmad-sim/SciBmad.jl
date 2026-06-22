"""
    struct Observable

Stores information about an `Observable`, which can 
then be used to construct `twiss` call to compute it 
and its gradient value.
"""


()-> begin
  tw = twiss(
    beamline; 
    GTPSA_descriptor, # Requires np -> number of knobs
    spin, 
    de_moivre, 
    normalizing_map, 
    RDTs, 
    at, 
    in_body_coordinates, 
    v0,
    v0_and_coast,
    symplectic_tol=1e-8,
  )

  # Now we need to define expressions that extract the given quantities
  # e.g. chromaticity would have to do 
  # tw.tunes[1][[0,0,0,0,0,1,:]] # includes the derivatives
end

struct Observable
  what::Symbol
  s_range::NTuple{2,Number}
  beamline::Beamline
  no::Vector{UInt8}
  mo::UInt8
  po::UInt8
end


# "what" will tell you if de_moivre, if spin
const DATUM_GTPSA_MAP = Dict{Symbol,Tuple{}}

# Calling TimeDependentParam
(d::TimeDependentParam)(t) = d.f(t)

# Conversion of types to TimeDependentParam
TimeDependentParam(a::Number) = TimeDependentParam((t)->a, true) 
TimeDependentParam(a::TimeDependentParam) = a

# Make these apply via convert
Base.convert(::Type{D}, a::Number) where {D<:TimeDependentParam} = D((t)->a,true)
Base.convert(::Type{D}, a::D) where {D<:TimeDependentParam} = a

Base.zero(::TimeDependentParam) = TimeDependentParam((t)->0, true)
Base.one(::TimeDependentParam) = TimeDependentParam((t)->1, true)

# Now define the math operations:
for op in (:+,:-,:*,:/,:^)
  @eval begin
    Base.$op(da::TimeDependentParam, b::Number)   = (let fa = da.f, _isconst=da._isconst, b = b; return TimeDependentParam((t)-> $op(fa(t), b), _isconst); end)
    Base.$op(a::Number,   db::TimeDependentParam) = (let fb = db.f, _isconst=db._isconst, a = a; return TimeDependentParam((t)-> $op(a, fb(t)), _isconst); end)
    function Base.$op(da::TimeDependentParam, db::TimeDependentParam)
      let fa = da.f, fb = db.f, _isconst = da._isconst && db._isconst # true only if both are const
        return TimeDependentParam((t)-> $op(fa(t), fb(t)), _isconst)
      end
    end
  end
end

function Base.literal_pow(::typeof(^), da::TimeDependentParam, ::Val{N}) where {N} 
  let fa = da.f, _isconst=da._isconst
    return TimeDependentParam((t)->Base.literal_pow(^, fa(t), Val{N}()), _isconst)
  end
end

for t = (:+, :-, :sqrt, :exp, :log, :sin, :cos, :tan, :cot, :sinh, :cosh, :tanh, :inv,
  :coth, :asin, :acos, :atan, :acot, :asinh, :acosh, :atanh, :acoth, :sinc, :csc, :float,
  :csch, :acsc, :acsch, :sec, :sech, :asec, :asech, :conj, :log10, :isnan, :sign, :abs)
  @eval begin
    Base.$t(d::TimeDependentParam) = (let f = d.f, _isconst = d._isconst; return TimeDependentParam((t)-> ($t)(f(t)), _isconst); end)
  end
end

atan2(d1::TimeDependentParam, d2::TimeDependentParam) = (let f1 = d1.f, f2 = d2.f, _isconst = d1._isconst && d2._isconst; return TimeDependentParam((t)->atan2(f1(t),f2(t)), _isconst); end)

for t = (:unit, :sincu, :sinhc, :sinhcu, :asinc, :asincu, :asinhc, :asinhcu, :erf, 
         :erfc, :erfcx, :erfi, :wf, :rect)
  @eval begin
    GTPSA.$t(d::TimeDependentParam) = (let f = d.f, _isconst = d._isconst; return TimeDependentParam((t)-> ($t)(f(t)), _isconst); end)
  end
end

Base.promote_rule(::Type{TimeDependentParam}, ::Type{U}) where {U<:Number} = TimeDependentParam
Base.broadcastable(o::TimeDependentParam) = Ref(o)

Base.isapprox(d::TimeDependentParam, n::Number; kwargs...) = d._isconst ? isapprox(d(0), n) : false
Base.isapprox(n::Number, d::TimeDependentParam; kwargs...) = d._isconst ? isapprox(n, d(0)) : false
Base.:(==)(d::TimeDependentParam, n::Number) = d._isconst ? d(0) == n : false
Base.:(==)(n::Number, d::TimeDependentParam) = d._isconst ? n == d(0) : false
Base.isinf(d::TimeDependentParam) = d._isconst ? isinf(d(0)) : false

@inline teval(f::TimeFunction, t) = f(t)
@inline teval(f, t) = f

# === THIS BLOCK WAS PARTIALLY WRITTEN BY CLAUDE ===
# Generated function for arbitrary-length tuples
@generated function teval(f::T, t) where {T<:Tuple}
    N = length(T.parameters)
    # Use getfield with literal integer arguments
    exprs = [:(teval(Base.getfield(f, $i), t)) for i in 1:N]
    return :(tuple($(exprs...)))
end
# === END CLAUDE ===

time_lower(tp::TimeDependentParam) = tp.f
time_lower(tp) = tp
# We can use map on the CPU, but not the GPU. This step of time_lower-ing is on
# the CPU and we are already type unstable here anyways, so we should do this.
time_lower(tp::T) where {T<:Tuple} = map(ti->time_lower(ti), tp)

# Arrays MUST be converted into tuples, for SIMD
time_lower(tp::SArray{N,TimeDependentParam}) where {N} = time_lower(Tuple(tp))
static_timecheck(tp) = false
static_timecheck(::TimeFunction) = true
@unroll function static_timecheck(t::Tuple)
  @unroll for ti in t
    if static_timecheck(ti)
      return true
    end
  end
  return false
end