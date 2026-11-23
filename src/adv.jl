"""
    AmplitudeDependentValue

A value for the whole ring (e.g. a tune) as a Taylor series in δ (or the longitudinal action J₃
with synchrotron motion) and the actions J₁, J₂, and in GTPSA parameters if there are any.

- `q[delta=2]`, `q[J1=2, J2=1]`: a derivative, here ∂²q/∂δ² and ∂³q/∂J₁²∂J₂ (`q[delta=1]` is the
  chromaticity)
- `getterm(q; delta=2)`: the coefficient of that term of the Taylor series, i.e. the derivative
  divided by the factorials (∂²q/∂δ²/2), as the series is printed

With `as_taylor_series=true` the result is a Taylor series in the GTPSA parameters.
"""
struct AmplitudeDependentValue
  taylor_series::TPS64{GTPSA.Dynamic}
  coast::Bool
  # Degree in the phase space variables of the map terms that give the value without amplitudes:
  # 1 for the orbital tunes and the slip (from the linear part of the map), 0 for the spin tune
  degree0::Int
end
AmplitudeDependentValue(taylor_series, coast) = AmplitudeDependentValue(taylor_series, coast, 1)

Base.getindex(a::AmplitudeDependentValue; delta::Integer=0, J1::Integer=0, J2::Integer=0,
              J3::Integer=0, as_taylor_series::Bool=false) =
  factorial(delta)*factorial(J1)*factorial(J2)*factorial(J3) *
  _adv_get(a, delta, J1, J2, J3, Val{as_taylor_series}())

"""
    getterm(q::AmplitudeDependentValue; delta=0, J1=0, J2=0, J3=0, as_taylor_series=false)

The coefficient of the term δ^delta J₁^J1 J₂^J2 J₃^J3 of the Taylor series `q` (as it is printed).
`q[delta=..., J1=...]` is the derivative instead (the coefficient times the factorials).
"""
getterm(a::AmplitudeDependentValue; delta::Integer=0, J1::Integer=0, J2::Integer=0, J3::Integer=0, as_taylor_series::Bool=false) = _adv_get(a, delta, J1, J2, J3, Val{as_taylor_series}())
# A value computed at the truncation order 1 is a number: its only term is the constant
function getterm(x::Real; delta::Integer=0, J1::Integer=0, J2::Integer=0, J3::Integer=0,
                 as_taylor_series::Bool=false)
  delta == J1 == J2 == J3 == 0 && return x
  error("This value is a number, computed at the truncation order 1: its terms in δ and in the ",
        "amplitudes need `chrom` (δ) or `order` (the amplitudes) of at least 2")
end
function _adv_get(a, delta, J1, J2, J3, ::Val{as_taylor_series}) where {as_taylor_series}
  if a.coast && J3 != 0
    error("`J3` is NOT an amplitude in this value (beam is coasting), use `delta` instead")
  elseif !a.coast && delta != 0
    error("`delta` is NOT an amplitude in this value (beam is not coasting), use `J3` instead")
  end

  tps = a.taylor_series
  nn = TI.ndiffs(tps)
  mono = zeros(UInt8, nn)

  mono[1:2] .= J1
  mono[3:4] .= J2
  
  if a.coast
    mono[6] = delta
  else
    mono[5:6] .= J3
  end

  if !GTPSA.mad_desc_isvalidm(GTPSA.getdesc(tps).desc, Cint(nn), mono)
    error("Unable to get term (delta=$delta, J1=$J1, J2=$J2, J3=$J3): GTPSA truncation order(s) must be higher")
  end
  # The term comes from terms of the map of degree n = degree0 + delta + 2(J1 + J2 + J3) (the
  # tunes are cut below the maximum order): a term in δ alone needs the maximum order n and δ to
  # the power delta, a term in the amplitudes all phase space variables to order n (with lower
  # orders of some variables it is inexact). With lower truncation orders it would be zero or
  # wrong, silently
  desc = unsafe_load(GTPSA.getdesc(tps).desc)
  no = unsafe_wrap(Vector{UInt8}, desc.no, 6)
  n = a.degree0 + delta + 2*(J1 + J2 + J3)
  if desc.mo < n || (a.coast && no[6] < delta) ||
     (J1 + J2 + J3 > 0 && minimum(view(no, 1:(a.coast ? 4 : 6))) < n)
    error("The term (delta=$delta, J1=$J1, J2=$J2, J3=$J3) is not resolved by the truncation: ",
          J1 + J2 + J3 == 0 ? "it needs `chrom` (or `order`) of at least $n" :
                              "it needs `order` of at least $n")
  end

  if !as_taylor_series
    return TI.getm(tps, mono)
  else
    # cycle
    out = zero(tps)
    ords = similar(mono)
    v = Ref{TI.numtype(typeof(tps))}()
    idx = TI.cycle!(tps, -1, mono=ords, val=v) # -1 to include the constant term (index 0)
    while idx > -1
      if view(ords, 1:6) == view(mono, 1:6)
        ords[1:6] .= 0
        TI.setm!(out, v[], ords)
      end
      idx = TI.cycle!(tps, idx, mono=ords, val=v)
    end
    return out
  end
end

subscript(i) = join(Char(0x2080 + d) for d in reverse!(digits(i)))

function superscript(i)
  if i < 0
    c = [Char(0x207B)]
  else
    c = []
  end
  for d in reverse(digits(abs(i)))
    if d == 0 push!(c, Char(0x2070)) end
    if d == 1 push!(c, Char(0x00B9)) end
    if d == 2 push!(c, Char(0x00B2)) end
    if d == 3 push!(c, Char(0x00B3)) end
    if d > 3 push!(c, Char(0x2070+d)) end
  end
  return join(c)
end

function _show_adv(t)
  io = IOBuffer()
  tps = t.taylor_series
  coast = t.coast
  nn = TI.ndiffs(tps)
  first = true
  nhv = coast ? 4 : 6
  if scalar(tps) != 0
    val = scalar(tps)
    if sign(val) == -1
      print(io, "-")
    end
    print(io, repr(abs(val)))
    first = false
  end
  ords = zeros(UInt8, nn)
  v = Ref{TI.numtype(tps)}()
  idx = TI.cycle!(tps, 0, mono=ords, val=v)
  while idx > -1
    if NNF.is_tune_shift(0, ords, nhv, true) && abs(v[]) > 1e-10
      if first
        if sign(v[]) == -1
          print(io, "-")
        end
        first = false
      else
        if sign(v[]) == -1
          print(io, " - ")
        else
          print(io, " + ")
        end
      end
      print(io, repr(abs(v[])))
      j1ord = ords[1]
      j2ord = ords[3]
      j3deltaord = ords[6]
      if j3deltaord != 0
        if coast
          print(io, " δ")
        else
          print(io, " J₃")
        end
        if j3deltaord > 1
          print(io, superscript(j3deltaord))
        end
      end
      if j1ord != 0
        print(io, " J₁")
        if j1ord > 1
          print(io, superscript(j1ord))
        end
      end
      if j2ord != 0
        print(io, " J₂")
        if j2ord > 1
          print(io, superscript(j2ord))
        end
      end
      for k in 7:nn
        if ords[k] != 0
          print(io, " Δk", subscript(k-6))
          if ords[k] > 1
            print(io, superscript(ords[k]))
          end
        end
      end
    end
    idx = TI.cycle!(tps, idx, mono=ords, val=v)
  end
  return String(take!(io)) 
end

function Base.show(io::IO, t::AmplitudeDependentValue)
  println(io, "AmplitudeDependentValue:")
  println(io, " " * _show_adv(t))
  return
end

