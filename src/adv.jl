struct AmplitudeDependentValue
  taylor_series::TPS64{GTPSA.Dynamic}
  coast::Bool
end

Base.getindex(a::AmplitudeDependentValue; delta::Integer=0, J1::Integer=0, J2::Integer=0, J3::Integer=0, as_taylor_series::Bool=false) = _adv_get(a, delta, J1, J2, J3, Val{as_taylor_series}())

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
    error("Unable to get $s: GTPSA truncation order(s) must be higher")
  end

  if !as_taylor_series
    return TI.getm(tps, mono)
  else
    # cycle
    out = zero(tps)
    ords = similar(mono)
    v = Ref{TI.numtype(typeof(tps))}()
    idx = TI.cycle!(tps, 0, mono=ords, val=v)
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

