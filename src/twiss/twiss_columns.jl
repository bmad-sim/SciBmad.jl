#=

For parametric analysis, T is TPSA, else Float64
In the Twiss hot loop, TypedTables.jl is used internally
Then after fully constructed, the desired columns the user 
wants are extracted and placed into a DataFrame.

This is done 1) because TypedTables is already implemented 
and fast, 2) because then there isn't an explosion of different 
types and operations/callbacks inside the kernel depending on 
what columns the user wants, and 3) because there isn't really 
much cost extra to computing/not computing certain lattice functions.

Perhaps one day this could be checked if it really makes much difference 

Only linear dispersion approximation is included (1st order quantity)
To compute any higher-order chromatic quantities (e.g. chromaticity = 2nd order),
require RF off.


=#

struct Twiss{T}
  _summary::Dict{Symbol,T}
  table::DataFrame  # s-dependent quantities
end

function Base.getproperty(tw::Twiss, s::Symbol)
  if s == :table
    return getfield(tw, :table)
  elseif s == :_summary
    return getfield(tw, :_summary)
  else
    if haskey(summary, s)
      return summary[s]
    else
      error("Twiss summary does not have $s")
    end
  end
end

function make_twiss_utils(eye)
 if NNF.ndiffs(eye) == 6 
    extract(t) = scalar(t)
    extractdp(t, ord) = ord == 1 ? t[6] : t[UInt8[0,0,0,0,0,ord]]
    
  else
    let par_mono1=GTPSA.setup_mono(first(eye.v), [0,0,0,0,0,0,:], nothing, nothing),
      par_mono2=GTPSA.setup_mono(first(eye.v), [0,0,0,0,0,0,:], nothing, nothing),
      extract(t) = GTPSA.slice(t, par_mono1, false)
      extractdp(t, ord) = (par_mono2[6] .= ord; return GTPSA.slice(t, par_mono2, false);)
      summ = Dict{Symbol,}()
      return extract, extractdp, summ
    end
  end
end

# eye is passed bc contains all info about GTPSA statically
# Twiss will ALWAYS compute lattice functions at start and end
function make_twiss(eye::S, tunes, internal_lf_table) where {S}
  coast = NNF.nvars(eye) == 5
  T = NNF.ndiffs(eye) == 6 ? Float64 : eltype(eye.v)
  summ = Dict{Symbol,T}()
  extract, extractdp = make_twiss_utils(eye)

  lfend = internal_lf_table[end]

  # Orbit path length:
  Lco = extract(lfend.s + lfend.orbit_z)

  summ[:q1] = extract(tunes[1])
  summ[:q2] = extract(tunes[2])

  if coast
    summ[:alpha_c] = extractdp(tunes[3], 1) / Lco
    # 
    #summ[:eta_p] = 
    #summ[:gamma_t] = 
  else
    q3 = extract(tunes[3])
    summ[:q3] = q3
    if haskey(lfend, :B)
      B = lfend.B
      if B isa Matrix
        summ[:alpha_c] = B[5,6]*sin(q3*2*pi) / Lco
      else
        summ[:alpha_c] = extractdp(B.v[5], 1)*sin(q3*2*pi) / Lco
      end
    else
      summ[:alpha_c] = extract(lfend.slip) / Lco
    end
  end
end

#=
  q1::T       # Fractional horizontal-like tune
  q2::T       # Fractional vertical-like tune
  q3::T       # Fractional longitudinal-like tune (0 if coasting beam)
  alpha_c::T  # Momentum compaction factor
  eta_c::T    # Slip factor
  gamma_t::T  # Gamma transition
=#

function beta_1(tw::Twiss)

  return (; beta_1 =0 )
end