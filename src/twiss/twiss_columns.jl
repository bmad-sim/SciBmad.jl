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

  q1::T       # Fractional horizontal-like tune
  q2::T       # Fractional vertical-like tune
  q3::T       # Fractional longitudinal-like tune (0 if coasting beam)
  alpha_c::T  # Momentum compaction factor
  eta_c::T    # Slip factor
  gamma_t::T  # Gamma transition



function beta_1(tw::Twiss)

  return (; beta_1 =0 )
end