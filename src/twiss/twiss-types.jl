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