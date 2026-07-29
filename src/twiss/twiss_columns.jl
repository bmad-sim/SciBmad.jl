struct Twiss{S,T}
  coasting_beam::Bool
  tunes::S
  table::T
end

function beta_1(tw::Twiss)

  return (; beta_1 =0 )
end