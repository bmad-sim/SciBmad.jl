

function Base.show(io::IO, tw::Twiss)
  println(io, "Twiss:")
  width = length(" coasting_beam")
  println(io, rpad(" coasting_beam", width), " = ", tw.coasting_beam)
  spin = length(tw.tunes) == 4

  print(io, rpad(" tunes[1:$(length(tw.tunes))]", width), " = [Qx, Qy")
  if tw.coasting_beam
    print(io, ", slip")
  else
    print(io, ", Qz")
  end
  if spin
    print(io, ", Qspin")
  end

  print(io, "]\n")

  if !isnothing(tw.table)
    print(io, rpad(" table", width), " has columns: ") 
    cols = keys(getfield(tw.table, :data))
    for col in cols
      print(io, String(col))
      if col != last(cols)
        print(io, ", ")
      end
    end
  end
  return
end
