include(joinpath(ARGS[1], "12-sextupoles.jl"))
d = Dict{String,Float64}()
for n in ["qf","qd","qfss","qdss","qff1","qdf1","qfr1","qdr1","qff2","qdf2","qff3","qdf3","qfr2","qdr2","qfr3","qdr3",
          "qef1","qef2","qef3","qef4","qer1","qer2","qer3","qer4","qff2_2","qdf2_2","qff3_2","qdf3_2","qfss_2","qdss_2"]
  d[n*".Kn1"] = getfield(Main, Symbol(n)).Kn1
end
d["rf0.voltage"] = rf0.voltage
c = ring.context
for v in (:k2f, :k2d, :x1, :x2, :y1, :y2); d["context."*String(v)] = getproperty(c, v); end
for i in (3, 4); d["context.xf[$i]"] = c.xf[i]; d["context.xd[$i]"] = c.xd[i]; end
open(ARGS[2], "w") do io
  println(io, "{", join(["\"$k\": $(repr(v))" for (k, v) in sort(collect(d))], ", "), "}")
end
