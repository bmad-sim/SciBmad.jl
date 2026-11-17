module SciBmadMakieExt
using SciBmad, Makie

# A column of a Twiss table is plotted as the vector of its rows (e.g. lines(tw.s, tw.beta1))
Makie.convert_single_argument(c::SciBmad.TwissColumn) = collect(c)

end
