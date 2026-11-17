module SciBmadRecipesBaseExt
using SciBmad, RecipesBase

# A column of a Twiss table is plotted as the vector of its rows (e.g. plot(tw.s, tw.beta1))
RecipesBase.@recipe f(::Type{T}, c::T) where {T<:SciBmad.TwissColumnVector} = collect(c)

end
