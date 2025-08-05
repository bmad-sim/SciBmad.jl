module Constants
using AtomicAndPhysicalConstants
@APCdef tupleflag=false
isnullspecies(species_ref::Species) = getfield(species_ref, :kind) == Kind.NULL
end