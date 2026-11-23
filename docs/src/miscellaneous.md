# Miscellaneous

## Reference Time in Misaligned Elements

By convention, the reference time at a point in an element does not depend on the element's
alignment. It is the same whether the particle's position is expressed in branch coordinates or
in element body coordinates (see [Coordinates](#c:coords)). The conversion between the phase
space coordinate `z` and the absolute time then depends only on where the particle is in the
element, and not on how the element is aligned.

### Alignment transformation and reference time

This convention has a consequence for the transformation at the ends of a misaligned element.
At the entrance, the particle starts at the nominal element edge (without alignment shifts), with
its position in branch coordinates. It is transformed to body coordinates, and then "drifted" to
the actual element face. The reference time at the start of this drift, in branch coordinates, is
the same as at its end, in body coordinates. So no reference time passes during the drift, and
`z` changes by the particle's whole path length, not, as in a drift element, by the difference
from the reference particle.

The same happens at the exit: the particle is transformed from body coordinates to branch
coordinates and "drifted" to the nominal downstream edge.
