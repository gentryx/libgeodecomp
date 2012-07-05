#ifndef _libgeodecomp_misc_pointerneighborhood_h_
#define _libgeodecomp_misc_pointerneighborhood_h_

#include <libgeodecomp/misc/fixedcoord.h>
#include <libgeodecomp/misc/stencils.h>

namespace LibGeoDecomp {

/**
 * provides a neighborhood which can be adressed by parameters known
 * at compile time. It uses an array of pointers to access the cells,
 * which makes it suitable for any topology and storage. Short-lived
 * instances should be optimized away by the compiler.
 */
template<typename CELL, typename STENCIL>
class PointerNeighborhood 
{
public:
    PointerNeighborhood(CELL **cells) :
        cells(cells)
    {}

    template<int X, int Y, int Z>
    const CELL& operator[](FixedCoord<X, Y, Z>) const
    {
        return cells[Stencils::OffsetHelper<STENCIL, X, Y, Z>::VALUE][0];
    }

private:
    CELL **cells;
};

}

#endif
