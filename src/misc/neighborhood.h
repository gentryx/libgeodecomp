#ifndef _libgeodecomp_misc_neighborhood_h_
#define _libgeodecomp_misc_neighborhood_h_

#include <libgeodecomp/misc/fixedcoord.h>
#include <libgeodecomp/misc/stencils.h>

namespace LibGeoDecomp {

/**
 * gives cells access to their neighboring cells in a given stencil
 * shape. It is meant as a low-overhead replacement for CoordMap. The
 * flags are used in template specializations to cover those cases in
 * which accesses to neighboring cells need to be rerouted. In
 * essence, we go through the pain of having such many parameters to
 * resolve runtime conditionals for range checking at compile time.
 *
 * fixme:
 * BOUNDARY_WEST:
 *   X == -1 -> use WEST pointers
 =   X >=  0 -> use center pointers + X
 * 
 * BOUNDARY_EAST:
 *   X == 1 -> use EAST pointers
 *   X <= 0 -> use center pointers + X
 *
 * BOUNDARY_TOP:
 *   Y == -1 -> use TOP pointer (don't set this on a torus, as its not needed!)
 *   Y >=  0 -> use center pointers + X
 * 
 */
template<class CELL, class STENCIL, bool BOUNDARY_WEST, bool BOUNDARY_EAST, bool BOUNDARY_TOP, bool BOUNDARY_BOTTOM, bool BOUNDARY_SOUTH, bool BOUNDARY_NORTH>
class Neighborhood
{
};

template<class CELL, class STENCIL>
class Neighborhood<CELL, STENCIL, false, false, false, false, false, false>
{
public:
    Neighborhood(CELL **_lines, long *_offset) :
        lines(_lines),
        offset(_offset)
    {}

    template<int X, int Y, int Z>
    const CELL& operator[](FixedCoord<X, Y, Z>) const
    {
        return lines[1 + Y][X + *offset];
    }

private:
    CELL **lines;
    long *offset;
};


}

#endif
