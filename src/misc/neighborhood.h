#ifndef _libgeodecomp_misc_neighborhood_h_
#define _libgeodecomp_misc_neighborhood_h_

#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/fixedcoord.h>

namespace LibGeoDecomp {

// fixme: move!

class Stencils
{
public:
    class Moore
    {
    };

    class VonNeumann
    {
    };
};

/**
 * gives cells access to their neighboring cells in a given stencil
 * shape. It is meant as a low-overhead replacement for CoordMap.
 */
template<class CELL, class STENCIL>
class Neighborhood
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
