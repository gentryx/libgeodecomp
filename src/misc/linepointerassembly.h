#ifndef _libgeodecomp_misc_linepointerassembly_h_
#define _libgeodecomp_misc_linepointerassembly_h_

#include <libgeodecomp/misc/stencils.h>
#include <libgeodecomp/misc/streak.h>

namespace LibGeoDecomp {

/**
 * will initialize an array of pointers so that it can be used with
 * LinePointerNeighborhood.
 */
template<class STENCIL>
class LinePointerAssembly
{};

template<>
class LinePointerAssembly<Stencils::Moore<2, 1> >
{
public:
    template<typename CELL, typename GRID>
    void operator()(CELL *pointers[9], const Streak<2>& streak, GRID& grid)
    {
        pointers[0] = &grid[Coord<2>(streak.origin.x() - 1, streak.origin.y() - 1)];
        pointers[1] = &grid[Coord<2>(streak.origin.x() + 0, streak.origin.y() - 1)];
        pointers[2] = &grid[Coord<2>(streak.endX,           streak.origin.y() - 1)];
        pointers[3] = &grid[Coord<2>(streak.origin.x() - 1, streak.origin.y() + 0)];
        pointers[4] = &grid[Coord<2>(streak.origin.x() + 0, streak.origin.y() + 0)];
        pointers[5] = &grid[Coord<2>(streak.endX,           streak.origin.y() + 0)];
        pointers[6] = &grid[Coord<2>(streak.origin.x() - 1, streak.origin.y() + 1)];
        pointers[7] = &grid[Coord<2>(streak.origin.x() + 0, streak.origin.y() + 1)];
        pointers[8] = &grid[Coord<2>(streak.endX,           streak.origin.y() + 1)];
    }
};

}

#endif
