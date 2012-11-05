#ifndef _libgeodecomp_misc_coordmap_h_
#define _libgeodecomp_misc_coordmap_h_

#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/fixedcoord.h>
#include <libgeodecomp/misc/topologies.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE, typename TOPOLOGY>
class Grid;

/**
 * provides access to neighboring cells in a grid via relative coordinates. Slow!
 */
template<typename CELL_TYPE, typename GRID_TYPE=Grid<CELL_TYPE, Topologies::Cube<2>::Topology> >
class CoordMap 
{
public:
    const static int DIM = GRID_TYPE::DIM;

    inline CoordMap(const Coord<DIM>& origin, const GRID_TYPE *grid) :
        _origin(origin), _grid(grid) {};

    /**
     * This operator doesn't implement out-of-bounds-checking. This
     * isn't a bug, it's a feature. It allows us to remain independent
     * of the currently used neighborhood definition and it is
     * unlikely to lead to errors as the Grid itself implements range
     * checks.
     */
    inline const CELL_TYPE& operator[](const Coord<DIM>& relCoord) const
    {
        return (*_grid)[_origin + relCoord];
    }

    template<int X, int Y, int Z>
    inline const CELL_TYPE& operator[](FixedCoord<X, Y, Z> relCoord) const
    {
        return (*this)[Coord<DIM>(relCoord)];
    }

    std::string toString() const
    {
        return "CoordMap origin: " + _origin.toString() + "\n";
    }

private:
    Coord<DIM> _origin;
    const GRID_TYPE *_grid;
};

};

#endif
