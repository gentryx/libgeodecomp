#ifndef _libgeodecomp_misc_coordmap_h_
#define _libgeodecomp_misc_coordmap_h_

#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/topologies.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE, typename TOPOLOGY>
class Grid;

template<typename CELL_TYPE, typename GRID_TYPE=Grid<CELL_TYPE, Topologies::Cube<2>::Topology> >
class CoordMap 
{
public:
    const static int DIMENSIONS = GRID_TYPE::DIMENSIONS;

    inline CoordMap(const Coord<DIMENSIONS>& origin, GRID_TYPE *grid) :
        _origin(origin), _grid(grid) {};

    /**
     * This operator doesn't implement out-of-bounds-checking. This
     * isn't a bug, it's a feature. It allows us to remain independent
     * of the currently used neighborhood definition and it is
     * unlikely to lead to errors as the Grid itself implements range
     * checks.
     */
    inline const CELL_TYPE& operator[](const Coord<DIMENSIONS>& relCoord) const
    {
        return (*_grid)[_origin + relCoord];
    }


    std::string toString() const
    {
        return "CoordMap origin: " + _origin.toString() + "\n";
    }


private:
    Coord<DIMENSIONS> _origin;
    GRID_TYPE *_grid;
};

};

#endif
