#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDNEIGHBORHOODNEW_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDNEIGHBORHOODNEW_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

namespace LibGeoDecomp {

/**
 * Simple neighborhood which is used for hoodNew in updateLineX().
 * Provides access to cells via an identifier which is returned by
 * hoodOld (see Iterator classe above).
 */
template<
    typename CELL,
    std::size_t MATRICES = 1,
    typename VALUE_TYPE = double,
    int C = 64,
    int SIGMA = 1>
class UnstructuredNeighborhoodNew
{
public:
    using Grid = ReorderingUnstructuredGrid<UnstructuredGrid<CELL, MATRICES, VALUE_TYPE, C, SIGMA> >;

    inline explicit
    UnstructuredNeighborhoodNew(Grid& grid) :
        grid(grid)
    {}

    inline
    CELL& operator[](int index)
    {
        return grid[index];
    }

    inline
    const CELL& operator[](int index) const
    {
        return grid[index];
    }

private:
    Grid& grid;                 /**< new grid */
};

}

#endif
#endif
