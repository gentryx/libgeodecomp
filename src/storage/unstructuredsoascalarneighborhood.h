#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDSOASCALARNEIGHBORHOOD_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDSOASCALARNEIGHBORHOOD_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libgeodecomp/storage/unstructuredsoaneighborhood.h>

namespace LibGeoDecomp {

/**
 * This neighborhood is used in SoA cells in update() function. Update()
 * may be called due to loop peeling. The only differences to
 * UnstructuredNeighborhood are the grid type and the []-operator.
 */
template<
    typename GRID,
    typename CELL,
    std::size_t MATRICES = 1,
    typename VALUE_TYPE = double,
    int C = 64,
    int SIGMA = 1>
class UnstructuredSoAScalarNeighborhood : public UnstructuredNeighborhoodHelpers::UnstructuredNeighborhoodBase<
    CELL, GRID, MATRICES, VALUE_TYPE, C, SIGMA, false>
{
public:
    using UnstructuredNeighborhoodHelpers::UnstructuredNeighborhoodBase<CELL, GRID, MATRICES, VALUE_TYPE, C, SIGMA, false>::grid;

    inline
    UnstructuredSoAScalarNeighborhood(const GRID& grid, long startX) :
        UnstructuredNeighborhoodHelpers::UnstructuredNeighborhoodBase<CELL, GRID, MATRICES, VALUE_TYPE, C, SIGMA, false>(grid, startX)
    {}

    CELL operator[](int index) const
    {
        return grid.get(Coord<1>(index));
    }
};

}

#endif
#endif
