#ifndef LIBGEODECOMP_GEOMETRY_GRIDSPACINGCALCULATOR_H
#define LIBGEODECOMP_GEOMETRY_GRIDSPACINGCALCULATOR_H

#include <cmath>

namespace LibGeoDecomp {

/**
 * This class helps with setting up unstructed mesh codes. When
 * wrapping those in a regular mesh (see storage/containercell.h)
 * we'll need to decide on the container cells' extent in the physical
 * domain. Once that is decided upon, we can determine which
 * simulation entities go into which cell of the regular grid. Related
 * question: what's the logical size of the regular grid? (i.e.how
 * many cells is it made up from?)
 */
class GridSpacingCalculator {
public:
    template<typename COORD_LIST, typename NEIGHBORS_LIST, typename GRID_DIM_TYPE, template<int D> class COORD_TYPE, int DIM>
    static
    void determineGridDimensions(
        const COORD_LIST& points,
        const NEIGHBORS_LIST& neighbors,
        GRID_DIM_TYPE *logicalGridDim,
        COORD_TYPE<DIM> *cellDim)
    {
        COORD_TYPE<DIM> gridDim;
        determineMaximumDiameter(points, neighbors, &gridDim, cellDim);
        COORD_TYPE<DIM> rawLocigalGridDim = gridDim / *cellDim;

        for (int i = 0; i < DIM; ++i) {
            (*logicalGridDim)[i] = round(rawLocigalGridDim[i] + 0.5);
        }
    }

private:
    template<typename COORD_ARRAY, typename NEIGHBORS_ARRAY, typename COORD_TYPE>
    static
    void determineMaximumDiameter(
        const COORD_ARRAY& points,
        const NEIGHBORS_ARRAY& neighbors,
        COORD_TYPE *gridDim,
        COORD_TYPE *cellDim)
    {
        typedef typename NEIGHBORS_ARRAY::value_type CoordsArrayType;

        if (points.size() == 0) {
            return;
        }

        COORD_TYPE maxDelta;
        COORD_TYPE minCoord = points[0];
        COORD_TYPE maxCoord = points[0];

        for (std::size_t i = 0; i < points.size(); ++i) {
            for (typename CoordsArrayType::const_iterator j = neighbors[i].begin(); j != neighbors[i].end(); ++j) {
                COORD_TYPE delta = (points[i] - points[*j]).abs();
                maxDelta = maxDelta.max(delta);
            }

            minCoord = minCoord.min(points[i]);
            maxCoord = maxCoord.max(points[i]);
        }

        *gridDim = maxCoord - minCoord;
        *cellDim = maxDelta;
    }

};

}

#endif
