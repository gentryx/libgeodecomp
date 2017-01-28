#ifndef LIBGEODECOMP_GEOMETRY_UNSTRUCTUREDGRIDMESHER_H
#define LIBGEODECOMP_GEOMETRY_UNSTRUCTUREDGRIDMESHER_H

#include <cmath>

namespace LibGeoDecomp {

/**
 * This class helps with setting up unstructed mesh codes. When
 * wrapping those in a regular mesh (see storage/containercell.h)
 * we'll need to decide on the container cells' extent in the physical
 * domain.
 *
 * The mesher distinguishes real-valued positions (in the physical
 * space) and logical coordinates (within the regular grid).
 *
 * Once the spacing of the grid is decided upon, we can determine
 * which simulation entities go into which cell of the regular grid.
 * Related question: what's the logical size of the regular grid?
 * (i.e.how many cells is it made up from?)
 */
template<int DIM>
class UnstructuredGridMesher {
public:
    UnstructuredGridMesher()
    {}

    template<typename COORD_LIST, typename NEIGHBORS_LIST>
    UnstructuredGridMesher(
        const COORD_LIST& points,
        const NEIGHBORS_LIST& neighbors)
    {
        determineMaximumDiameter(points, neighbors);
        FloatCoord<DIM> rawLocigalGridDim = gridDim / cellDim;

        for (int i = 0; i < DIM; ++i) {
            // round up, even if the division leaves no remainder.
            // Example: for 3 elements, evenly spaced on a straight
            // line, the division would yield 2, but the last element
            // would be just out of the rightmost grid cell.
            logicalGridDim[i] = round(rawLocigalGridDim[i] + 0.5);
        }
    }

    const FloatCoord<DIM>& cellDimension() const
    {
        return cellDim;
    }

    const Coord<DIM>& logicalGridDimension() const
    {
        return logicalGridDim;
    }

    Coord<DIM> positionToLogicalCoord(const FloatCoord<DIM>& pos)
    {
        FloatCoord<DIM> rawCoord = (pos - minCoord) / cellDim;
        Coord<DIM> ret;
        for (int i = 0; i < DIM; ++i) {
            // round down:
            ret[i] = floor(rawCoord[i]);
        }

        return ret;
    }

private:
    Coord<DIM> logicalGridDim;
    FloatCoord<DIM> cellDim;
    FloatCoord<DIM> gridDim;
    FloatCoord<DIM> minCoord;
    FloatCoord<DIM> maxCoord;

    template<typename COORD_ARRAY, typename NEIGHBORS_ARRAY>
    void determineMaximumDiameter(
        const COORD_ARRAY& points,
        const NEIGHBORS_ARRAY& neighbors)
    {
        typedef typename NEIGHBORS_ARRAY::value_type CoordsArrayType;

        if (points.size() == 0) {
            return;
        }

        FloatCoord<DIM> maxDelta;
        minCoord = points[0];
        maxCoord = points[0];

        for (std::size_t i = 0; i < points.size(); ++i) {
            for (typename CoordsArrayType::const_iterator j = neighbors[i].begin(); j != neighbors[i].end(); ++j) {
                FloatCoord<DIM> delta = (points[i] - points[*j]).abs();
                maxDelta = maxDelta.max(delta);
            }

            minCoord = minCoord.min(points[i]);
            maxCoord = maxCoord.max(points[i]);
        }

        gridDim = maxCoord - minCoord;
        cellDim = maxDelta;
    }

};

}

#endif
