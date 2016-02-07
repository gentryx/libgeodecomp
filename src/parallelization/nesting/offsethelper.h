#ifndef LIBGEODECOMP_PARALLELIZATION_NESTING_OFFSETHELPER_H
#define LIBGEODECOMP_PARALLELIZATION_NESTING_OFFSETHELPER_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>

namespace LibGeoDecomp {

/**
 * This class tries to find an offset so that the bounding box of a
 * subdomain in a simulation space with periodic boundary conditions
 * is minimized. Example: consider the 10x5 grid pictured below. We
 * assume that the model uses periodic boundary conditions.
 *
 *   0123456789
 *   ----------
 * 0|XX........
 * 1|XX........
 * 2|..........
 * 3|..........
 * 4|.......XXX
 *
 * The subdomain marked by 7 "X" could be stored in such a 10x5 grid,
 * but that would be wasteful. Rearragement of the offsets would allow
 * us the layout sketched out below:
 *
 *   78901
 *   -----
 * 4|XXX..
 * 0|...XX
 * 1|...XX
 *
 */
template<int INDEX, int DIM, typename TOPOLOGY>
class OffsetHelper
{
public:
    void operator()(
        Coord<DIM> *offset,
        Coord<DIM> *dimensions,
        const CoordBox<DIM>& ownBoundingBox,
        const CoordBox<DIM>& simulationArea,
        int ghostZoneWidth)
    {
        (*offset)[INDEX] = 0;
        if (TOPOLOGY::template WrapsAxis<INDEX>::VALUE) {
            int enlargedWidth =
                ownBoundingBox.dimensions[INDEX] + 2 * ghostZoneWidth;
            if (enlargedWidth < simulationArea.dimensions[INDEX]) {
                (*offset)[INDEX] =
                    ownBoundingBox.origin[INDEX] - ghostZoneWidth;
            } else {
                (*offset)[INDEX] = 0;
            }
            (*dimensions)[INDEX] =
                (std::min)(enlargedWidth, simulationArea.dimensions[INDEX]);
        } else {
            (*offset)[INDEX] =
                (std::max)(0, ownBoundingBox.origin[INDEX] - ghostZoneWidth);
            int end = (std::min)(simulationArea.origin[INDEX] +
                               simulationArea.dimensions[INDEX],
                               ownBoundingBox.origin[INDEX] +
                               ownBoundingBox.dimensions[INDEX] +
                               ghostZoneWidth);
            (*dimensions)[INDEX] = end - (*offset)[INDEX];
        }

        OffsetHelper<INDEX - 1, DIM, TOPOLOGY>()(
            offset,
            dimensions,
            ownBoundingBox,
            simulationArea,
            ghostZoneWidth);
    }
};

/**
 * see above
 */
template<int DIM, typename TOPOLOGY>
class OffsetHelper<-1, DIM, TOPOLOGY>
{
public:
    void operator()(
        Coord<DIM> *offset,
        Coord<DIM> *dimensions,
        const CoordBox<DIM>& ownBoundingBox,
        const CoordBox<DIM>& simulationArea,
        int ghostZoneWidth)
    {}
};

/**
 * see above
 */
template<int INDEX, int DIM>
class OffsetHelper<INDEX, DIM, Topologies::Unstructured::Topology>
{
public:
    void operator()(
        Coord<DIM> *offset,
        Coord<DIM> *dimensions,
        const CoordBox<DIM>& ownBoundingBox,
        const CoordBox<DIM>& simulationArea,
        int ghostZoneWidth)
    {
        *offset = ownBoundingBox.origin;
        *dimensions = simulationArea.dimensions;
    }
};

}

#endif
