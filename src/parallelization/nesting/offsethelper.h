#ifndef LIBGEODECOMP_PARALLELIZATION_NESTING_OFFSETHELPER_H
#define LIBGEODECOMP_PARALLELIZATION_NESTING_OFFSETHELPER_H

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>

namespace LibGeoDecomp {

/**
 * This class tries to find an offset so that the bounding box of a
 * subdomain is minimized. This is expecially important if periodic
 * boundary conditions are being used since a node's subdomain may
 * have external ghost zones on the opposite side of the simulation
 * space. But even with constand boundary conditions certain domain
 * decomposition techniques (e.g. the Z-curve) may yield
 * non-contiguous subdomains.
 *
 * ownExpandedRegion is expected to contain not just a node's domain,
 * but also the adjacent external halo.
 *
 * Example: consider the 10x5 grid pictured below. We
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
 * The expanded marked by 7 "X" could be stored in such a 10x5 grid,
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
        const Region<DIM>& ownExpandedRegion,
        const CoordBox<DIM>& simulationArea)
    {
        CoordBox<DIM> ownBoundingBox = ownExpandedRegion.boundingBox();

        (*offset)[INDEX] = 0;
        if (TOPOLOGY::template WrapsAxis<INDEX>::VALUE) {
            int width = ownBoundingBox.dimensions[INDEX];
            if (width < simulationArea.dimensions[INDEX]) {
                (*offset)[INDEX] = ownBoundingBox.origin[INDEX];
            } else {
                (*offset)[INDEX] = 0;
            }

            (*dimensions)[INDEX] = (std::min)(width, simulationArea.dimensions[INDEX]);
        } else {
            (*offset)[INDEX] = (std::max)(0, ownBoundingBox.origin[INDEX]);

            int end = (std::min)(
                simulationArea.origin[INDEX] + simulationArea.dimensions[INDEX],
                ownBoundingBox.origin[INDEX] + ownBoundingBox.dimensions[INDEX]);

            (*dimensions)[INDEX] = end - (*offset)[INDEX];
        }

        OffsetHelper<INDEX - 1, DIM, TOPOLOGY>()(
            offset,
            dimensions,
            ownExpandedRegion,
            simulationArea);
    }
};

/**
 * See above. Terminates the recursive inheritance hierarchy.
 */
template<int DIM, typename TOPOLOGY>
class OffsetHelper<-1, DIM, TOPOLOGY>
{
public:
    void operator()(
        Coord<DIM> *offset,
        Coord<DIM> *dimensions,
        const Region<DIM>& ownExpandedRegion,
        const CoordBox<DIM>& simulationArea)
    {}
};

/**
 * See above. Shortcut for unstructured topologies.
 */
template<int INDEX, int DIM>
class OffsetHelper<INDEX, DIM, Topologies::Unstructured::Topology>
{
public:
    void operator()(
        Coord<DIM> *offset,
        Coord<DIM> *dimensions,
        const Region<DIM>& ownExpandedRegion,
        const CoordBox<DIM>& simulationArea)
    {
        *offset = ownExpandedRegion.boundingBox().origin;
        *dimensions = ownExpandedRegion.boundingBox().dimensions;
    }
};

}

#endif
