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
        OffsetHelper<INDEX - 1, DIM, TOPOLOGY>()(offset, dimensions, ownExpandedRegion, simulationArea);

        CoordBox<DIM> ownBoundingBox = ownExpandedRegion.boundingBox();
        // no point in wrapping over edges if topology doesn't permit
        // this or our bounding box already tells use that the region
        // is too small anyway:
        if ((!TOPOLOGY::template WrapsAxis<INDEX>::VALUE) ||
            (ownBoundingBox.dimensions[INDEX] < (simulationArea.dimensions[INDEX] / 2))) {

            (*offset)[INDEX] = ownBoundingBox.origin[INDEX];
            (*dimensions)[INDEX] = ownBoundingBox.dimensions[INDEX];

            return;
        }

        // look for gaps which can be exploited by wrapping around the edge of the grid:
        Region<1> gapStorage;
        int oppositeSide = ownBoundingBox.origin[INDEX] + ownBoundingBox.dimensions[INDEX];

        for (int i = ownBoundingBox.origin[INDEX]; i < oppositeSide; ++i) {
            CoordBox<DIM> cutBox = ownBoundingBox;
            cutBox.origin[INDEX] = i;
            cutBox.dimensions[INDEX] = 1;

            Region<DIM> cutRegion;
            cutRegion << cutBox;

            if ((ownExpandedRegion & cutRegion).empty()) {
                gapStorage << Coord<1>(i);
            }
        }

        Streak<1> widestGap;
        for (Region<1>::StreakIterator i = gapStorage.beginStreak(); i != gapStorage.endStreak(); ++i) {
            if (i->length() > widestGap.length()) {
                widestGap = *i;
            }
        }

        int wrappedWidth = simulationArea.dimensions[INDEX] - widestGap.length();

        if (wrappedWidth < ownBoundingBox.dimensions[INDEX]) {
            (*offset)[INDEX] = widestGap.endX;
            (*dimensions)[INDEX] = wrappedWidth;
        }
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
