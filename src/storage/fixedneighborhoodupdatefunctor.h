#ifndef LIBGEODECOMP_STORAGE_FIXEDNEIGHBORHOODUPDATEFUNCTOR_H
#define LIBGEODECOMP_STORAGE_FIXEDNEIGHBORHOODUPDATEFUNCTOR_H

#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/storage/fixedneighborhood.h>
#include <libgeodecomp/misc/apitraits.h>

namespace LibGeoDecomp {

namespace FixedNeighborhoodUpdateFunctorHelpers {

/**
 * Recursively bind template parameters for the different boundary conditions.
 */
template<
    typename CELL,
    int CUR_DIM,
    bool HIGH,
    typename TOPOLOGY,
    bool BOUNDARY_TOP = false,
    bool BOUNDARY_BOTTOM = false,
    bool BOUNDARY_SOUTH = false,
    bool BOUNDARY_NORTH = false>
class Invoke
{
public:
    static const int DIM = TOPOLOGY::DIM;

    template<typename ACCESSOR1, typename ACCESSOR2>
    void operator()(
        const Streak<DIM>& streak,
        ACCESSOR1& hoodOld,
        ACCESSOR2& hoodNew,
        const Coord<DIM> *offsetOld,
        const Coord<DIM> *offsetNew,
        const Coord<DIM> *dimensions,
        int nanoStep) const
    {
        if ((CUR_DIM == 2) && (HIGH == true)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                ((streak.origin[CUR_DIM] + (*offsetOld)[CUR_DIM]) == ((*dimensions)[CUR_DIM] - 1))) {
                Invoke<CELL, CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, true>()(
                    streak, hoodOld, hoodNew, offsetOld, offsetNew, dimensions, nanoStep);
            } else {
                Invoke<CELL, CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, false>()(
                    streak, hoodOld, hoodNew, offsetOld, offsetNew, dimensions, nanoStep);
            }
        }

        if ((CUR_DIM == 2) && (HIGH == false)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                ((streak.origin[CUR_DIM] + (*offsetOld)[CUR_DIM]) == 0)) {
                Invoke<CELL, CUR_DIM - 1, true, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, true,  BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, offsetOld, offsetNew, dimensions, nanoStep);
            } else {
                Invoke<CELL, CUR_DIM - 1, true, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, false, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, offsetOld, offsetNew, dimensions, nanoStep);
            }
        }

        if ((CUR_DIM == 1) && (HIGH == true)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                ((streak.origin[CUR_DIM] + (*offsetOld)[CUR_DIM]) == ((*dimensions)[CUR_DIM] - 1))) {
                Invoke<CELL, CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, true,  BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, offsetOld, offsetNew, dimensions, nanoStep);
            } else {
                Invoke<CELL, CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, false, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, offsetOld, offsetNew, dimensions, nanoStep);
            }
        }

        if ((CUR_DIM == 1) && (HIGH == false)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                ((streak.origin[CUR_DIM]+ (*offsetOld)[CUR_DIM]) == 0)) {
                Invoke<CELL, CUR_DIM - 1, true, TOPOLOGY, true,  BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, offsetOld, offsetNew, dimensions, nanoStep);
            } else {
                Invoke<CELL, CUR_DIM - 1, true, TOPOLOGY, false, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, offsetOld, offsetNew, dimensions, nanoStep);
            }
        }
    }
};

/**
 * See above
 */
template<
    typename CELL,
    bool HIGH,
    typename TOPOLOGY,
    bool BOUNDARY_TOP,
    bool BOUNDARY_BOTTOM,
    bool BOUNDARY_SOUTH,
    bool BOUNDARY_NORTH>
class Invoke<CELL, 0, HIGH, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>
{
public:
    static const int DIM = TOPOLOGY::DIM;

    template<typename ACCESSOR1, typename ACCESSOR2>
    void operator()(
        const Streak<DIM>& streak,
        ACCESSOR1& hoodOld,
        ACCESSOR2& hoodNew,
        const Coord<DIM> *offsetOld,
        const Coord<DIM> *offsetNew,
        const Coord<DIM> *dimensionsNew,
        int nanoStep) const

    {
        // this copy is required to blow our potentially 1D or 2D
        // input coords to 3D, which is required by LibFlatArray.
        Coord<3> originOld;
        Coord<3> originNew;
        for (int i = 0; i < DIM; ++i) {
            originOld[i] = streak.origin[i] + (*offsetOld)[i];
            originNew[i] = streak.origin[i] + (*offsetNew)[i];
        }
        long indexOld = hoodOld.gen_index(originOld.x(), originOld.y(), originOld.z());
        long indexNew = hoodNew.gen_index(originNew.x(), originNew.y(), originNew.z());
        hoodOld.index = indexOld;
        hoodNew.index = indexNew;
        long indexEnd = hoodOld.index + streak.length();
        long tempIndex;

        long boundaryWest;
        long boundaryEast;
        long boundaryTop;
        long boundaryBottom;
        long boundarySouth;
        long boundaryNorth;

        // special case: on left boundary
        if (TOPOLOGY::template WrapsAxis<0>::VALUE && (originOld.x() == 0)) {
            boundaryWest   = (*dimensionsNew)[0];
            boundaryEast   = 0;
            boundaryTop    = BOUNDARY_TOP    ?  (*dimensionsNew)[1] : 0;
            boundaryBottom = BOUNDARY_BOTTOM ? -(*dimensionsNew)[1] : 0;
            boundarySouth  = BOUNDARY_SOUTH  ?  (*dimensionsNew)[2] : 0;
            boundaryNorth  = BOUNDARY_NORTH  ? -(*dimensionsNew)[2] : 0;

            // fixme: handle special case for boundingBoxNew with width 1 (i.e.
            // we're on the western and eastern boundary
            // simultaneously)
            FixedNeighborhood<
                CELL,
                ACCESSOR1::DIM_X, ACCESSOR1::DIM_Y, ACCESSOR1::DIM_Z, 0> hoodLeft(
                    hoodOld,
                    tempIndex,
                    boundaryWest,
                    boundaryEast,
                    boundaryTop,
                    boundaryBottom,
                    boundarySouth,
                    boundaryNorth);

            long indexEnd = hoodOld.index + 1;
            CELL::updateLineX(hoodLeft, indexEnd, hoodNew, nanoStep);
        }

        boundaryWest   = 0;
        boundaryEast   = 0;
        boundaryTop    = BOUNDARY_TOP    ?  (*dimensionsNew)[1] : 0;
        boundaryBottom = BOUNDARY_BOTTOM ? -(*dimensionsNew)[1] : 0;
        boundarySouth  = BOUNDARY_SOUTH  ?  (*dimensionsNew)[2] : 0;
        boundaryNorth  = BOUNDARY_NORTH  ? -(*dimensionsNew)[2] : 0;

        FixedNeighborhood<
            CELL,
            ACCESSOR1::DIM_X, ACCESSOR1::DIM_Y, ACCESSOR1::DIM_Z, 0> hood(
                hoodOld,
                tempIndex,
                boundaryWest,
                boundaryEast,
                boundaryTop,
                boundaryBottom,
                boundarySouth,
                boundaryNorth);


        // other special case: right boundary
        if (TOPOLOGY::template WrapsAxis<0>::VALUE &&
            ((originOld.x() + streak.length()) == (*dimensionsNew).x())) {
            long indexEndRight = indexEnd - 1;
            CELL::updateLineX(hood, indexEndRight, hoodNew, nanoStep);

            boundaryWest   = 0;
            boundaryEast   = -(*dimensionsNew)[0];
            boundaryTop    = BOUNDARY_TOP    ?  (*dimensionsNew)[1] : 0;
            boundaryBottom = BOUNDARY_BOTTOM ? -(*dimensionsNew)[1] : 0;
            boundarySouth  = BOUNDARY_SOUTH  ?  (*dimensionsNew)[2] : 0;
            boundaryNorth  = BOUNDARY_NORTH  ? -(*dimensionsNew)[2] : 0;

            FixedNeighborhood<
                CELL,
                ACCESSOR1::DIM_X, ACCESSOR1::DIM_Y, ACCESSOR1::DIM_Z, 0> hoodRight(
                    hoodOld,
                    tempIndex,
                    boundaryWest,
                    boundaryEast,
                    boundaryTop,
                    boundaryBottom,
                    boundarySouth,
                    boundaryNorth);

            CELL::updateLineX(hoodRight, indexEnd, hoodNew, nanoStep);
        } else {
            CELL::updateLineX(hood,      indexEnd, hoodNew, nanoStep);
        }
    }
};

}

/**
 * This class takes over the (tedious) handling of boundary conditions
 * when using a FixedNeighborhood.
 */
template<typename CELL>
class FixedNeighborhoodUpdateFunctor
{
public:
    typedef typename APITraits::SelectTopology<CELL>::Value Topology;
    static const int DIM = Topology::DIM;

    FixedNeighborhoodUpdateFunctor(
        const Region<DIM> *region,
        const Coord<DIM> *offsetOld,
        const Coord<DIM> *offsetNew,
        const Coord<DIM> *dimensionsNew,
        int nanoStep) :
        region(region),
        offsetOld(offsetOld),
        offsetNew(offsetNew),
        dimensionsNew(dimensionsNew),
        nanoStep(nanoStep)
    {}

    template<typename ACCESSOR1, typename ACCESSOR2>
    void operator()(
        ACCESSOR1& hoodOld,
        ACCESSOR2& hoodNew) const
    {
        for (typename Region<DIM>::StreakIterator i = region->beginStreak(); i != region->endStreak(); ++i) {
            FixedNeighborhoodUpdateFunctorHelpers::Invoke<CELL, DIM - 1, true, Topology>()(
                *i,
                hoodOld,
                hoodNew,
                offsetOld,
                offsetNew,
                dimensionsNew,
                nanoStep);
        }
    }

private:
    const Region<DIM> *region;
    const Coord<DIM> *offsetOld;
    const Coord<DIM> *offsetNew;
    const Coord<DIM> *dimensionsNew;
    int nanoStep;
};

}

#endif
