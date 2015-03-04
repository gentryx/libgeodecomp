#ifndef LIBGEODECOMP_STORAGE_FIXEDNEIGHBORHOODUPDATEFUNCTOR_H
#define LIBGEODECOMP_STORAGE_FIXEDNEIGHBORHOODUPDATEFUNCTOR_H

#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/storage/fixedneighborhood.h>
#include <libgeodecomp/misc/apitraits.h>

namespace LibGeoDecomp {

namespace FixedNeighborhoodUpdateFunctorHelpers {

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
        const CoordBox<DIM>& box,
        int nanoStep) const
    {
        if ((CUR_DIM == 2) && (HIGH == true)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                (streak.origin[CUR_DIM] == (box.origin[CUR_DIM] + box.dimensions[CUR_DIM] - 1))) {
                Invoke<CELL, CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, true>()(
                    streak, hoodOld, hoodNew, box, nanoStep);
            } else {
                Invoke<CELL, CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, false>()(
                    streak, hoodOld, hoodNew, box, nanoStep);
            }
        }

        if ((CUR_DIM == 2) && (HIGH == false)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                (streak.origin[CUR_DIM] == box.origin[CUR_DIM])) {
                Invoke<CELL, CUR_DIM - 1, true, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, true,  BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, box, nanoStep);
            } else {
                Invoke<CELL, CUR_DIM - 1, true, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, false, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, box, nanoStep);
            }
        }

        if ((CUR_DIM == 1) && (HIGH == true)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                (streak.origin[CUR_DIM] == (box.origin[CUR_DIM] + box.dimensions[CUR_DIM] - 1))) {
                Invoke<CELL, CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, true,  BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, box, nanoStep);
            } else {
                Invoke<CELL, CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, false, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, box, nanoStep);
            }
        }

        if ((CUR_DIM == 1) && (HIGH == false)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                (streak.origin[CUR_DIM] == box.origin[CUR_DIM])) {
                Invoke<CELL, CUR_DIM - 1, true, TOPOLOGY, true,  BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, box, nanoStep);
            } else {
                Invoke<CELL, CUR_DIM - 1, true, TOPOLOGY, false, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, box, nanoStep);
            }
        }
    }
};

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
        const CoordBox<DIM>& box,
        int nanoStep) const

    {
        // fixme: fix x/y/z offsets
        FixedNeighborhood<
            CELL,
            ACCESSOR1::DIM_X, ACCESSOR1::DIM_Y, ACCESSOR1::DIM_Z, 0,
            -1, 1,
            -1, 1,
            -1, 1> hood(hoodOld);

        // this copy is required to blow our potentially 1D or 2D
        // input coords to 3D, which is required by LibFlatArray.
        Coord<3> origin;
        for (int i = 0; i < DIM; ++i) {
            origin[i] = streak.origin[i];
        }
        long index = hoodOld.gen_index(origin.x(), origin.y(), origin.z());
        hoodOld.index = index;
        hoodNew.index = index;
        long indexEnd = hoodOld.index + streak.length();

        CELL::updateLineX(hoodOld, indexEnd, hoodNew, nanoStep);
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
        const CoordBox<DIM>& boundingBox,
        int nanoStep) :
        region(region),
        boundingBox(boundingBox),
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
                boundingBox,
                nanoStep);
        }
    }

private:
    const Region<DIM> *region;
    CoordBox<DIM> boundingBox;
    int nanoStep;
};

}

#endif
