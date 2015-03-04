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
        const CoordBox<DIM>& box) const
    {
        if ((CUR_DIM == 2) && (HIGH == true)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                (streak.origin[CUR_DIM] == (box.origin[CUR_DIM] + box.dimensions[CUR_DIM] - 1))) {
                Invoke<CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, true>()(
                    streak, hoodOld, hoodNew, box);
            } else {
                Invoke<CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, false>()(
                    streak, hoodOld, hoodNew, box);
            }
        }

        if ((CUR_DIM == 2) && (HIGH == false)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                (streak.origin[CUR_DIM] == box.origin[CUR_DIM])) {
                Invoke<CUR_DIM - 1, true, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, true,  BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, box);
            } else {
                Invoke<CUR_DIM - 1, true, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, false, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, box);
            }
        }

        if ((CUR_DIM == 1) && (HIGH == true)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                (streak.origin[CUR_DIM] == (box.origin[CUR_DIM] + box.dimensions[CUR_DIM] - 1))) {
                Invoke<CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, true,  BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, box);
            } else {
                Invoke<CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, false, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, box);
            }
        }

        if ((CUR_DIM == 1) && (HIGH == false)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                (streak.origin[CUR_DIM] == box.origin[CUR_DIM])) {
                Invoke<CUR_DIM - 1, true, TOPOLOGY, true,  BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, box);
            } else {
                Invoke<CUR_DIM - 1, true, TOPOLOGY, false, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    streak, hoodOld, hoodNew, box);
            }
        }
    }
};

template<
    bool HIGH,
    typename TOPOLOGY,
    bool BOUNDARY_TOP,
    bool BOUNDARY_BOTTOM,
    bool BOUNDARY_SOUTH,
    bool BOUNDARY_NORTH>
class Invoke<0, HIGH, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>
{
public:
    static const int DIM = TOPOLOGY::DIM;

    template<typename ACCESSOR1, typename ACCESSOR2>
    void operator()(
        const Streak<DIM>& streak,
        ACCESSOR1& hoodOld,
        ACCESSOR2& hoodNew,
        const CoordBox<DIM>& box) const

    {
        std::cout << "foobar\n";
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
        const CoordBox<DIM>& boundingBox) :
        region(region),
        boundingBox(boundingBox)
    {}

    template<typename ACCESSOR1, typename ACCESSOR2>
    void operator()(
        ACCESSOR1& hoodOld,
        ACCESSOR2& hoodNew) const
    {
        for (typename Region<DIM>::StreakIterator i = region->beginStreak(); i != region->endStreak(); ++i) {
            FixedNeighborhoodUpdateFunctorHelpers::Invoke<DIM - 1, true, Topology>()(
                *i,
                hoodOld,
                hoodNew,
                boundingBox);
        }
    }

private:
    const Region<DIM> *region;
    CoordBox<DIM> boundingBox;
};

}

#endif
