#ifndef LIBGEODECOMP_STORAGE_FIXEDNEIGHBORHOODUPDATEFUNCTOR_H
#define LIBGEODECOMP_STORAGE_FIXEDNEIGHBORHOODUPDATEFUNCTOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/future.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#endif

#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/storage/fixedneighborhood.h>
#include <libgeodecomp/storage/updatefunctormacros.h>
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

    template<typename ACCESSOR1, typename ACCESSOR2, typename CONCURRENCY_FUNCTOR, typename ANY_THREADED_UPDATE>
    void operator()(
        const Streak<DIM>& streak,
        ACCESSOR1& hoodOld,
        ACCESSOR2& hoodNew,
        const Coord<DIM> *offsetOld,
        const Coord<DIM> *offsetNew,
        const Coord<DIM> *dimensionsOld,
        const Coord<DIM> *dimensionsNew,
        const Coord<DIM> *topologicalDimensions,
        int nanoStep,
        const CONCURRENCY_FUNCTOR *concurrencySpec,
        const ANY_THREADED_UPDATE *modelThreadingSpec) const
    {
        Coord<DIM> normalizedOriginOld = streak.origin + *offsetOld;
        if (*topologicalDimensions != Coord<DIM>()) {
            normalizedOriginOld = TOPOLOGY::normalize(streak.origin + *offsetOld, *topologicalDimensions);
        }

#define LGD_FIXEDNEIGHBORHOODUPDATEFUNCTORHELPERS_INVOKE_PARAMS         \
            streak,                                                     \
            hoodOld,                                                    \
            hoodNew,                                                    \
            offsetOld,                                                  \
            offsetNew,                                                  \
            dimensionsOld,                                              \
            dimensionsNew,                                              \
            topologicalDimensions,                                      \
            nanoStep,                                                   \
            concurrencySpec,                                            \
            modelThreadingSpec                                          \

        if ((CUR_DIM == 2) && (HIGH == true)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                (normalizedOriginOld[CUR_DIM] == ((*dimensionsOld)[CUR_DIM] - 1))) {
                Invoke<CELL, CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, true>()(
                    LGD_FIXEDNEIGHBORHOODUPDATEFUNCTORHELPERS_INVOKE_PARAMS);
            } else {
                Invoke<CELL, CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, false>()(
                    LGD_FIXEDNEIGHBORHOODUPDATEFUNCTORHELPERS_INVOKE_PARAMS);
            }
        }

        if ((CUR_DIM == 2) && (HIGH == false)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                (normalizedOriginOld[CUR_DIM] == 0)) {
                Invoke<CELL, CUR_DIM - 1, true, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, true,  BOUNDARY_NORTH>()(
                    LGD_FIXEDNEIGHBORHOODUPDATEFUNCTORHELPERS_INVOKE_PARAMS);
            } else {
                Invoke<CELL, CUR_DIM - 1, true, TOPOLOGY, BOUNDARY_TOP, BOUNDARY_BOTTOM, false, BOUNDARY_NORTH>()(
                    LGD_FIXEDNEIGHBORHOODUPDATEFUNCTORHELPERS_INVOKE_PARAMS);
            }
        }

        if ((CUR_DIM == 1) && (HIGH == true)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                (normalizedOriginOld[CUR_DIM] == ((*dimensionsOld)[CUR_DIM] - 1))) {
                Invoke<CELL, CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, true,  BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    LGD_FIXEDNEIGHBORHOODUPDATEFUNCTORHELPERS_INVOKE_PARAMS);
            } else {
                Invoke<CELL, CUR_DIM, false, TOPOLOGY, BOUNDARY_TOP, false, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    LGD_FIXEDNEIGHBORHOODUPDATEFUNCTORHELPERS_INVOKE_PARAMS);
            }
        }

        if ((CUR_DIM == 1) && (HIGH == false)) {
            if (TOPOLOGY::template WrapsAxis<CUR_DIM>::VALUE &&
                (normalizedOriginOld[CUR_DIM] == 0)) {
                Invoke<CELL, CUR_DIM - 1, true, TOPOLOGY, true,  BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    LGD_FIXEDNEIGHBORHOODUPDATEFUNCTORHELPERS_INVOKE_PARAMS);
            } else {
                Invoke<CELL, CUR_DIM - 1, true, TOPOLOGY, false, BOUNDARY_BOTTOM, BOUNDARY_SOUTH, BOUNDARY_NORTH>()(
                    LGD_FIXEDNEIGHBORHOODUPDATEFUNCTORHELPERS_INVOKE_PARAMS);
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

    template<typename ACCESSOR1, typename ACCESSOR2, typename CONCURRENCY_FUNCTOR, typename ANY_THREADED_UPDATE>
    void operator()(
        const Streak<DIM>& streak,
        ACCESSOR1& hoodOld,
        ACCESSOR2& hoodNew,
        const Coord<DIM> *offsetOld,
        const Coord<DIM> *offsetNew,
        const Coord<DIM> *dimensionsOld,
        const Coord<DIM> *dimensionsNew,
        const Coord<DIM> *topologicalDimensions,
        int nanoStep,
        const CONCURRENCY_FUNCTOR *concurrencySpec,
        const ANY_THREADED_UPDATE *modelThreadingSpec) const
    {
        Coord<DIM> normalizedOriginOld = streak.origin + *offsetOld;
        Coord<DIM> normalizedOriginNew = streak.origin + *offsetNew;
        // don't normalize out of bounds accesses because SoA uses
        // padding for constant boundary conditions:
        if ((*topologicalDimensions != Coord<DIM>()) &&
            !(TOPOLOGY::isOutOfBounds(normalizedOriginOld, *topologicalDimensions))) {
            normalizedOriginOld = TOPOLOGY::normalize(normalizedOriginOld, *topologicalDimensions);
            normalizedOriginNew = TOPOLOGY::normalize(normalizedOriginNew, *topologicalDimensions);
        }

        // this copy is required to expand our potentially 1D or 2D
        // input coords to 3D, which is required by LibFlatArray.
        Coord<3> originOld;
        Coord<3> originNew;
        for (int i = 0; i < DIM; ++i) {
            originOld[i] = normalizedOriginOld[i];
            originNew[i] = normalizedOriginNew[i];
        }

        long indexOld = hoodOld.gen_index(
            static_cast<long>(originOld.x()),
            static_cast<long>(originOld.y()),
            static_cast<long>(originOld.z()));
        long indexNew = hoodNew.gen_index(
            static_cast<long>(originNew.x()),
            static_cast<long>(originNew.y()),
            static_cast<long>(originNew.z()));

        hoodOld.index() = indexOld;
        hoodNew.index() = indexNew;
        long indexEnd = hoodOld.index() + streak.length();
        long tempIndex;

        long boundaryWest;
        long boundaryEast;
        long boundaryTop;
        long boundaryBottom;
        long boundarySouth;
        long boundaryNorth;

        boundaryTop    = BOUNDARY_TOP    ?  (*dimensionsNew)[1] : 0;
        boundaryBottom = BOUNDARY_BOTTOM ? -(*dimensionsNew)[1] : 0;
        boundarySouth  = BOUNDARY_SOUTH  ?  (*dimensionsNew)[2] : 0;
        boundaryNorth  = BOUNDARY_NORTH  ? -(*dimensionsNew)[2] : 0;

        // special case: on left boundary
        if (TOPOLOGY::template WrapsAxis<0>::VALUE && (originOld.x() == 0)) {
            boundaryWest   = (*dimensionsNew)[0];
            boundaryEast   = 0;
            if (TOPOLOGY::template WrapsAxis<0>::VALUE &&
                ((originOld.x() + 1) == (*dimensionsNew).x())) {
                boundaryEast   = -(*dimensionsNew)[0];
            }

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

            long indexEnd = hoodOld.index() + 1;
            CELL::updateLineX(hoodLeft, indexEnd, hoodNew, nanoStep);
        }

        boundaryWest   = 0;
        boundaryEast   = 0;

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
            CELL::updateLineX(hood, indexEnd, hoodNew, nanoStep);
        }
    }
};

}

/**
 * This class takes over the (tedious) handling of boundary conditions
 * when using a FixedNeighborhood.
 */
template<typename CELL, typename CONCURRENCY_FUNCTOR, typename ANY_THREADED_UPDATE>
class FixedNeighborhoodUpdateFunctor
{
public:
    typedef typename APITraits::SelectTopology<CELL>::Value Topology;
    static const int DIM = Topology::DIM;

    FixedNeighborhoodUpdateFunctor(
        const Region<DIM> *region,
        const Coord<DIM> *offsetOld,
        const Coord<DIM> *offsetNew,
        const Coord<DIM> *dimensionsOld,
        const Coord<DIM> *dimensionsNew,
        const Coord<DIM> *topologicalDimensions,
        int nanoStep,
        const CONCURRENCY_FUNCTOR *concurrencySpec,
        const ANY_THREADED_UPDATE *modelThreadingSpec) :
        myRegion(region),
        offsetOld(offsetOld),
        offsetNew(offsetNew),
        dimensionsOld(dimensionsOld),
        dimensionsNew(dimensionsNew),
        topologicalDimensions(topologicalDimensions),
        nanoStep(nanoStep),
        myConcurrencySpec(concurrencySpec),
        myModelThreadingSpec(modelThreadingSpec)
    {}

    template<typename ACCESSOR1, typename ACCESSOR2>
    void operator()(
        ACCESSOR1& hoodOld,
        ACCESSOR2& hoodNew) const
    {
        const CONCURRENCY_FUNCTOR concurrencySpec = *myConcurrencySpec;
        const ANY_THREADED_UPDATE modelThreadingSpec = *myModelThreadingSpec;
        const Region<DIM>& region = *myRegion;

#define LGD_UPDATE_FUNCTOR_BODY                                         \
        ACCESSOR1 hoodOldCopy = hoodOld;                                \
        ACCESSOR2 hoodNewCopy = hoodNew;                                \
        FixedNeighborhoodUpdateFunctorHelpers::Invoke<CELL, DIM - 1, true, Topology>()( \
            *i,                                                         \
            hoodOldCopy,                                                \
            hoodNewCopy,                                                \
            offsetOld,                                                  \
            offsetNew,                                                  \
            dimensionsOld,                                              \
            dimensionsNew,                                              \
            topologicalDimensions,                                      \
            nanoStep,                                                   \
            &concurrencySpec,                                           \
            &modelThreadingSpec);

        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_1
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_2
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_3
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_4
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_5
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_6
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_7
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_8
#undef LGD_UPDATE_FUNCTOR_BODY
    }

private:
    const Region<DIM> *myRegion;
    const Coord<DIM> *offsetOld;
    const Coord<DIM> *offsetNew;
    const Coord<DIM> *dimensionsOld;
    const Coord<DIM> *dimensionsNew;
    const Coord<DIM> *topologicalDimensions;
    int nanoStep;
    const CONCURRENCY_FUNCTOR *myConcurrencySpec;
    const ANY_THREADED_UPDATE *myModelThreadingSpec;
};

}

#endif
