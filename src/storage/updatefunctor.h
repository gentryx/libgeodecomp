#ifndef LIBGEODECOMP_STORAGE_UPDATEFUNCTOR_H
#define LIBGEODECOMP_STORAGE_UPDATEFUNCTOR_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/fixedneighborhoodupdatefunctor.h>
#include <libgeodecomp/storage/linepointerassembly.h>
#include <libgeodecomp/storage/linepointerupdatefunctor.h>
#include <libgeodecomp/storage/vanillaupdatefunctor.h>
#include <libgeodecomp/storage/unstructuredupdatefunctor.h>

namespace LibGeoDecomp {

namespace UpdateFunctorHelpers {

/**
 * Switches between different implementations of the UpdateFunctor,
 * depending on the properties of the model/grid. Not to be confused
 * with src/storage/selector.h.
 */
template<typename CELL>
class Selector
{
public:
    typedef typename APITraits::SelectStencil<CELL>::Value Stencil;
    typedef typename APITraits::SelectTopology<CELL>::Value Topology;

    static const int DIM = Topology::DIM;

    class SoARegionUpdateHelper
    {
    public:
        SoARegionUpdateHelper(
            const Region<DIM> *region,
            const Coord<DIM> *offsetOld,
            const Coord<DIM> *offsetNew,
            const Coord<DIM> *dimensionsNew,
            const unsigned nanoStep) :
            region(region),
            offsetOld(offsetOld),
            offsetNew(offsetNew),
            dimensionsNew(dimensionsNew),
            nanoStep(nanoStep)
        {}

        template<
            typename CELL1, long MY_DIM_X1, long MY_DIM_Y1, long MY_DIM_Z1, long INDEX1,
            typename CELL2, long MY_DIM_X2, long MY_DIM_Y2, long MY_DIM_Z2, long INDEX2>
        void operator()(
            LibFlatArray::soa_accessor<CELL1, MY_DIM_X1, MY_DIM_Y1, MY_DIM_Z1, INDEX1>& hoodOld,
            LibFlatArray::soa_accessor<CELL2, MY_DIM_X2, MY_DIM_Y2, MY_DIM_Z2, INDEX2>& hoodNew) const
        {
            FixedNeighborhoodUpdateFunctor<CELL>(region, offsetOld, offsetNew, dimensionsNew, nanoStep)(hoodOld, hoodNew);
        }

    private:
        const Region<DIM> *region;
        const Coord<DIM> *offsetOld;
        const Coord<DIM> *offsetNew;
        const Coord<DIM> *dimensionsNew;
        const unsigned nanoStep;
    };

    template<typename GRID1, typename GRID2, typename CONCURRENCY_FUNCTOR, typename ANY_TOPOLOGY, typename ANY_THREADED_UPDATE>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& sourceOffset,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        const CONCURRENCY_FUNCTOR& concurrencySpec,
        // SelectFixedCoordsOnlyUpdate
        APITraits::TrueType,
        // SelectSoA
        APITraits::TrueType,
        // SelectUpdateLineX
        APITraits::TrueType,
        // SelectTopology,
        ANY_TOPOLOGY,
        // SelectThreadedUpdate,
        ANY_THREADED_UPDATE)
    {
        Coord<DIM> gridOldOrigin = gridOld.boundingBox().origin;
        Coord<DIM> gridNewOrigin = gridNew->boundingBox().origin;
        Coord<DIM> gridNewDimensions = gridNew->boundingBox().dimensions;

        Coord<DIM> realSourceOffset = sourceOffset - gridOldOrigin + gridOld.getEdgeRadii();
        Coord<DIM> realTargetOffset = targetOffset - gridNewOrigin + gridNew->getEdgeRadii();

        gridOld.callback(
            gridNew,
            SoARegionUpdateHelper(
                &region,
                &realSourceOffset,
                &realTargetOffset,
                &gridNewDimensions,
                nanoStep));
    }

    template<typename GRID1, typename GRID2, typename CONCURRENCY_FUNCTOR, typename ANY_API, typename ANY_TOPOLOGY, typename ANY_THREADED_UPDATE>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& sourceOffset,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        const CONCURRENCY_FUNCTOR& concurrencySpec,
        // SelectFixedCoordsOnlyUpdate
        APITraits::TrueType,
        // SelectSoA
        APITraits::FalseType,
        // SelectUpdateLineX
        ANY_API,
        // SelectTopology
        ANY_TOPOLOGY,
        // SelectThreadedUpdate,
        ANY_THREADED_UPDATE modelThreadingSpec)
    {
        const CELL *pointers[Stencil::VOLUME];

#ifdef LIBGEODECOMP_WITH_THREADS
        if (concurrencySpec.enableOpenMP() && !modelThreadingSpec.hasOpenMP()) {
#pragma omp parallel for schedule(static)
            for (std::size_t c = 0; c < region.numPlanes(); ++c) {
                typename Region<DIM>::StreakIterator e = region.planeStreakIterator(c + 1);
                for (typename Region<DIM>::StreakIterator i = region.planeStreakIterator(c + 0); i != e; ++i) {
                    Streak<DIM> streak(i->origin + sourceOffset, i->endX + sourceOffset.x());
                    Coord<DIM> realTargetCoord = i->origin + targetOffset;

                    LinePointerAssembly<Stencil>()(pointers, *i, gridOld);
                    LinePointerUpdateFunctor<CELL>()(
                        streak, gridOld.boundingBox(), pointers, &(*gridNew)[realTargetCoord], nanoStep);
                }
            }
            return;
        }
#endif

#ifdef LIBGEODECOMP_WITH_HPX
        if (concurrencySpec.enableHPX() && !modelThreadingSpec.hasHPX()) {
            for (std::size_t c = 0; c < region.numPlanes(); ++c) {
                typename Region<DIM>::StreakIterator e = region.planeStreakIterator(c + 1);
                for (typename Region<DIM>::StreakIterator i = region.planeStreakIterator(c + 0); i != e; ++i) {
                    Streak<DIM> streak(i->origin + sourceOffset, i->endX + sourceOffset.x());
                    Coord<DIM> realTargetCoord = i->origin + targetOffset;

                    LinePointerAssembly<Stencil>()(pointers, *i, gridOld);
                    LinePointerUpdateFunctor<CELL>()(
                        streak, gridOld.boundingBox(), pointers, &(*gridNew)[realTargetCoord], nanoStep);
                }
            }
            return;
        }
#endif

        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            Streak<DIM> streak(i->origin + sourceOffset, i->endX + sourceOffset.x());
            Coord<DIM> realTargetCoord = i->origin + targetOffset;

            LinePointerAssembly<Stencil>()(pointers, *i, gridOld);
            LinePointerUpdateFunctor<CELL>()(
                streak, gridOld.boundingBox(), pointers, &(*gridNew)[realTargetCoord], nanoStep);
        }
    }

    template<typename GRID1, typename GRID2, typename CONCURRENCY_FUNCTOR, typename ANY_API, typename ANY_THREADED_UPDATE>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& sourceOffset,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        const CONCURRENCY_FUNCTOR& concurrencySpec,
        // SelectFixedCoordsOnlyUpdate
        APITraits::FalseType,
        // SelectSoA
        APITraits::FalseType,
        // SelectUpdateLineX
        ANY_API,
        // SelectTopology
        TopologiesHelpers::Topology<DIM, false, false, false>,
        // SelectThreadedUpdate,
        ANY_THREADED_UPDATE)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            Streak<DIM> sourceStreak(i->origin + sourceOffset, i->endX + sourceOffset.x());
            Coord<DIM> targetOrigin = i->origin + targetOffset;
            VanillaUpdateFunctor<CELL>()(sourceStreak, targetOrigin, gridOld, gridNew, nanoStep);
        }
    }

    template<typename GRID1, typename GRID2, typename CONCURRENCY_FUNCTOR, typename ANY_API, typename ANY_THREADED_UPDATE>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& sourceOffset,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        const CONCURRENCY_FUNCTOR& concurrencySpec,
        // SelectFixedCoordsOnlyUpdate
        APITraits::FalseType,
        // SelectSoA
        APITraits::FalseType,
        // SelectUpdateLineX
        ANY_API,
        // SelectTopology
        TopologiesHelpers::Topology<DIM, true, true, true>,
        // SelectThreadedUpdate,
        ANY_THREADED_UPDATE)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            Streak<DIM> sourceStreak(i->origin + sourceOffset, i->endX + sourceOffset.x());
            Coord<DIM> targetOrigin = i->origin + targetOffset;
            VanillaUpdateFunctor<CELL>()(sourceStreak, targetOrigin, gridOld, gridNew, nanoStep);
        }
    }

#ifdef LIBGEODECOMP_WITH_CPP14
    template<typename GRID1, typename GRID2, typename CONCURRENCY_FUNCTOR, typename ANY_API, typename ANY_GRID_TYPE, typename ANY_THREADED_UPDATE>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& sourceOffset,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        const CONCURRENCY_FUNCTOR& concurrencySpec,
        // SelectFixedCoordsOnlyUpdate
        APITraits::FalseType,
        // SelectSoA
        ANY_GRID_TYPE,
        // SelectUpdateLineX
        ANY_API,
        // SelectTopology
        TopologiesHelpers::UnstructuredTopology,
        // SelectThreadedUpdate,
        ANY_THREADED_UPDATE)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            Streak<DIM> sourceStreak(i->origin + sourceOffset, i->endX + sourceOffset.x());
            UnstructuredUpdateFunctor<CELL>()(sourceStreak, gridOld, gridNew, nanoStep);
        }
    }
#endif
};

/**
 * The default CONCURRENCY_FUNCTOR for UpdateFunctor: won't request
 * threading and won't execute any sideband actions.
 */
class ConcurrencyNoP
{
public:
    bool enableOpenMP() const
    {
        return false;
    }
};

class ConcurrencyEnableOpenMP
{
public:
    bool enableOpenMP() const
    {
        return true;
    }
};

}

/**
 * UpdateFunctor is a wrapper which delegates the update of a set of
 * cells to a suitable implementation. The implementation may depend
 * on the properties of the CELL as well as the datastructure which
 * holds the grid data.
 *
 * The CONCURRENCY_FUNCTOR can be used to control threading and to
 * execute sideband functions (e.g. MPI pacing).
 */
template<typename CELL, typename CONCURRENCY_FUNCTOR = UpdateFunctorHelpers::ConcurrencyNoP>
class UpdateFunctor
{
public:
    typedef typename APITraits::SelectTopology<CELL>::Value Topology;

    static const int DIM = Topology::DIM;

    template<typename GRID1, typename GRID2>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& sourceOffset,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        const CONCURRENCY_FUNCTOR& concurrencySpec = UpdateFunctorHelpers::ConcurrencyNoP())
    {
        UpdateFunctorHelpers::Selector<CELL>()(
            region, sourceOffset, targetOffset, gridOld, gridNew, nanoStep, concurrencySpec,
            typename APITraits::SelectFixedCoordsOnlyUpdate<CELL>::Value(),
            typename APITraits::SelectSoA<CELL>::Value(),
            typename APITraits::SelectUpdateLineX<CELL>::Value(),
            typename APITraits::SelectTopology<CELL>::Value(),
            typename APITraits::SelectThreadedUpdate<CELL>::Value());
    }
};

}

#endif

