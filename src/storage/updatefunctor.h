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

    template<typename GRID1, typename GRID2, typename ANY_TOPOLOGY>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& sourceOffset,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        // SelectFixedCoordsOnlyUpdate
        APITraits::TrueType,
        // SelectSoA
        APITraits::TrueType,
        // SelectUpdateLineX
        APITraits::TrueType,
        // SelectTopology,
        ANY_TOPOLOGY)
    {
        Coord<DIM> gridOldOrigin = gridOld.boundingBox().origin;
        Coord<DIM> gridNewOrigin = gridNew->boundingBox().origin;
        Coord<DIM> gridNewDimensions = gridNew->boundingBox().dimensions;

        Coord<DIM> realSourceOffset = sourceOffset - gridOldOrigin + gridOld.getEdgeRadii();
        Coord<DIM> realTargetOffset = targetOffset - gridNewOrigin + gridNew->getEdgeRadii();

        gridOld.callback(
            gridNew,
            SoARegionUpdateHelper(&region, &realSourceOffset, &realTargetOffset, &gridNewDimensions, nanoStep));
    }

    template<typename GRID1, typename GRID2, typename ANY_API, typename ANY_TOPOLOGY>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& sourceOffset,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        // SelectFixedCoordsOnlyUpdate
        APITraits::TrueType,
        // SelectSoA
        APITraits::FalseType,
        // SelectUpdateLineX
        ANY_API,
        // SelectTopology
        ANY_TOPOLOGY)
    {
        const CELL *pointers[Stencil::VOLUME];

        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            Streak<DIM> streak(i->origin + sourceOffset, i->endX + sourceOffset.x());
            Coord<DIM> realTargetCoord = i->origin + targetOffset;

            LinePointerAssembly<Stencil>()(pointers, *i, gridOld);
            LinePointerUpdateFunctor<CELL>()(
                streak, gridOld.boundingBox(), pointers, &(*gridNew)[realTargetCoord], nanoStep);
        }
    }

    template<typename GRID1, typename GRID2, typename ANY_API>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& sourceOffset,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        // SelectFixedCoordsOnlyUpdate
        APITraits::FalseType,
        // SelectSoA
        APITraits::FalseType,
        // SelectUpdateLineX
        ANY_API,
        // SelectTopology
        TopologiesHelpers::Topology<DIM, false, false, false>)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            Streak<DIM> sourceStreak(i->origin + sourceOffset, i->endX + sourceOffset.x());
            Coord<DIM> targetOrigin = i->origin + targetOffset;
            VanillaUpdateFunctor<CELL>()(sourceStreak, targetOrigin, gridOld, gridNew, nanoStep);
        }
    }

    template<typename GRID1, typename GRID2, typename ANY_API>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& sourceOffset,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        // SelectFixedCoordsOnlyUpdate
        APITraits::FalseType,
        // SelectSoA
        APITraits::FalseType,
        // SelectUpdateLineX
        ANY_API,
        // SelectTopology
        TopologiesHelpers::Topology<DIM, true, true, true>)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            Streak<DIM> sourceStreak(i->origin + sourceOffset, i->endX + sourceOffset.x());
            Coord<DIM> targetOrigin = i->origin + targetOffset;
            VanillaUpdateFunctor<CELL>()(sourceStreak, targetOrigin, gridOld, gridNew, nanoStep);
        }
    }

#ifdef LIBGEODECOMP_WITH_CPP14
    template<typename GRID1, typename GRID2, typename ANY_API, typename ANY_GRID_TYPE>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& sourceOffset,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        // SelectFixedCoordsOnlyUpdate
        APITraits::FalseType,
        // SelectSoA
        ANY_GRID_TYPE,
        // SelectUpdateLineX
        ANY_API,
        // SelectTopology
        TopologiesHelpers::UnstructuredTopology)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            Streak<DIM> sourceStreak(i->origin + sourceOffset, i->endX + sourceOffset.x());
            UnstructuredUpdateFunctor<CELL>()(sourceStreak, gridOld, gridNew, nanoStep);
        }
    }
#endif
};

}

/**
 * UpdateFunctor a wrapper which delegates the update of a set of
 * cells to a suitable implementation. The implementation may depend
 * on the properties of the CELL as well as the datastructure which
 * holds the grid data.
 */
template<typename CELL>
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
        unsigned nanoStep)
    {
        UpdateFunctorHelpers::Selector<CELL>()(
            region, sourceOffset, targetOffset, gridOld, gridNew, nanoStep,
            typename APITraits::SelectFixedCoordsOnlyUpdate<CELL>::Value(),
            typename APITraits::SelectSoA<CELL>::Value(),
            typename APITraits::SelectUpdateLineX<CELL>::Value(),
            typename APITraits::SelectTopology<CELL>::Value());
    }
};

}

#endif

