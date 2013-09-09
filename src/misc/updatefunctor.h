#ifndef LIBGEODECOMP_MISC_UPDATEFUNCTOR_H
#define LIBGEODECOMP_MISC_UPDATEFUNCTOR_H

#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/fixedneighborhood.h>
#include <libgeodecomp/misc/linepointerassembly.h>
#include <libgeodecomp/misc/linepointerupdatefunctor.h>
#include <libgeodecomp/misc/vanillaupdatefunctor.h>

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
            const Region<DIM>& region,
            const Coord<DIM>& sourceOffset,
            const Coord<DIM>& targetOffset,
            const unsigned nanoStep) :
            region(region),
            sourceOffset(sourceOffset),
            targetOffset(targetOffset),
            nanoStep(nanoStep)
        {}

        template<typename CELL1, int MY_DIM_X1, int MY_DIM_Y1, int MY_DIM_Z1, int INDEX1,
                 typename CELL2, int MY_DIM_X2, int MY_DIM_Y2, int MY_DIM_Z2, int INDEX2>
        void operator()(const LibFlatArray::soa_accessor<CELL1, MY_DIM_X1, MY_DIM_Y1, MY_DIM_Z1, INDEX1>& hoodOld, int *indexOld,
                              LibFlatArray::soa_accessor<CELL2, MY_DIM_X2, MY_DIM_Y2, MY_DIM_Z2, INDEX2>& hoodNew, int *indexNew) const
        {
            for (typename Region<DIM>::StreakIterator i = region.beginStreak();
                 i != region.endStreak();
                 ++i) {
                Streak<DIM> relativeSourceStreak(
                    i->origin + sourceOffset,
                    i->endX   + sourceOffset.x());
                Coord<DIM> relativeTargetOrigin = i->origin + targetOffset;

                *indexOld =
                    relativeSourceStreak.origin.z() * MY_DIM_X1 * MY_DIM_Y1 +
                    relativeSourceStreak.origin.y() * MY_DIM_X1 +
                    relativeSourceStreak.origin.x();
                Coord<DIM> end = relativeSourceStreak.end();
                int indexEnd =
                    end.z() * MY_DIM_X1 * MY_DIM_Y1 +
                    end.y() * MY_DIM_X1 +
                    end.x();
                *indexNew =
                    relativeTargetOrigin.z() * MY_DIM_X2 * MY_DIM_Y2 +
                    relativeTargetOrigin.y() * MY_DIM_X2 +
                    relativeTargetOrigin.x();
                CELL::updateLineX(
                    FixedNeighborhood<CELL, Topology, MY_DIM_X1, MY_DIM_Y1, MY_DIM_Z1, INDEX1>(hoodOld),
                    indexOld,
                    indexEnd,
                    hoodNew,
                    indexNew,
                    nanoStep);
            }
        }

    private:
        const Region<DIM>& region;
        const Coord<DIM>& sourceOffset;
        const Coord<DIM>& targetOffset;
        const unsigned nanoStep;
    };

    template<typename GRID1, typename GRID2>
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
        APITraits::TrueType)
    {
        Coord<DIM> gridOldOrigin = gridOld.boundingBox().origin;
        Coord<DIM> gridNewOrigin = gridNew->boundingBox().origin;

        Coord<DIM> realSourceOffset = sourceOffset - gridOldOrigin + gridOld.getEdgeRadii();
        Coord<DIM> realTargetOffset = targetOffset - gridNewOrigin + gridNew->getEdgeRadii();

        gridOld.callback(
            gridNew,
            SoARegionUpdateHelper(region, realSourceOffset, realTargetOffset, nanoStep));
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
        APITraits::TrueType,
        // SelectSoA
        APITraits::FalseType,
        // SelectUpdateLineX
        ANY_API)
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
        ANY_API)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            Streak<DIM> sourceStreak(i->origin + sourceOffset, i->endX + sourceOffset.x());
            Coord<DIM> targetOrigin = i->origin + targetOffset;
            VanillaUpdateFunctor<CELL>()(sourceStreak, targetOrigin, gridOld, gridNew, nanoStep);
        }
    }
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
            typename APITraits::SelectUpdateLineX<CELL>::Value());
    }
};

}

#endif

