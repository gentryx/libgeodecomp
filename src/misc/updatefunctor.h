#ifndef LIBGEODECOMP_MISC_UPDATEFUNCTOR_H
#define LIBGEODECOMP_MISC_UPDATEFUNCTOR_H

#include <libgeodecomp/misc/cellapitraits.h>
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
    typedef typename CELL::Stencil Stencil;
    typedef typename CELL::API CellAPI;
    typedef typename CELL::Topology Topology;
    static const int DIM = CELL::Topology::DIM;

    class SoARegionUpdateHelper
    {
    public:
        SoARegionUpdateHelper(
            const Region<DIM>& region,
            const Coord<DIM>& targetOffset,
            const Coord<DIM>& gridOldOrigin,
            const Coord<DIM>& gridNewOrigin,
            const unsigned nanoStep) :
            region(region),
            targetOffset(targetOffset),
            gridOldOrigin(gridOldOrigin),
            gridNewOrigin(gridNewOrigin),
            nanoStep(nanoStep)
        {}

        template<typename CELL1, int MY_DIM_X1, int MY_DIM_Y1, int MY_DIM_Z1, int INDEX1,
                 typename CELL2, int MY_DIM_X2, int MY_DIM_Y2, int MY_DIM_Z2, int INDEX2>
        void operator()(const LibFlatArray::soa_accessor<CELL1, MY_DIM_X1, MY_DIM_Y1, MY_DIM_Z1, INDEX1>& hoodOld, int *indexOld,
                              LibFlatArray::soa_accessor<CELL2, MY_DIM_X2, MY_DIM_Y2, MY_DIM_Z2, INDEX2>& hoodNew, int *indexNew) const
        {
            for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
                Streak<DIM> relativeStreak(
                    i->origin - gridOldOrigin,
                    i->endX - gridOldOrigin.x());
                Coord<DIM> relativeTargetOrigin = relativeStreak.origin + targetOffset - gridNewOrigin;

                *indexOld =
                    relativeStreak.origin.z() * MY_DIM_X1 * MY_DIM_Y1 +
                    relativeStreak.origin.y() * MY_DIM_X1 +
                    relativeStreak.origin.x();
                Coord<DIM> end = relativeStreak.end();
                int indexEnd =
                    end.z() * MY_DIM_X1 * MY_DIM_Y1 +
                    end.y() * MY_DIM_X1 +
                    end.x();
                *indexNew =
                    relativeTargetOrigin.z() * MY_DIM_X2 * MY_DIM_Y2 +
                    relativeTargetOrigin.y() * MY_DIM_X2 +
                    relativeTargetOrigin.x();
                CELL::updateLineX(FixedNeighborhood<CELL, Topology, MY_DIM_X1, MY_DIM_Y1, MY_DIM_Z1, INDEX1>(hoodOld), indexOld, indexEnd, hoodNew, indexNew);
            }
        }

    private:
        const Region<DIM>& region;
        const Coord<DIM>& targetOffset;
        const Coord<DIM>& gridOldOrigin;
        const Coord<DIM>& gridNewOrigin;
        const unsigned nanoStep;
    };

    class SoAStreakUpdateHelper
    {
    public:
        SoAStreakUpdateHelper(
            const Streak<DIM>& streak,
            const Coord<DIM>& targetOrigin,
            const unsigned nanoStep) :
            streak(streak),
            targetOrigin(targetOrigin),
            nanoStep(nanoStep)
        {}

        template<typename CELL1, int MY_DIM_X1, int MY_DIM_Y1, int MY_DIM_Z1, int INDEX1,
                 typename CELL2, int MY_DIM_X2, int MY_DIM_Y2, int MY_DIM_Z2, int INDEX2>
        void operator()(const LibFlatArray::soa_accessor<CELL1, MY_DIM_X1, MY_DIM_Y1, MY_DIM_Z1, INDEX1>& hoodOld, int *indexOld,
                              LibFlatArray::soa_accessor<CELL2, MY_DIM_X2, MY_DIM_Y2, MY_DIM_Z2, INDEX2>& hoodNew, int *indexNew) const
        {
            *indexOld =
                streak.origin.z() * MY_DIM_X1 * MY_DIM_Y1 +
                streak.origin.y() * MY_DIM_X1 +
                streak.origin.x();
            Coord<DIM> end = streak.end();
            int indexEnd =
                end.z() * MY_DIM_X1 * MY_DIM_Y1 +
                end.y() * MY_DIM_X1 +
                end.x();
            *indexNew =
                targetOrigin.z() * MY_DIM_X2 * MY_DIM_Y2 +
                targetOrigin.y() * MY_DIM_X2 +
                targetOrigin.x();
            CELL::updateLineX(FixedNeighborhood<CELL, Topology, MY_DIM_X1, MY_DIM_Y1, MY_DIM_Z1, INDEX1>(hoodOld), indexOld, indexEnd, hoodNew, indexNew);
        }

    private:
        const Streak<DIM>& streak;
        const Coord<DIM>& targetOrigin;
        const unsigned nanoStep;
    };

    template<typename GRID1, typename GRID2>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        CellAPITraits::Fixed,
        CellAPITraits::Base,
        CellAPITraitsFixme::TrueType,
        CellAPITraitsFixme::TrueType)
    {
        Coord<DIM> gridOldOrigin = gridOld.boundingBox().origin;
        Coord<DIM> gridNewOrigin = gridNew->boundingBox().origin;
        gridOld.callback(
            gridNew,
            SoARegionUpdateHelper(region, targetOffset, gridOldOrigin, gridNewOrigin, nanoStep));
    }

    template<typename GRID1, typename GRID2>
    void operator()(
        const Streak<DIM>& streak,
        const Coord<DIM>& targetOrigin,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        CellAPITraits::Fixed,
        CellAPITraits::Base,
        CellAPITraitsFixme::TrueType,
        CellAPITraitsFixme::TrueType)
    {
        Coord<DIM> gridOldOrigin = gridOld.boundingBox().origin;
        Streak<DIM> relativeStreak(streak.origin - gridOldOrigin, streak.endX - gridOldOrigin.x());
        Coord<DIM> gridNewOrigin = gridNew->boundingBox().origin;
        Coord<DIM> relativeTargetOrigin = targetOrigin - gridNewOrigin;
        gridOld.callback(gridNew, SoAStreakUpdateHelper(relativeStreak, relativeTargetOrigin, nanoStep));
    }

    template<typename GRID1, typename GRID2, typename UPDATE_POLICY>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        CellAPITraits::Fixed,
        UPDATE_POLICY,
        CellAPITraitsFixme::FalseType,
        CellAPITraitsFixme::FalseType)
    {
        const CELL *pointers[Stencil::VOLUME];

        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            LinePointerAssembly<Stencil>()(pointers, *i, gridOld);
            LinePointerUpdateFunctor<CELL>()(
                *i, gridOld.boundingBox(), pointers, &(*gridNew)[i->origin + targetOffset], nanoStep);
        }
    }

    template<typename GRID1, typename GRID2, typename UPDATE_POLICY>
    void operator()(
        const Streak<DIM>& streak,
        const Coord<DIM>& targetOrigin,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        CellAPITraits::Fixed,
        UPDATE_POLICY,
        CellAPITraitsFixme::FalseType,
        CellAPITraitsFixme::FalseType)
    {
        const CELL *pointers[Stencil::VOLUME];
        LinePointerAssembly<Stencil>()(pointers, streak, gridOld);
        LinePointerUpdateFunctor<CELL>()(
            streak, gridOld.boundingBox(), pointers, &(*gridNew)[targetOrigin], nanoStep);
    }

    template<typename GRID1, typename GRID2>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        CellAPITraits::Base,
        CellAPITraits::Base,
        CellAPITraitsFixme::FalseType,
        CellAPITraitsFixme::FalseType)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            VanillaUpdateFunctor<CELL>()(*i, i->origin + targetOffset, gridOld, gridNew, nanoStep);
        }
    }

    template<typename GRID1, typename GRID2>
    void operator()(
        const Streak<DIM>& streak,
        const Coord<DIM>& targetOrigin,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        CellAPITraits::Base,
        CellAPITraits::Base,
        CellAPITraitsFixme::FalseType,
        CellAPITraitsFixme::FalseType)
    {
        VanillaUpdateFunctor<CELL>()(streak, targetOrigin, gridOld, gridNew, nanoStep);
    }
};

}

/**
 * is a wrapper which delegates the update of a line of cells to a
 * suitable implementation. The implementation may depend on the
 * properties of the CELL as well as the Simulator which is calling
 * the UpdateFunctor.
 */
template<typename CELL>
class UpdateFunctor
{
public:
    static const int DIM = CELL::Topology::DIM;

    template<typename GRID1, typename GRID2>
    void operator()(
        const Region<DIM>& region,
        const Coord<DIM>& targetOffset,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep)
    {
        UpdateFunctorHelpers::Selector<CELL>()(
            region, targetOffset, gridOld, gridNew, nanoStep,
            typename CELL::API(), typename CELL::API(),
            typename CellAPITraitsFixme::SelectGridType<CELL>::Value(),
            typename CellAPITraitsFixme::SelectUpdateLineX<CELL>::Value());
    }

    template<typename GRID1, typename GRID2>
    void operator()(
        const Streak<DIM>& sourceStreak,
        const Coord<DIM>& targetCoord,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep)
    {
        UpdateFunctorHelpers::Selector<CELL>()(
            sourceStreak, targetCoord, gridOld, gridNew, nanoStep,
            typename CELL::API(), typename CELL::API(),
            typename CellAPITraitsFixme::SelectGridType<CELL>::Value(),
            typename CellAPITraitsFixme::SelectUpdateLineX<CELL>::Value());
    }
};

}

#endif

