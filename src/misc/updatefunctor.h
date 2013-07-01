#ifndef LIBGEODECOMP_MISC_UPDATEFUNCTOR_H
#define LIBGEODECOMP_MISC_UPDATEFUNCTOR_H

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
    static const int DIM = CELL::Topology::DIM;

    template<typename GRID1, typename GRID2, typename UPDATE_POLICY>
    void operator()(
        const Streak<DIM>& streak,
        const Coord<DIM>& targetOrigin,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        CellAPITraits::Fixed,
        UPDATE_POLICY)
    {
        const CELL *pointers[Stencil::VOLUME];
        LinePointerAssembly<Stencil>()(pointers, streak, gridOld);
        LinePointerUpdateFunctor<CELL>()(
            streak, gridOld.boundingBox(), pointers, &(*gridNew)[targetOrigin], nanoStep);
    }

    template<typename GRID1, typename GRID2>
    void operator()(
        const Streak<DIM>& streak,
        const Coord<DIM>& targetOrigin,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        CellAPITraits::Base,
        CellAPITraits::Base)
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
        const Streak<DIM>& sourceStreak,
        const Coord<DIM>& targetCoord,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep)
    {
        UpdateFunctorHelpers::Selector<CELL>()(
            sourceStreak, targetCoord, gridOld, gridNew, nanoStep, typename CELL::API(), typename CELL::API());
    }

};

}

#endif

