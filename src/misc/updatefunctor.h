#ifndef _libgeodecomp_misc_updatefunctor_h_
#define _libgeodecomp_misc_updatefunctor_h_

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
    static const int DIM = CELL::Topology::DIMENSIONS;
 
    template<typename GRID, typename UPDATE_POLICY>
    void operator()(
        const Streak<DIM>& streak,
        const GRID& gridOld,
        GRID *gridNew,
        unsigned nanoStep,
        APIs::Fixed,
        UPDATE_POLICY) 
    {
        const CELL *pointers[Stencil::VOLUME];
        LinePointerAssembly<Stencil>()(pointers, streak, gridOld);
        LinePointerUpdateFunctor<CELL>()(
            streak, gridOld.boundingBox(), pointers, &(*gridNew)[streak.origin], nanoStep);
    }

    template<typename GRID>
    void operator()(
        const Streak<DIM>& streak,
        const GRID& gridOld,
        GRID *gridNew,
        unsigned nanoStep,
        APIs::Base, 
        APIs::Base) 
    {
        VanillaUpdateFunctor<CELL>()(streak, gridOld, gridNew, nanoStep);
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
    static const int DIM = CELL::Topology::DIMENSIONS;

    template<typename GRID>
    void operator()(
        const Streak<DIM>& streak,
        const GRID& gridOld,
        GRID *gridNew,
        unsigned nanoStep) 
    {
        UpdateFunctorHelpers::Selector<CELL>()(
            streak, gridOld, gridNew, nanoStep, typename CELL::API(), typename CELL::API());
    }

};

}

#endif

