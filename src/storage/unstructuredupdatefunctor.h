#ifndef _UNSTRUCTUREDUPDATEFUNCTOR_H_
#define _UNSTRUCTUREDUPDATEFUNCTOR_H_

#include <libgeodecomp/storage/unstructuredneighborhood.h>

namespace LibGeoDecomp {

template<typename CELL>
class UnstrutucturedUpdateFunctor
{
public:
    typedef typename APITraits::SelectTopology<CELL>::Value Topology;
    static const int DIM = Topology::DIM;

    template<typename GRID1, typename GRID2>
    void operator()(
        const Streak<DIM>& streak,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep)
    {
        UnstructuredNeighborhood<CELL> hoodOld(const_cast<GRID1&>(gridOld), streak.origin.x());
        UnstructuredNeighborhood<CELL> hoodNew(*gridNew, 0);

        CELL::updateLineX(hoodNew, streak.endX, hoodOld, nanoStep);
    }
};

}

#endif /* _UNSTRUCTUREDUPDATEFUNCTOR_H_ */
