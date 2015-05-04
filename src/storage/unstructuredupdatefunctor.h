#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDUPDATEFUNCTOR_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDUPDATEFUNCTOR_H

#include <libgeodecomp/misc/apitraits.h>
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
        typedef typename APITraits::SelectUpdateLineX<CELL>::Value UpdateLineXFlag;

        UnstructuredNeighborhood<CELL> hoodOld(const_cast<GRID1&>(gridOld), streak.origin.x());
        UnstructuredNeighborhood<CELL> hoodNew(*gridNew, 0);

        // switch between updateLineX() and update()
        updateWrapper(hoodNew, streak.endX, hoodOld, nanoStep, UpdateLineXFlag());
    }

private:
    template<typename HOOD_NEW, typename HOOD_OLD>
    void updateWrapper(HOOD_NEW& hoodNew, int endX, HOOD_OLD& hoodOld, unsigned nanoStep, APITraits::FalseType)
    {
        for (int i = hoodOld.index(); i < endX; ++i, ++hoodOld) {
            hoodNew[i].update(hoodOld, nanoStep);
        }
    }

    template<typename HOOD_NEW, typename HOOD_OLD>
    void updateWrapper(HOOD_NEW& hoodNew, int endX, HOOD_OLD& hoodOld, unsigned nanoStep, APITraits::TrueType)
    {
        CELL::updateLineX(hoodNew, endX, hoodOld, nanoStep);
    }
};

}

#endif
