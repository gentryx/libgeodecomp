#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDUPDATEFUNCTOR_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDUPDATEFUNCTOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/unstructuredneighborhood.h>

namespace LibGeoDecomp {

/**
 * Unsurprisingly, this update functor negotiates between cells that
 * support unstructured grids and the actual grid implementation.
 */
template<typename CELL>
class UnstructuredUpdateFunctor
{
private:
    typedef typename APITraits::SelectTopology<CELL>::Value Topology;
    typedef typename APITraits::SelectSellType<CELL>::Value ValueType;
    static const std::size_t MATRICES = APITraits::SelectSellMatrices<CELL>::VALUE;
    static const int C = APITraits::SelectSellC<CELL>::VALUE;
    static const int SIGMA = APITraits::SelectSellSigma<CELL>::VALUE;
    static const int DIM = Topology::DIM;

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

public:
    template<typename GRID1, typename GRID2>
    void operator()(
        const Streak<DIM>& streak,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep)
    {
        typedef typename APITraits::SelectUpdateLineX<CELL>::Value UpdateLineXFlag;

        UnstructuredNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>
            hoodOld(gridOld, streak.origin.x());
        CellIDNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>
            hoodNew(*gridNew);

        // switch between updateLineX() and update()
        updateWrapper(hoodNew, streak.endX, hoodOld, nanoStep, UpdateLineXFlag());
    }
};

}

#endif
#endif
