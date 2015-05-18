#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDUPDATEFUNCTOR_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDUPDATEFUNCTOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/unstructuredneighborhood.h>
#include <libgeodecomp/storage/unstructuredsoaneighborhood.h>

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
    void apiWrapper(HOOD_NEW& hoodNew, int endX, HOOD_OLD& hoodOld, unsigned nanoStep, APITraits::FalseType)
    {
        for (int i = hoodOld.index(); i < endX; ++i, ++hoodOld) {
            hoodNew[i].update(hoodOld, nanoStep);
        }
    }

    template<typename HOOD_NEW, typename HOOD_OLD>
    void apiWrapper(HOOD_NEW& hoodNew, int endX, HOOD_OLD& hoodOld, unsigned nanoStep, APITraits::TrueType)
    {
        CELL::updateLineX(hoodNew, endX, hoodOld, nanoStep);
    }

    template<typename GRID1, typename GRID2>
    void soaWrapper(const Streak<DIM>& streak, const GRID1& gridOld,
                    GRID2 *gridNew, unsigned nanoStep, APITraits::FalseType)
    {
        typedef typename APITraits::SelectUpdateLineX<CELL>::Value UpdateLineXFlag;

        UnstructuredNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>
            hoodOld(gridOld, streak.origin.x());
        CellIDNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>
            hoodNew(*gridNew);

        // switch between updateLineX() and update()
        apiWrapper(hoodNew, streak.endX, hoodOld, nanoStep, UpdateLineXFlag());
    }

    template<typename GRID1, typename GRID2>
    void soaWrapper(const Streak<DIM>& streak, const GRID1& gridOld,
                    GRID2 *gridNew, unsigned nanoStep, APITraits::TrueType)
    {
        typedef typename APITraits::SelectUpdateLineX<CELL>::Value UpdateLineXFlag;
        static const int SELLC = APITraits::SelectSellC<CELL>::VALUE;

        UnstructuredSoANeighborhood<CELL, MATRICES, ValueType, C, SIGMA>
            hoodOld(gridOld, streak.origin.x());
        UnstructuredSoANeighborhoodNew<CELL, MATRICES, ValueType, C, SIGMA>
            hoodNew(*gridNew);

        // switch between updateLineX() and update()
        apiWrapper(hoodNew, streak.endX / SELLC, hoodOld, nanoStep, UpdateLineXFlag());
    }

public:
    template<typename GRID1, typename GRID2>
    void operator()(
        const Streak<DIM>& streak,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep)
    {
        typedef typename APITraits::SelectSoA<CELL>::Value SoAFlag;

        // switch between SoA and non-SoA code
        soaWrapper(streak, gridOld, gridNew, nanoStep, SoAFlag());
    }
};

}

#endif
#endif
