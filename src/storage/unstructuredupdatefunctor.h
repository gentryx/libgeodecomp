#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDUPDATEFUNCTOR_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDUPDATEFUNCTOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/geometry/coord.h>
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
    using Topology = typename APITraits::SelectTopology<CELL>::Value;
    using ValueType = typename APITraits::SelectSellType<CELL>::Value;
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
        // Assumption: Cell has both (updateLineX and update())

        // loop peeling: streak's start might point to middle of chunks
        // if so: vectorization cannot be done -> solution: additionally
        // update the first and last chunk of complete streak scalar by
        // calling update() instead
        int startX = streak.origin.x();
        if ((startX % C) != 0) {
            UnstructuredSoAScalarNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>
                hoodOld(gridOld, startX);
            const int cellsToUpdate = C - (startX % C);
            CELL cells[cellsToUpdate];
            Streak<1> cellStreak(Coord<1>(startX), startX + cellsToUpdate);

            // update SoA grid: copy cells to local buffer, update, copy data back to grid
            gridNew->get(cellStreak, cells);
            // call update
            for (int i = 0; i < cellsToUpdate; ++i, ++hoodOld) {
                cells[i].update(hoodOld, nanoStep);
            }
            gridNew->set(cellStreak, cells);

            startX += cellsToUpdate;
        }

        // call updateLineX with adjusted indices
        UnstructuredSoANeighborhood<CELL, MATRICES, ValueType, C, SIGMA>
            hoodOld(gridOld, startX);
        UnstructuredSoANeighborhoodNew<CELL, MATRICES, ValueType, C, SIGMA>
            hoodNew(*gridNew);
        const int endX = streak.endX / C;
        CELL::updateLineX(hoodNew, endX, hoodOld, nanoStep);

        // call scalar updates for last chunk
        if ((streak.endX % C) != 0) {
            const int cellsToUpdate = streak.endX % C;
            UnstructuredSoAScalarNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>
                hoodOld(gridOld, streak.endX - cellsToUpdate);
            CELL cells[cellsToUpdate];
            Streak<1> cellStreak(Coord<1>(streak.endX - cellsToUpdate), streak.endX);

            // update SoA grid: copy cells to local buffer, update, copy data back to grid
            gridNew->get(cellStreak, cells);
            // call update
            for (int i = 0; i < cellsToUpdate; ++i, ++hoodOld) {
                cells[i].update(hoodOld, nanoStep);
            }
            gridNew->set(cellStreak, cells);
        }
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
