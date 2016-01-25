#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDUPDATEFUNCTOR_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDUPDATEFUNCTOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libflatarray/soa_accessor.hpp>

#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/async.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#endif

#include <boost/iterator/counting_iterator.hpp>

#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/storage/fixedarray.h>
#include <libgeodecomp/storage/unstructuredsoagrid.h>
#include <libgeodecomp/storage/unstructuredneighborhood.h>
#include <libgeodecomp/storage/unstructuredsoaneighborhood.h>
#include <libgeodecomp/storage/updatefunctormacros.h>

namespace LibGeoDecomp {

namespace UnstructuredUpdateFunctorHelpers {

/**
 * Functor to be used from with LibFlatArray from within
 * UnstructuredUpdateFunctor. Hides much of the boilerplate code.
 */
template<typename CELL>
class UnstructuredGridSoAUpdateHelper
{
public:
    using Topology = typename APITraits::SelectTopology<CELL>::Value;
    using ValueType = typename APITraits::SelectSellType<CELL>::Value;
    static const auto MATRICES = APITraits::SelectSellMatrices<CELL>::VALUE;
    static const auto C = APITraits::SelectSellC<CELL>::VALUE;
    static const auto SIGMA = APITraits::SelectSellSigma<CELL>::VALUE;
    static const auto DIM = Topology::DIM;
    using Grid = UnstructuredSoAGrid<CELL, MATRICES, ValueType, C, SIGMA>;

    UnstructuredGridSoAUpdateHelper(
        const Grid& gridOld,
        Grid *gridNew,
        const Region<DIM>& region,
        unsigned nanoStep) :
        gridOld(gridOld),
        gridNew(gridNew),
        region(region),
        nanoStep(nanoStep)
    {}

    template<
        typename CELL1, long MY_DIM_X1, long MY_DIM_Y1, long MY_DIM_Z1, long INDEX1,
        typename CELL2, long MY_DIM_X2, long MY_DIM_Y2, long MY_DIM_Z2, long INDEX2>
    void operator()(
        LibFlatArray::soa_accessor<CELL1, MY_DIM_X1, MY_DIM_Y1, MY_DIM_Z1, INDEX1>& oldAccessor,
        LibFlatArray::soa_accessor<CELL2, MY_DIM_X2, MY_DIM_Y2, MY_DIM_Z2, INDEX2>& newAccessor) const
    {
        // fixme: threading!
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            // Assumption: Cell has both (updateLineX and update())

            // loop peeling: streak's start might point to middle of chunks
            // if so: vectorization cannot be done -> solution: additionally
            // update the first and last chunk of complete streak scalar by
            // calling update() instead
            int startX = i->origin.x();
            if ((startX % C) != 0) {
                UnstructuredSoAScalarNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>
                    hoodOld(gridOld, startX);
                const int cellsToUpdate = C - (startX % C);
                FixedArray<CELL, C> cells;
                Streak<1> cellStreak(Coord<1>(startX), startX + cellsToUpdate);

                // update SoA grid: copy cells to local buffer, update, copy data back to grid
                gridNew->get(cellStreak, cells.begin());
                // call update
                for (int i = 0; i < cellsToUpdate; ++i, ++hoodOld) {
                    cells[i].update(hoodOld, nanoStep);
                }
                // fixme: woah, avoid these copies!
                gridNew->set(cellStreak, cells.begin());

                startX += cellsToUpdate;
            }

            // call updateLineX with adjusted indices
            UnstructuredSoANeighborhood<CELL, MY_DIM_X1, MY_DIM_Y1, MY_DIM_Z1, INDEX1,
                                        MATRICES, ValueType, C, SIGMA>
                hoodOld(oldAccessor, gridOld, startX);
            UnstructuredSoANeighborhoodNew<CELL, MY_DIM_X2, MY_DIM_Y2, MY_DIM_Z2, INDEX2>
                hoodNew(newAccessor);
            const int endX = i->endX / C;
            CELL::updateLineX(hoodNew, endX, hoodOld, nanoStep);

            // call scalar updates for last chunk
            if ((i->endX % C) != 0) {
                const int cellsToUpdate = i->endX % C;
                UnstructuredSoAScalarNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>
                    hoodOld(gridOld, i->endX - cellsToUpdate);
                std::array<CELL, C> cells;
                Streak<1> cellStreak(Coord<1>(i->endX - cellsToUpdate), i->endX);

                // update SoA grid: copy cells to local buffer, update, copy data back to grid
                gridNew->get(cellStreak, cells.begin());
                // call update
                for (int i = 0; i < cellsToUpdate; ++i, ++hoodOld) {
                    cells[i].update(hoodOld, nanoStep);
                }
                gridNew->set(cellStreak, cells.begin());
            }
        }
    }

private:
    const Grid& gridOld;
    Grid *gridNew;
    const Region<DIM>& region;
    unsigned nanoStep;
};

}

/**
 * Unsurprisingly, this update functor negotiates between cells that
 * support unstructured grids and the actual grid implementation.
 */
template<typename CELL>
class UnstructuredUpdateFunctor
{
public:
    using Topology = typename APITraits::SelectTopology<CELL>::Value;
    using ValueType = typename APITraits::SelectSellType<CELL>::Value;
    static const std::size_t MATRICES = APITraits::SelectSellMatrices<CELL>::VALUE;
    static const int C = APITraits::SelectSellC<CELL>::VALUE;
    static const int SIGMA = APITraits::SelectSellSigma<CELL>::VALUE;
    static const int DIM = Topology::DIM;

    template<typename GRID1, typename GRID2, typename CONCURRENCY_FUNCTOR, typename ANY_THREADED_UPDATE>
    void operator()(
        const Region<DIM>& region,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        const CONCURRENCY_FUNCTOR& concurrencySpec,
        const ANY_THREADED_UPDATE& modelThreadingSpec)
    {
        typedef typename APITraits::SelectSoA<CELL>::Value SoAFlag;

        soaWrapper(region, gridOld, gridNew, nanoStep, concurrencySpec, modelThreadingSpec, SoAFlag());
    }


    template<typename GRID1, typename GRID2, typename CONCURRENCY_FUNCTOR, typename ANY_THREADED_UPDATE>
    void apiWrapper(
        const Region<DIM>& region,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        const CONCURRENCY_FUNCTOR& concurrencySpec,
        const ANY_THREADED_UPDATE& modelThreadingSpec,
        // has cell no updateLineX()?
        APITraits::FalseType)
    {
        // HOOD_NEW& hoodNew, int endX, HOOD_OLD& hoodOld,
        // UnstructuredNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>
        //     hoodOld(gridOld, streak.origin.x());
        // CellIDNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>
        //     hoodNew(*gridNew);


#ifdef LIBGEODECOMP_WITH_HPX
        // fixme: manual hack, should use infrastructure from updatefunctormacros.
        // fixme: also desirable: user-selectable switch for granularity
        // fixme: hotfix for zach
        if (concurrencySpec.enableHPX()) {
        // if (concurrencySpec.enableHPX() && concurrencySpec.preferFineGrainedParallelism()) {
            std::vector<hpx::future<void> > updateFutures;
            for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
                UnstructuredNeighborhood<CELL, MATRICES, ValueType, C, SIGMA> hoodOld(gridOld, i->origin.x());
                CellIDNeighborhood<CELL, MATRICES, ValueType, C, SIGMA> hoodNew(*gridNew);
                int origin = i->origin.x();
                for (int offset = 0; offset < i->length(); ++offset) {
                    updateFutures << hpx::async(
                        [&hoodOld, &hoodNew, origin, offset, nanoStep]() {
                            UnstructuredNeighborhood<CELL, MATRICES, ValueType, C, SIGMA> hoodOldMoved = hoodOld;
                            hoodOldMoved += long(offset);
                            hoodNew[origin + offset].update(hoodOldMoved, nanoStep);
                        });
                }
            }

            hpx::lcos::wait_all(std::move(updateFutures));
            // hpx::parallel::for_each(
            //     hpx::parallel::par,
            //     boost::make_counting_iterator(hoodOld.index()),
            //     boost::make_counting_iterator(long(endX)),
            //     [&](std::size_t i) {
            //         HOOD_OLD hoodOldMoved = hoodOld;
            //         hoodOldMoved += long(i);
            //         hoodNew[i].update(hoodOldMoved, nanoStep);
            //     });

            return;
        }
#endif
#define LGD_UPDATE_FUNCTOR_BODY                                         \
        UnstructuredNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>   \
            hoodOld(gridOld, i->origin.x());                            \
        CellIDNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>         \
            hoodNew(*gridNew);                                          \
        for (int id = i->origin.x(); id != i->endX; ++id, ++hoodOld) {  \
            hoodNew[id].update(hoodOld, nanoStep);                      \
        }                                                               \
        /**/
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_1
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_2
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_3
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_4
#undef LGD_UPDATE_FUNCTOR_BODY
    }

    template<typename GRID1, typename GRID2, typename CONCURRENCY_FUNCTOR, typename ANY_THREADED_UPDATE>
    void apiWrapper(
        const Region<DIM>& region,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        const CONCURRENCY_FUNCTOR& concurrencySpec,
        const ANY_THREADED_UPDATE& modelThreadingSpec,
        // has cell updateLineX()?
        APITraits::TrueType)
    {
#define LGD_UPDATE_FUNCTOR_BODY                                         \
        UnstructuredNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>   \
            hoodOld(gridOld, i->origin.x());                            \
        CellIDNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>         \
            hoodNew(*gridNew);                                          \
        CELL::updateLineX(hoodNew, i->endX, hoodOld, nanoStep);         \
        /**/
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_1
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_2
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_3
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_4
#undef LGD_UPDATE_FUNCTOR_BODY
    }

    template<typename GRID1, typename GRID2, typename CONCURRENCY_FUNCTOR, typename ANY_THREADED_UPDATE>
    void soaWrapper(
        const Region<DIM>& region,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        const CONCURRENCY_FUNCTOR& concurrencySpec,
        const ANY_THREADED_UPDATE& modelThreadingSpec,
        // has cell no struct-of-arrays format?
        APITraits::FalseType)
    {
        typedef typename APITraits::SelectUpdateLineX<CELL>::Value UpdateLineXFlag;
        // switch between updateLineX() and update()
        apiWrapper(region, gridOld, gridNew, nanoStep, concurrencySpec, modelThreadingSpec, UpdateLineXFlag());
    }

    template<typename GRID1, typename GRID2, typename CONCURRENCY_FUNCTOR, typename ANY_THREADED_UPDATE>
    void soaWrapper(
        const Region<DIM>& region,
        const GRID1& gridOld,
        GRID2 *gridNew,
        unsigned nanoStep,
        const CONCURRENCY_FUNCTOR& concurrencySpec,
        const ANY_THREADED_UPDATE& modelThreadingSpec,
        // is cell struct-of-arrays-enabled?
        APITraits::TrueType)
    {
        gridOld.callback(
            gridNew,
            // fixme: is missing concurrencySpec, modelThreadingSpec
            UnstructuredUpdateFunctorHelpers::UnstructuredGridSoAUpdateHelper<CELL>(
                gridOld,
                gridNew,
                region,
                nanoStep));
    }
};

}

#endif
#endif
