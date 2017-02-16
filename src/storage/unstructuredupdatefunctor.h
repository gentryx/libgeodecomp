#ifndef LIBGEODECOMP_STORAGE_UNSTRUCTUREDUPDATEFUNCTOR_H
#define LIBGEODECOMP_STORAGE_UNSTRUCTUREDUPDATEFUNCTOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_CPP14

#include <libflatarray/soa_accessor.hpp>

#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/async.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#endif

#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/geometry/streak.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/storage/fixedarray.h>
#include <libgeodecomp/storage/unstructuredsoagrid.h>
#include <libgeodecomp/storage/unstructuredneighborhood.h>
#include <libgeodecomp/storage/unstructuredsoaneighborhood.h>
#include <libgeodecomp/storage/unstructuredsoaneighborhoodnew.h>
#include <libgeodecomp/storage/unstructuredsoascalarneighborhood.h>
#include <libgeodecomp/storage/updatefunctormacros.h>

namespace LibGeoDecomp {

namespace UnstructuredUpdateFunctorHelpers {

/**
 * Functor to be used from with LibFlatArray from within
 * UnstructuredUpdateFunctor. Hides much of the boilerplate code.
 */
template<
    typename CELL,
    typename GRID_TYPE,
    typename CONCURRENCY_SPEC,
    typename MODEL_THREADING_SPEC>
class UnstructuredGridSoAUpdateHelper
{
public:
    using Topology = typename APITraits::SelectTopology<CELL>::Value;
    using ValueType = typename APITraits::SelectSellType<CELL>::Value;
    static const auto MATRICES = APITraits::SelectSellMatrices<CELL>::VALUE;
    static const auto C = APITraits::SelectSellC<CELL>::VALUE;
    static const auto SIGMA = APITraits::SelectSellSigma<CELL>::VALUE;
    static const auto DIM = Topology::DIM;

    UnstructuredGridSoAUpdateHelper(
        const GRID_TYPE& gridOld,
        GRID_TYPE *gridNew,
        const Region<DIM>& region,
        unsigned nanoStep,
        const CONCURRENCY_SPEC& concurrencySpec,
        const MODEL_THREADING_SPEC& modelThreadingSpec) :
        gridOld(gridOld),
        gridNew(gridNew),
        region(region),
        nanoStep(nanoStep),
        concurrencySpec(concurrencySpec),
        modelThreadingSpec(modelThreadingSpec)
    {}

    template<
        typename CELL1, long MY_DIM_X1, long MY_DIM_Y1, long MY_DIM_Z1, long INDEX1,
        typename CELL2, long MY_DIM_X2, long MY_DIM_Y2, long MY_DIM_Z2, long INDEX2>
    void operator()(
        LibFlatArray::soa_accessor<CELL1, MY_DIM_X1, MY_DIM_Y1, MY_DIM_Z1, INDEX1>& oldAccessor,
        LibFlatArray::soa_accessor<CELL2, MY_DIM_X2, MY_DIM_Y2, MY_DIM_Z2, INDEX2>& newAccessor) const
    {
#define LGD_UPDATE_FUNCTOR_BODY                                         \
        typedef UnstructuredSoANeighborhood<GRID_TYPE, CELL, MY_DIM_X1, MY_DIM_Y1, MY_DIM_Z1, INDEX1, MATRICES, ValueType, C, SIGMA> HoodOld; \
        int intraChunkOffset = i->origin.x() % HoodOld::ARITY;          \
        HoodOld hoodOld(                                                \
            oldAccessor,                                                \
            gridOld,                                                    \
            i->origin.x(),                                              \
            intraChunkOffset);                                          \
        LibFlatArray::soa_accessor<CELL2, MY_DIM_X2, MY_DIM_Y2, MY_DIM_Z2, INDEX2> newAccessorCopy( \
            newAccessor.data(),                                         \
            i->origin.x());                                             \
        UnstructuredSoANeighborhoodNew<CELL, MY_DIM_X2, MY_DIM_Y2, MY_DIM_Z2, INDEX2> hoodNew(&newAccessorCopy); \
        CELL::updateLineX(hoodNew, i->endX, hoodOld, nanoStep); \
        /**/
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_1
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_2
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_3
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_4
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_5
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_6
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_7
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_8
#undef LGD_UPDATE_FUNCTOR_BODY

        //     // fixme: threading!
        // for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
        //     // call updateLineX with adjusted indices
        //     typedef UnstructuredSoANeighborhood<GRID_TYPE, CELL, MY_DIM_X1, MY_DIM_Y1, MY_DIM_Z1, INDEX1, MATRICES, ValueType, C, SIGMA> HoodOld;
        //     int intraChunkOffset = i->origin.x() % HoodOld::ARITY;
        //     HoodOld hoodOld(oldAccessor, gridOld, i->origin.x(), intraChunkOffset);
        //     newAccessor.index() = i->origin.x();
        //     UnstructuredSoANeighborhoodNew<CELL, MY_DIM_X2, MY_DIM_Y2, MY_DIM_Z2, INDEX2> hoodNew(&newAccessor);
        //     CELL::updateLineX(hoodNew, i->origin.x(), i->endX, hoodOld, nanoStep);
        // }
    }

private:
    const GRID_TYPE& gridOld;
    GRID_TYPE *gridNew;
    const Region<DIM>& region;
    unsigned nanoStep;
    const CONCURRENCY_SPEC& concurrencySpec;
    const MODEL_THREADING_SPEC& modelThreadingSpec;
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

        switchAoSvsSoA(region, gridOld, gridNew, nanoStep, concurrencySpec, modelThreadingSpec, SoAFlag());
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

// #ifdef LIBGEODECOMP_WITH_HPX
//         // fixme: manual hack, should use infrastructure from updatefunctormacros.
//         // fixme: also desirable: user-selectable switch for granularity
//         // fixme: hotfix for zach
//         if (concurrencySpec.enableHPX()) {
//             std::cout << "in HPX loop\n";
//         // if (concurrencySpec.enableHPX() && concurrencySpec.preferFineGrainedParallelism()) {
//             std::vector<hpx::future<void> > updateFutures;
//             for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
//                 UnstructuredNeighborhood<CELL, MATRICES, ValueType, C, SIGMA> hoodOld(gridOld, i->origin.x());
//                 CELL *hoodNew = &(*gridNew)[i->origin.x()];
//                 int origin = i->origin.x();
//                 for (int offset = 0; offset < i->length(); ++offset) {
//                     updateFutures << hpx::async(
//                         [&hoodOld, &hoodNew, origin, offset, nanoStep]() {
//                             UnstructuredNeighborhood<CELL, MATRICES, ValueType, C, SIGMA> hoodOldMoved = hoodOld;
//                             hoodOldMoved += long(offset);
//                             hoodNew[offset].update(hoodOldMoved, nanoStep);
//                         });
//                 }
//             }

//             hpx::lcos::wait_all(std::move(updateFutures));
//             // hpx::parallel::for_each(
//             //     hpx::parallel::par,
//             //     boost::make_counting_iterator(hoodOld.index()),
//             //     boost::make_counting_iterator(long(endX)),
//             //     [&](std::size_t i) {
//             //         HOOD_OLD hoodOldMoved = hoodOld;
//             //         hoodOldMoved += long(i);
//             //         hoodNew[i].update(hoodOldMoved, nanoStep);
//             //     });

//             return;
//         }
// #endif

#define LGD_UPDATE_FUNCTOR_BODY                                         \
        UnstructuredNeighborhood<CELL, MATRICES, ValueType, C, SIGMA>   \
            hoodOld(gridOld, i->origin.x());                            \
        CELL *hoodNew = &(*gridNew)[i->origin.x()];                     \
        for (int offset = 0; offset != i->length(); ++offset, ++hoodOld) { \
            hoodNew[offset].update(hoodOld, nanoStep);                  \
        }                                                               \
        /**/
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_1
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_2
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_3
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_4
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_5
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_6
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_7
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_8
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
        CELL *hoodNew = &(*gridNew)[i->origin.x()];                     \
        CELL::updateLineX(hoodNew, i->endX, hoodOld, nanoStep); \
        /**/
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_1
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_2
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_3
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_4
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_5
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_6
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_7
        LGD_UPDATE_FUNCTOR_THREADING_SELECTOR_8
#undef LGD_UPDATE_FUNCTOR_BODY
    }

    template<typename GRID1, typename GRID2, typename CONCURRENCY_FUNCTOR, typename ANY_THREADED_UPDATE>
    void switchAoSvsSoA(
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
    void switchAoSvsSoA(
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
            UnstructuredUpdateFunctorHelpers::UnstructuredGridSoAUpdateHelper<CELL, GRID1, CONCURRENCY_FUNCTOR, ANY_THREADED_UPDATE>(
                gridOld,
                gridNew,
                region,
                nanoStep,
                concurrencySpec,
                modelThreadingSpec));
    }
};

}

#endif
#endif
