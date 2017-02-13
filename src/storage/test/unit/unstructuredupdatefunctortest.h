#include <cxxtest/TestSuite.h>

#include <libgeodecomp/config.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/unstructuredupdatefunctor.h>
#include <libgeodecomp/storage/unstructuredgrid.h>
#include <libgeodecomp/storage/unstructuredsoagrid.h>
#include <libgeodecomp/storage/updatefunctor.h>
#include <libgeodecomp/storage/sellcsigmasparsematrixcontainer.h>

#include <libflatarray/api_traits.hpp>
#include <libflatarray/macros.hpp>
#include <libflatarray/soa_accessor.hpp>
#include <libflatarray/short_vec.hpp>

#include <vector>
#include <map>

using namespace LibGeoDecomp;
using namespace LibFlatArray;

#ifdef LIBGEODECOMP_WITH_CPP14

class EmptyUnstructuredTestCellAPI
{};

class DefaultUnstructuredTestCellAPI : public APITraits::HasUpdateLineX
{};

template<int SIGMA, typename ADDITIONAL_API = DefaultUnstructuredTestCellAPI>
class SimpleUnstructuredTestCell
{
public:
    class API :
        public ADDITIONAL_API,
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasSellType<double>,
        public APITraits::HasSellMatrices<1>,
        public APITraits::HasSellC<4>,
        public APITraits::HasSellSigma<SIGMA>
    {};

    inline explicit SimpleUnstructuredTestCell(double v = 0) :
        value(v), sum(0)
    {}

    template<typename HOOD_NEW, typename HOOD_OLD>
    static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */)
    {
        for (; hoodOld.index() < indexEnd; ++hoodOld) {
            hoodNew->sum = 0.0;
            for (const auto& j: hoodOld.weights(0)) {
                hoodNew->sum += hoodOld[j.first()].value * j.second();
            }
            ++hoodNew;
        }
    }

    template<typename NEIGHBORHOOD>
    void update(NEIGHBORHOOD& neighborhood, unsigned /* nanoStep */)
    {
        sum = 0.;
        for (const auto& j: neighborhood.weights(0)) {
            sum += neighborhood[j.first()].value * j.second();
        }
    }

    inline bool operator==(const SimpleUnstructuredTestCell& other)
    {
        return sum == other.sum;
    }

    inline bool operator!=(const SimpleUnstructuredTestCell& other)
    {
        return !(*this == other);
    }

    double value;
    double sum;
};

template<int SIGMA>
class SimpleUnstructuredSoATestCell
{
public:
    typedef short_vec<double, 4> ShortVec;

    class API :
        public APITraits::HasUpdateLineX,
        public APITraits::HasSoA,
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasPredefinedMPIDataType<double>,
        public APITraits::HasSellType<double>,
        public APITraits::HasSellMatrices<1>,
        public APITraits::HasSellC<4>,
        public APITraits::HasSellSigma<SIGMA>
    {
    public:
        LIBFLATARRAY_CUSTOM_SIZES((16)(32)(64)(128)(256)(512), (1), (1))
    };

    inline explicit SimpleUnstructuredSoATestCell(double v = 0) :
        value(v), sum(0)
    {}

    template<typename SHORT_VEC_TYPE, typename COUNTER_TYPE1, typename COUNTER_TYPE2, typename HOOD_OLD, typename LAMBDA>
    static
    void unstructuredLoopPeeler(COUNTER_TYPE1 *counter, const COUNTER_TYPE2& end, HOOD_OLD& hoodOld, const LAMBDA& lambda)
    {
        typedef SHORT_VEC_TYPE lfa_local_short_vec;
        typedef typename LibFlatArray::detail::flat_array::
            sibling_short_vec_switch<SHORT_VEC_TYPE, 1>::VALUE
            lfa_local_scalar;

        COUNTER_TYPE1 nextStop = *counter;
        COUNTER_TYPE1 remainder = *counter & (SHORT_VEC_TYPE::ARITY - 1);
        if (remainder != 0) {
            nextStop += SHORT_VEC_TYPE::ARITY - remainder;
        }
        COUNTER_TYPE1 lastStop = end - end & (SHORT_VEC_TYPE::ARITY - 1);

        typedef UnstructuredSoANeighborhoodHelpers::WrappedNeighborhood<HOOD_OLD> WrappedHood;
        WrappedHood wrappedHood(hoodOld);

        lambda(lfa_local_scalar(),    counter, nextStop, wrappedHood);
        lambda(lfa_local_short_vec(), counter, lastStop, hoodOld);
        lambda(lfa_local_scalar(),    counter, end,      wrappedHood);
    }

    // fixme: duplicate these tests and rerun them as OpenMP tests,
    // with different OpenMP threading specs (true, true), (true,
    // false), (false, true), (false, false).
    // optionally add public APITraits::HasThreadedUpdate<16> to API for fine-grained updates


    // fixme: get rid of indexStart
    template<typename HOOD_NEW, typename HOOD_OLD>
    static void updateLineX(HOOD_NEW& hoodNew, int indexStart, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */)
    {
        unstructuredLoopPeeler<ShortVec>(&hoodNew.index(), indexEnd, hoodOld, [&hoodNew](auto REAL, auto *counter, const auto& end, auto& hoodOld) {
                typedef decltype(REAL) ShortVec;
                for (; hoodNew.index() < end; hoodNew += ShortVec::ARITY) {
                    ShortVec tmp;
                    tmp.load_aligned(&hoodNew->sum());

                    for (const auto& j: hoodOld.weights()) {
                        ShortVec weights, values;
                        weights.load_aligned(j.second());
                        values.gather(&hoodOld->value(), j.first());
                        tmp += values * weights;

                    }

                    &hoodNew->sum() << tmp;
                    ++hoodOld;
                }
            });
    }

    template<typename NEIGHBORHOOD>
    void update(NEIGHBORHOOD& neighborhood, unsigned /* nanoStep */)
    {
        sum = 0.;
        for (const auto& j: neighborhood.weights(0)) {
            sum += neighborhood[j.first()].value * j.second();
        }
    }

    inline bool operator==(const SimpleUnstructuredSoATestCell& cell) const
    {
        return cell.sum == sum;
    }

    inline bool operator!=(const SimpleUnstructuredSoATestCell& cell) const
    {
        return !(*this == cell);
    }

    double value;
    double sum;
};

LIBFLATARRAY_REGISTER_SOA(SimpleUnstructuredSoATestCell<1 >, ((double)(sum))((double)(value)))
LIBFLATARRAY_REGISTER_SOA(SimpleUnstructuredSoATestCell<60>, ((double)(sum))((double)(value)))
#endif

namespace LibGeoDecomp {

class UnstructuredUpdateFunctorTest : public CxxTest::TestSuite
{
public:
    void testBasicSansUpdateLineX()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        const int DIM = 150;
        Coord<1> dim(DIM);
        Region<1> boundingRegion;
        boundingRegion << CoordBox<1>(Coord<1>(0), dim);

        typedef SimpleUnstructuredTestCell<1, EmptyUnstructuredTestCellAPI> TestCellType;
        TestCellType defaultCell(200);
        TestCellType edgeCell(-1);

        typedef ReorderingUnstructuredGrid<UnstructuredGrid<TestCellType, 1, double, 4, 1> > GridType;
        GridType gridOld(boundingRegion, defaultCell, edgeCell);
        GridType gridNew(boundingRegion, defaultCell, edgeCell);

        Region<1> region;
        region << Streak<1>(Coord<1>(10),   30);
        region << Streak<1>(Coord<1>(40),   60);
        region << Streak<1>(Coord<1>(100), 150);

        // weights matrix looks like this: 1 0 1 0 1 0 ...
        std::map<Coord<2>, double> matrix;
        for (int row = 0; row < DIM; ++row) {
            for (int col = 0; col < DIM; col += 2) {
                matrix[Coord<2>(row, col)] = 1;
            }
        }
        gridOld.setWeights(0, matrix);

        UnstructuredUpdateFunctor<TestCellType > functor;
        UpdateFunctorHelpers::ConcurrencyNoP concurrencySpec;
        APITraits::SelectThreadedUpdate<TestCellType>::Value modelThreadingSpec;

        functor(region, gridOld, &gridNew, 0, concurrencySpec, modelThreadingSpec);

        for (Coord<1> coord(0); coord < Coord<1>(150); ++coord.x()) {
            if (((coord.x() >=  10) && (coord.x() <  30)) ||
                ((coord.x() >=  40) && (coord.x() <  60)) ||
                ((coord.x() >= 100) && (coord.x() < 150))) {
                const double sum = (DIM / 2.0) * 200.0;
                TS_ASSERT_EQUALS(sum, gridNew.get(coord).sum);
            } else {
                TS_ASSERT_EQUALS(0.0, gridNew.get(coord).sum);
            }
        }
#endif
    }

    void testBasicWithUpdateLineX()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        const int DIM = 150;
        Coord<1> dim(DIM);
        Region<1> boundingRegion;
        boundingRegion << CoordBox<1>(Coord<1>(0), dim);

        typedef SimpleUnstructuredTestCell<1> TestCellType;
        TestCellType defaultCell(200);
        TestCellType edgeCell(-1);

        typedef ReorderingUnstructuredGrid<UnstructuredGrid<TestCellType, 1, double, 4, 1> > GridType;
        GridType gridOld(boundingRegion, defaultCell, edgeCell);
        GridType gridNew(boundingRegion, defaultCell, edgeCell);

        Region<1> region;
        region << Streak<1>(Coord<1>(10),   30);
        region << Streak<1>(Coord<1>(40),   60);
        region << Streak<1>(Coord<1>(100), 150);

        // weights matrix looks like this: 1 0 1 0 1 0 ...
        std::map<Coord<2>, double> matrix;
        for (int row = 0; row < DIM; ++row) {
            for (int col = 0; col < DIM; col += 2) {
                matrix[Coord<2>(row, col)] = 1;
            }
        }
        gridOld.setWeights(0, matrix);

        UnstructuredUpdateFunctor<TestCellType > functor;
        UpdateFunctorHelpers::ConcurrencyNoP concurrencySpec;
        APITraits::SelectThreadedUpdate<TestCellType>::Value modelThreadingSpec;

        functor(region, gridOld, &gridNew, 0, concurrencySpec, modelThreadingSpec);

        for (Coord<1> coord(0); coord < Coord<1>(150); ++coord.x()) {
            if (((coord.x() >=  10) && (coord.x() <  30)) ||
                ((coord.x() >=  40) && (coord.x() <  60)) ||
                ((coord.x() >= 100) && (coord.x() < 150))) {
                const double sum = (DIM / 2.0) * 200.0;
                TS_ASSERT_EQUALS(sum, gridNew.get(coord).sum);
            } else {
                TS_ASSERT_EQUALS(0.0, gridNew.get(coord).sum);
            }
        }
#endif
    }

    void testBasicWithSigmaButSansUpdateLineX()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        const int DIM = 150;
        Coord<1> dim(DIM);
        Region<1> boundingRegion;
        boundingRegion << CoordBox<1>(Coord<1>(0), dim);

        typedef SimpleUnstructuredTestCell<128, EmptyUnstructuredTestCellAPI> TestCellType;
        TestCellType defaultCell(200);
        TestCellType edgeCell(-1);

        typedef ReorderingUnstructuredGrid<UnstructuredGrid<TestCellType, 1, double, 4, 128> > GridType;
        GridType gridOld(boundingRegion, defaultCell, edgeCell);
        GridType gridNew(boundingRegion, defaultCell, edgeCell);

        // weights matrix looks like this:
        // 0
        // 1
        // 1 1
        // 1 1 1
        // ...
        // -> force sorting
        std::map<Coord<2>, double> matrix;
        for (int row = 0; row < DIM; ++row) {
            for (int col = 0; col < row; ++col) {
                matrix[Coord<2>(row, col)] = 1;
            }
        }
        gridOld.setWeights(0, matrix);
        gridNew.setWeights(0, matrix);

        Region<1> region;
        region << Streak<1>(Coord<1>(10),   30);
        region << Streak<1>(Coord<1>(40),   60);
        region << Streak<1>(Coord<1>(100), 150);
        region = gridOld.remapRegion(region);

        UnstructuredUpdateFunctor<TestCellType > functor;
        UpdateFunctorHelpers::ConcurrencyNoP concurrencySpec;
        APITraits::SelectThreadedUpdate<TestCellType>::Value modelThreadingSpec;

        functor(region, gridOld, &gridNew, 0, concurrencySpec, modelThreadingSpec);

        for (Coord<1> coord(0); coord < Coord<1>(150); ++coord.x()) {
            if (((coord.x() >=  10) && (coord.x() <  30)) ||
                ((coord.x() >=  40) && (coord.x() <  60)) ||
                ((coord.x() >= 100) && (coord.x() < 150))) {
                const double sum = coord.x() * 200.0;
                TS_ASSERT_EQUALS(sum, gridNew.get(coord).sum);
            } else {
                TS_ASSERT_EQUALS(0.0, gridNew.get(coord).sum);
            }
        }
#endif
    }


    void testBasicWithSigmaAndWithUpdateLineX()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        const int DIM = 150;
        Coord<1> dim(DIM);
        Region<1> boundingRegion;
        boundingRegion << CoordBox<1>(Coord<1>(0), dim);

        typedef SimpleUnstructuredTestCell<128> TestCellType;
        TestCellType defaultCell(200);
        TestCellType edgeCell(-1);

        typedef ReorderingUnstructuredGrid<UnstructuredGrid<TestCellType, 1, double, 4, 128> > GridType;
        GridType gridOld(boundingRegion, defaultCell, edgeCell);
        GridType gridNew(boundingRegion, defaultCell, edgeCell);

        // weights matrix looks like this:
        // 0
        // 1
        // 1 1
        // 1 1 1
        // ...
        // -> force sorting
        std::map<Coord<2>, double> matrix;
        for (int row = 0; row < DIM; ++row) {
            for (int col = 0; col < row; ++col) {
                matrix[Coord<2>(row, col)] = 1;
            }
        }
        gridOld.setWeights(0, matrix);
        gridNew.setWeights(0, matrix);

        Region<1> region;
        region << Streak<1>(Coord<1>(10),   30);
        region << Streak<1>(Coord<1>(40),   60);
        region << Streak<1>(Coord<1>(100), 150);
        region = gridOld.remapRegion(region);

        UnstructuredUpdateFunctor<TestCellType > functor;
        UpdateFunctorHelpers::ConcurrencyNoP concurrencySpec;
        APITraits::SelectThreadedUpdate<TestCellType>::Value modelThreadingSpec;

        functor(region, gridOld, &gridNew, 0, concurrencySpec, modelThreadingSpec);

        for (Coord<1> coord(0); coord < Coord<1>(150); ++coord.x()) {
            if (((coord.x() >=  10) && (coord.x() <  30)) ||
                ((coord.x() >=  40) && (coord.x() <  60)) ||
                ((coord.x() >= 100) && (coord.x() < 150))) {

                const double sum = coord.x() * 200;
                TS_ASSERT_EQUALS(sum, gridNew.get(coord).sum);
            } else {
                TS_ASSERT_EQUALS(0.0, gridNew.get(coord).sum);
            }
        }
#endif
    }

    void testSoA()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        const int DIM = 150;
        CoordBox<1> dim(Coord<1>(0), Coord<1>(DIM));
        Region<1> boundingRegion;
        boundingRegion << dim;

        SimpleUnstructuredSoATestCell<1> defaultCell(200);
        SimpleUnstructuredSoATestCell<1> edgeCell(-1);

        typedef ReorderingUnstructuredGrid<UnstructuredSoAGrid<SimpleUnstructuredSoATestCell<1>, 1, double, 4, 1> > GridType;
        GridType gridOld(boundingRegion, defaultCell, edgeCell);
        GridType gridNew(boundingRegion, defaultCell, edgeCell);

        for (int i = 0; i < DIM; ++i) {
            gridOld.set(Coord<1>(i), SimpleUnstructuredSoATestCell<1>(2000 + i));
        }

        // weights matrix looks like this:
        // 0
        // 1
        // 2 12
        // 3 13 23
        // ...
        std::map<Coord<2>, double> matrix;
        // fixme: use non-uniform weights in all tests, make factor differ
        for (int row = 0; row < DIM; ++row) {
            for (int col = 0; col < row; ++col) {
                matrix[Coord<2>(row, col)] = row + col * 10;
            }
        }
        gridOld.setWeights(0, matrix);
        gridNew.setWeights(0, matrix);

        Region<1> region;
        // loop peeling in first and last chunk
        region << Streak<1>(Coord<1>(10),   30);
        // loop peeling in first chunk
        region << Streak<1>(Coord<1>(37),   60);
        // "normal" streak
        region << Streak<1>(Coord<1>(64),   80);
        // loop peeling in last chunk
        region << Streak<1>(Coord<1>(100), 149);
        region = gridOld.remapRegion(region);

        UnstructuredUpdateFunctor<SimpleUnstructuredSoATestCell<1> > functor;
        UpdateFunctorHelpers::ConcurrencyNoP concurrencySpec;
        APITraits::SelectThreadedUpdate<SimpleUnstructuredSoATestCell<1> >::Value modelThreadingSpec;

        functor(region, gridOld, &gridNew, 0, concurrencySpec, modelThreadingSpec);

        for (Coord<1> coord(0); coord < Coord<1>(150); ++coord.x()) {
            if (region.count(coord)) {
                double sum = 0;
                for (int i = 0; i < coord.x(); ++i) {
                    double weight = coord.x() + i * 10;
                    sum += weight * (2000 + i);
                }
                TS_ASSERT_EQUALS(sum, gridNew.get(coord).sum);
            } else {
                TS_ASSERT_EQUALS(0.0, gridNew.get(coord).sum);
            }
        }
#endif
    }

    void testSoAWithSIGMA()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        const int DIM = 150;
        CoordBox<1> dim(Coord<1>(0), Coord<1>(DIM));
        Region<1> boundingRegion;
        boundingRegion << dim;

        SimpleUnstructuredSoATestCell<60> defaultCell(200);
        SimpleUnstructuredSoATestCell<60> edgeCell(-1);

        typedef ReorderingUnstructuredGrid<UnstructuredSoAGrid<SimpleUnstructuredSoATestCell<60>, 1, double, 4, 60> > GridType;
        GridType gridOld(boundingRegion, defaultCell, edgeCell);
        GridType gridNew(boundingRegion, defaultCell, edgeCell);

        for (int i = 0; i < DIM; ++i) {
            gridOld.set(Coord<1>(i), SimpleUnstructuredSoATestCell<60>(3000 + i));
        }

        // weights matrix looks like this:
        // 0
        // 1
        // 2 12
        // 3 13 23
        // ...
        std::map<Coord<2>, double> matrix;
        for (int row = 0; row < DIM; ++row) {
            for (int col = 0; col < row; ++col) {
                matrix[Coord<2>(row, col)] = row + col * 10;
            }
        }
        gridOld.setWeights(0, matrix);
        gridNew.setWeights(0, matrix);

        Region<1> region;
        // use the same variation of Streaks as in the previous
        // example, even though they will not correspond to the same
        // special cases (with regard to starting/ending on chunk
        // boundaries) as the Region get's remapped anyway.
        region << Streak<1>(Coord<1>(10),   30);
        region << Streak<1>(Coord<1>(37),   60);
        region << Streak<1>(Coord<1>(64),   80);
        region << Streak<1>(Coord<1>(100), 149);
        Region<1> updateRegion = gridOld.remapRegion(region);

        UnstructuredUpdateFunctor<SimpleUnstructuredSoATestCell<60> > functor;
        UpdateFunctorHelpers::ConcurrencyNoP concurrencySpec;
        APITraits::SelectThreadedUpdate<SimpleUnstructuredSoATestCell<60> >::Value modelThreadingSpec;

        functor(updateRegion, gridOld, &gridNew, 0, concurrencySpec, modelThreadingSpec);

        for (Coord<1> coord(0); coord < Coord<1>(150); ++coord.x()) {
            if (region.count(coord)) {
                double sum = 0;
                for (int i = 0; i < coord.x(); ++i) {
                    double weight = coord.x() + i * 10;
                    sum += weight * (3000 + i);
                }
                TS_ASSERT_EQUALS(sum, gridNew.get(coord).sum);
            } else {
                TS_ASSERT_EQUALS(0.0, gridNew.get(coord).sum);
            }
        }
#endif
    }
};

}
