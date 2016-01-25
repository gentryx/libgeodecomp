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
class UnstructuredTestCell
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

    inline explicit UnstructuredTestCell(double v = 0) :
        value(v), sum(0)
    {}

    template<typename HOOD_NEW, typename HOOD_OLD>
    static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */)
    {
        for (int i = hoodOld.index(); i < indexEnd; ++i, ++hoodOld) {
            hoodNew[i].sum = 0.;
            for (const auto& j: hoodOld.weights(0)) {
                hoodNew[i].sum += hoodOld[j.first].value * j.second;
            }
        }
    }

    template<typename NEIGHBORHOOD>
    void update(NEIGHBORHOOD& neighborhood, unsigned /* nanoStep */)
    {
        sum = 0.;
        for (const auto& j: neighborhood.weights(0)) {
            sum += neighborhood[j.first].value * j.second;
        }
    }

    inline bool operator==(const UnstructuredTestCell& other)
    {
        return sum == other.sum;
    }

    inline bool operator!=(const UnstructuredTestCell& other)
    {
        return !(*this == other);
    }

    double value;
    double sum;
};

// fixme: unify with UnstructuredTestCell?
// fixme: move both test cell classes to dedicated file?
template<int SIGMA>
class UnstructuredSoATestCell
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

    inline explicit UnstructuredSoATestCell(double v = 0) :
        value(v), sum(0)
    {}

    template<typename HOOD_NEW, typename HOOD_OLD>
    static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */)
    {
        for (int i = hoodOld.index(); i < indexEnd; ++i, ++hoodOld) {
            ShortVec tmp;
            tmp.load_aligned(&hoodNew->sum() + i * 4);
            for (const auto& j: hoodOld.weights(0)) {
                ShortVec weights, values;
                weights.load_aligned(j.second);
                values.gather(&hoodOld->value(), j.first);
                tmp += values * weights;
            }
            tmp.store_aligned(&hoodNew->sum() + i * 4);
        }
    }

    template<typename NEIGHBORHOOD>
    void update(NEIGHBORHOOD& neighborhood, unsigned /* nanoStep */)
    {
        sum = 0.;
        for (const auto& j: neighborhood.weights(0)) {
            sum += neighborhood[j.first].value * j.second;
        }
    }

    inline bool operator==(const UnstructuredSoATestCell& cell) const
    {
        return cell.sum == sum;
    }

    inline bool operator!=(const UnstructuredSoATestCell& cell) const
    {
        return !(*this == cell);
    }

    double value;
    double sum;
};

LIBFLATARRAY_REGISTER_SOA(UnstructuredSoATestCell<1  >, ((double)(sum))((double)(value)))
LIBFLATARRAY_REGISTER_SOA(UnstructuredSoATestCell<150>, ((double)(sum))((double)(value)))
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

        typedef UnstructuredTestCell<1, EmptyUnstructuredTestCellAPI> TestCellType;
        TestCellType defaultCell(200);
        TestCellType edgeCell(-1);

        UnstructuredGrid<TestCellType, 1, double, 4, 1> gridOld(dim, defaultCell, edgeCell);
        UnstructuredGrid<TestCellType, 1, double, 4, 1> gridNew(dim, defaultCell, edgeCell);

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

        typedef UnstructuredTestCell<1> TestCellType;
        TestCellType defaultCell(200);
        TestCellType edgeCell(-1);

        UnstructuredGrid<TestCellType, 1, double, 4, 1> gridOld(dim, defaultCell, edgeCell);
        UnstructuredGrid<TestCellType, 1, double, 4, 1> gridNew(dim, defaultCell, edgeCell);

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

        typedef UnstructuredTestCell<128, EmptyUnstructuredTestCellAPI> TestCellType;
        TestCellType defaultCell(200);
        TestCellType edgeCell(-1);

        UnstructuredGrid<TestCellType, 1, double, 4, 128> gridOld(dim, defaultCell, edgeCell);
        UnstructuredGrid<TestCellType, 1, double, 4, 128> gridNew(dim, defaultCell, edgeCell);

        Region<1> region;
        region << Streak<1>(Coord<1>(10),   30);
        region << Streak<1>(Coord<1>(40),   60);
        region << Streak<1>(Coord<1>(100), 150);

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

        typedef UnstructuredTestCell<128> TestCellType;
        TestCellType defaultCell(200);
        TestCellType edgeCell(-1);

        UnstructuredGrid<TestCellType, 1, double, 4, 128> gridOld(dim, defaultCell, edgeCell);
        UnstructuredGrid<TestCellType, 1, double, 4, 128> gridNew(dim, defaultCell, edgeCell);

        Region<1> region;
        region << Streak<1>(Coord<1>(10),   30);
        region << Streak<1>(Coord<1>(40),   60);
        region << Streak<1>(Coord<1>(100), 150);

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

    void testSoA()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        const int DIM = 150;
        Coord<1> dim(DIM);

        UnstructuredSoATestCell<1> defaultCell(200);
        UnstructuredSoATestCell<1> edgeCell(-1);

        UnstructuredSoAGrid<UnstructuredSoATestCell<1>, 1, double, 4, 1> gridOld(dim, defaultCell, edgeCell);
        UnstructuredSoAGrid<UnstructuredSoATestCell<1>, 1, double, 4, 1> gridNew(dim, defaultCell, edgeCell);

        Region<1> region;
        // "normal" streak
        region << Streak<1>(Coord<1>(10),   30);
        // loop peeling in first chunk
        region << Streak<1>(Coord<1>(37),   60);
        // loop peeling in last chunk
        region << Streak<1>(Coord<1>(100), 149);

        // weights matrix looks like this:
        // 0
        // 1
        // 1 1
        // 1 1 1
        // ...
        std::map<Coord<2>, double> matrix;
        for (int row = 0; row < DIM; ++row) {
            for (int col = 0; col < row; ++col) {
                matrix[Coord<2>(row, col)] = 1;
            }
        }
        gridOld.setWeights(0, matrix);

        UnstructuredUpdateFunctor<UnstructuredSoATestCell<1> > functor;
        UpdateFunctorHelpers::ConcurrencyNoP concurrencySpec;
        APITraits::SelectThreadedUpdate<UnstructuredSoATestCell<1> >::Value modelThreadingSpec;

        functor(region, gridOld, &gridNew, 0, concurrencySpec, modelThreadingSpec);

        for (Coord<1> coord(0); coord < Coord<1>(150); ++coord.x()) {
            if (((coord.x() >=  10) && (coord.x() <  30)) ||
                ((coord.x() >=  37) && (coord.x() <  60)) ||
                ((coord.x() >= 100) && (coord.x() < 149))) {
                const double sum = coord.x() * 200.0;
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
        Coord<1> dim(DIM);

        UnstructuredSoATestCell<150> defaultCell(200);
        UnstructuredSoATestCell<150> edgeCell(-1);

        UnstructuredSoAGrid<UnstructuredSoATestCell<150>, 1, double, 4, 150> gridOld(dim, defaultCell, edgeCell);
        UnstructuredSoAGrid<UnstructuredSoATestCell<150>, 1, double, 4, 150> gridNew(dim, defaultCell, edgeCell);

        Region<1> region;
        // "normal" streak
        region << Streak<1>(Coord<1>(10),   30);
        // loop peeling in first chunk
        region << Streak<1>(Coord<1>(37),   60);
        // loop peeling in last chunk
        region << Streak<1>(Coord<1>(100), 149);

        // weights matrix looks like this:
        // 0
        // 1
        // 1 1
        // 1 1 1
        // ...
        std::map<Coord<2>, double> matrix;
        for (int row = 0; row < DIM; ++row) {
            for (int col = 0; col < row; ++col) {
                matrix[Coord<2>(row, col)] = 1;
            }
        }
        gridOld.setWeights(0, matrix);

        UnstructuredUpdateFunctor<UnstructuredSoATestCell<150> > functor;
        UpdateFunctorHelpers::ConcurrencyNoP concurrencySpec;
        APITraits::SelectThreadedUpdate<UnstructuredSoATestCell<150> >::Value modelThreadingSpec;

        functor(region, gridOld, &gridNew, 0, concurrencySpec, modelThreadingSpec);

        for (Coord<1> coord(0); coord < Coord<1>(150); ++coord.x()) {
            if (((coord.x() >=  10) && (coord.x() <  30)) ||
                ((coord.x() >=  37) && (coord.x() <  60)) ||
                ((coord.x() >= 100) && (coord.x() < 149))) {
                const double sum = (149 - coord.x()) * 200.0;
                TS_ASSERT_EQUALS(sum, gridNew.get(coord).sum);
            } else {
                TS_ASSERT_EQUALS(0.0, gridNew.get(coord).sum);
            }
        }
#endif
    }
};

}
