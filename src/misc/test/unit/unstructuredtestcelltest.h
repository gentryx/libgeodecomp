#include <libgeodecomp/config.h>
#include <libgeodecomp/io/unstructuredtestinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/misc/unstructuredtestcell.h>
#include <libgeodecomp/storage/unstructuredgrid.h>
#include <libgeodecomp/storage/unstructuredneighborhood.h>
#include <libgeodecomp/storage/unstructuredsoagrid.h>
#include <libgeodecomp/storage/updatefunctor.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class UnstructuredTestCellTest : public CxxTest::TestSuite
{
public:
#ifdef LIBGEODECOMP_WITH_CPP14
    typedef UnstructuredTestCell<UnstructuredTestCellHelpers::EmptyAPI, TestCellHelpers::NoOutput> TestCellType;
    typedef UnstructuredGrid<TestCellType> TestGridType;
#endif

    void setUp()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        startStep = 5;
        endStep = 60;
        UnstructuredTestInitializer<TestCellType> init(200, endStep, startStep);
        grid1 = TestGridType(Coord<1>(200));
        grid2 = TestGridType(Coord<1>(200));
        init.grid(&grid1);
        init.grid(&grid2);
#endif
    }

    void testBasicUpdates()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredNeighborhood<TestCellType, 1, double, 64, 1> hood(grid1, 0);

        for (int x = 0; x < 200; ++x, ++hood) {
            grid2[Coord<1>(x)].update(hood, 0);
        }

        int expectedCycle1 = startStep * TestCellType::NANO_STEPS;
        int expectedCycle2 = expectedCycle1 + 1;
        TS_ASSERT_TEST_GRID(TestGridType, grid1, expectedCycle1);
        TS_ASSERT_TEST_GRID(TestGridType, grid2, expectedCycle2);
#endif
    }

    void testWeightsChecking()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredNeighborhood<TestCellType, 1, double, 64, 1> hood(grid1, 0);
        // sabotage weights
        TS_ASSERT_EQUALS(grid1[Coord<1>(40)].expectedNeighborIDs[0], 41);
        grid1[Coord<1>(40)].expectedNeighborWeights[0] = 4711;

        for (int x = 0; x < 200; ++x, ++hood) {
            grid2[Coord<1>(x)].update(hood, 0);
        }

        for (int x = 0; x < 200; ++x, ++hood) {
            bool flag = true;
            if (x == 40) {
                flag = false;
            }

            TS_ASSERT_EQUALS(grid2[Coord<1>(x)].valid(), flag);
        }
#endif
    }

    void testNanoStepChecking()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredNeighborhood<TestCellType, 1, double, 64, 1> hood(grid1, 0);
        // sabotage nano step
        grid1[Coord<1>(40)].cycleCounter += 1;

        for (int x = 0; x < 200; ++x, ++hood) {
            grid2[Coord<1>(x)].update(hood, 0);
        }

        for (int x = 0; x < 200; ++x, ++hood) {
            bool flag = true;
            // determining affected neighbors is slightly more complicated here:
            if ((x >= 30) && (x <= 40)) {
                flag = false;
            }

            TS_ASSERT_EQUALS(grid2[Coord<1>(x)].valid(), flag);
        }
#endif
    }

    void testIDChecking()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredNeighborhood<TestCellType, 1, double, 64, 1> hood(grid1, 0);
        // sabotage ID
        grid1[Coord<1>(40)].id += 1;

        for (int x = 0; x < 200; ++x, ++hood) {
            grid2[Coord<1>(x)].update(hood, 0);
        }

        for (int x = 0; x < 200; ++x, ++hood) {
            bool flag = true;
            // determining affected neighbors is slightly more complicated here:
            if ((x >= 30) && (x <= 39)) {
                flag = false;
            }

            TS_ASSERT_EQUALS(grid2[Coord<1>(x)].valid(), flag);
        }
#endif
    }

    void testValidityChecking()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        UnstructuredNeighborhood<TestCellType, 1, double, 64, 1> hood(grid1, 0);
        // sabotage validity bit
        grid1[Coord<1>(40)].isValid = false;

        for (int x = 0; x < 200; ++x, ++hood) {
            grid2[Coord<1>(x)].update(hood, 0);
        }

        for (int x = 0; x < 200; ++x, ++hood) {
            bool flag = true;
            // determining affected neighbors is slightly more complicated here:
            if ((x >= 30) && (x <= 40)) {
                flag = false;
            }

            TS_ASSERT_EQUALS(grid2[Coord<1>(x)].valid(), flag);
        }
#endif
    }

    void testOperatorEquals()
    {
        UnstructuredTestCell<> cell1(12, 34, true, true);
        UnstructuredTestCell<> cell2 = cell1;

        TS_ASSERT_EQUALS(cell1, cell2);

        cell2 = cell1;
        cell2.id = 0;
        TS_ASSERT_DIFFERS(cell1, cell2);

        cell2 = cell1;
        cell2.cycleCounter = 0;
        TS_ASSERT_DIFFERS(cell1, cell2);

        cell2 = cell1;
        cell2.isValid = false;
        TS_ASSERT_DIFFERS(cell1, cell2);

        cell2 = cell1;
        cell2.isEdgeCell = false;
        TS_ASSERT_DIFFERS(cell1, cell2);

        cell2 = cell1;
        cell2.expectedNeighborIDs << 4711;
        TS_ASSERT_DIFFERS(cell1, cell2);

        cell2 = cell1;
        cell2.expectedNeighborWeights << 1234;
        TS_ASSERT_DIFFERS(cell1, cell2);
    }

    void testSoAVariant1()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA1 TestCellType;
        typedef UnstructuredSoAGrid<TestCellType, 1, double, 32, 1> TestGridType;

        UnstructuredTestInitializer<TestCellType> init(340, endStep, startStep);
        TestGridType grid1 = TestGridType(CoordBox<1>(Coord<1>(), Coord<1>(340)));
        TestGridType grid2 = TestGridType(CoordBox<1>(Coord<1>(), Coord<1>(340)));
        init.grid(&grid1);
        init.grid(&grid2);

        Region<1> region;
        region << Streak<1>(Coord<1>(0), 340);
        UnstructuredUpdateFunctor<TestCellType>()(
            region,
            grid1,
            &grid2,
            0,
            UpdateFunctorHelpers::ConcurrencyNoP(),
            APITraits::SelectThreadedUpdate<TestCellType>::Value());

        int expectedCycle1 = startStep * TestCellType::NANO_STEPS;
        int expectedCycle2 = expectedCycle1 + 1;
        TS_ASSERT_TEST_GRID(TestGridType, grid1, expectedCycle1);
        TS_ASSERT_TEST_GRID(TestGridType, grid2, expectedCycle2);
#endif
    }

    void testSoAVariant2()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA2 TestCellType;
        typedef UnstructuredSoAGrid<TestCellType, 1, double, 8, 1> TestGridType;

        UnstructuredTestInitializer<TestCellType> init(340, endStep, startStep);
        TestGridType grid1 = TestGridType(CoordBox<1>(Coord<1>(), Coord<1>(340)));
        TestGridType grid2 = TestGridType(CoordBox<1>(Coord<1>(), Coord<1>(340)));
        init.grid(&grid1);
        init.grid(&grid2);

        Region<1> region;
        region << Streak<1>(Coord<1>(0), 340);
        UnstructuredUpdateFunctor<TestCellType>()(
            region,
            grid1,
            &grid2,
            0,
            UpdateFunctorHelpers::ConcurrencyNoP(),
            APITraits::SelectThreadedUpdate<TestCellType>::Value());

        int expectedCycle1 = startStep * TestCellType::NANO_STEPS;
        int expectedCycle2 = expectedCycle1 + 1;
        TS_ASSERT_TEST_GRID(TestGridType, grid1, expectedCycle1);
        TS_ASSERT_TEST_GRID(TestGridType, grid2, expectedCycle2);
#endif
    }

    void testSoAVariant3()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        // fixme:

        // typedef UnstructuredTestCellSoA3 TestCellType;
        // typedef UnstructuredSoAGrid<TestCellType, 1, double, 8, 64> TestGridType;

        // UnstructuredTestInitializer<TestCellType> init(340, endStep, startStep);
        // TestGridType grid1 = TestGridType(CoordBox<1>(Coord<1>(), Coord<1>(340)));
        // TestGridType grid2 = TestGridType(CoordBox<1>(Coord<1>(), Coord<1>(340)));
        // init.grid(&grid1);
        // init.grid(&grid2);

        // Region<1> region;
        // region << Streak<1>(Coord<1>(0), 340);
        // UnstructuredUpdateFunctor<TestCellType>()(
        //     region,
        //     grid1,
        //     &grid2,
        //     0,
        //     UpdateFunctorHelpers::ConcurrencyNoP(),
        //     APITraits::SelectThreadedUpdate<TestCellType>::Value());

        // int expectedCycle1 = startStep * TestCellType::NANO_STEPS;
        // int expectedCycle2 = expectedCycle1 + 1;
        // TS_ASSERT_TEST_GRID(TestGridType, grid1, expectedCycle1);
        // TS_ASSERT_TEST_GRID(TestGridType, grid2, expectedCycle2);
#endif
    }

    void testSoAVariant4()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        // fixme:

        // typedef UnstructuredTestCellSoA4 TestCellType;
        // typedef UnstructuredSoAGrid<TestCellType, 1, double, 16, 32> TestGridType;

        // UnstructuredTestInitializer<TestCellType> init(340, endStep, startStep);
        // TestGridType grid1 = TestGridType(CoordBox<1>(Coord<1>(), Coord<1>(340)));
        // TestGridType grid2 = TestGridType(CoordBox<1>(Coord<1>(), Coord<1>(340)));
        // init.grid(&grid1);
        // init.grid(&grid2);

        // Region<1> region;
        // region << Streak<1>(Coord<1>(0), 340);
        // UnstructuredUpdateFunctor<TestCellType>()(
        //     region,
        //     grid1,
        //     &grid2,
        //     0,
        //     UpdateFunctorHelpers::ConcurrencyNoP(),
        //     APITraits::SelectThreadedUpdate<TestCellType>::Value());

        // int expectedCycle1 = startStep * TestCellType::NANO_STEPS;
        // int expectedCycle2 = expectedCycle1 + 1;
        // TS_ASSERT_TEST_GRID(TestGridType, grid1, expectedCycle1);
        // TS_ASSERT_TEST_GRID(TestGridType, grid2, expectedCycle2);
#endif
    }

private:

#ifdef LIBGEODECOMP_WITH_CPP14
    int startStep;
    int endStep;
    TestGridType grid1;
    TestGridType grid2;
#endif
};

}
