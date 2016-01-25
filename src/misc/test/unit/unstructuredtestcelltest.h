#include <libgeodecomp/config.h>
#include <libgeodecomp/io/unstructuredtestinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/misc/unstructuredtestcell.h>
#include <libgeodecomp/storage/unstructuredgrid.h>
#include <libgeodecomp/storage/unstructuredneighborhood.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class UnstructuredTestCellTest : public CxxTest::TestSuite
{
public:
#ifdef LIBGEODECOMP_WITH_CPP14
    typedef UnstructuredTestCell<TestCellHelpers::NoOutput> TestCellType;
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
        grid1[Coord<1>(40)].expectedNeighborWeights[41] = 4711;

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
            if (((x >= 20) && (x <= 40)) || (x >= 120)) {
                flag = false;
            }

            TS_ASSERT_EQUALS(grid2[Coord<1>(x)].valid(), flag);
        }
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
