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
    void testBasicUpdates()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredGrid<UnstructuredTestCell<> >TestGridType;
        int startStep = 5;
        int endStep = 60;
        UnstructuredTestInitializer<> init(200, endStep, startStep);
        TestGridType grid1(Coord<1>(200));
        TestGridType grid2(Coord<1>(200));
        init.grid(&grid1);
        init.grid(&grid2);

        UnstructuredNeighborhood<UnstructuredTestCell<>, 1, double, 64, 1> hood(grid1, 0);

        for (int x = 0; x < 200; ++x, ++hood) {
            grid2[Coord<1>(x)].update(hood, 0);
        }

        int expectedCycle1 = startStep * UnstructuredTestCell<>::NANO_STEPS;
        int expectedCycle2 = expectedCycle1 + 1;
        TS_ASSERT_TEST_GRID(TestGridType, grid1, expectedCycle1);
        TS_ASSERT_TEST_GRID(TestGridType, grid2, expectedCycle2);
        // fixme: needs test
#endif
    }
};

}
