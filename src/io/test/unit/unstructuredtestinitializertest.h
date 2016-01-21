#include <libgeodecomp/io/unstructuredtestinitializer.h>
#include <libgeodecomp/storage/unstructuredgrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class UnstructuredTestInitializerTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        UnstructuredTestInitializer<> initializer(
            100,
            50,
            10);

        TS_ASSERT_EQUALS(Coord<1>(100), initializer.gridDimensions());
        TS_ASSERT_EQUALS(50, initializer.maxSteps());
        TS_ASSERT_EQUALS(10, initializer.startStep());

        UnstructuredGrid<UnstructuredTestCell<> > grid(initializer.gridDimensions());
        initializer.grid(&grid);
        int expectedCycle = 10 * APITraits::SelectNanoSteps<UnstructuredTestCell<> >::VALUE;

        for (int i = 0; i < 100; ++i) {
            UnstructuredTestCell<> cell = grid.get(Coord<1>(i));

            TS_ASSERT_EQUALS(cell.id, i);
            TS_ASSERT_EQUALS(cell.cycle, expectedCycle);
            TS_ASSERT_EQUALS(cell.isValid, true);
            TS_ASSERT_EQUALS(cell.expectedNeighborWeights.size(), i + 1);

            std::map<int, double> expected;
            for (int j = i + 1; j < (2 * i + 2); ++j) {
                int neighbor = j % 100;
                expected[neighbor] = neighbor + 0.1;
            }
            TS_ASSERT_EQUALS(expected, cell.expectedNeighborWeights);
        }

        auto weights = grid.getAdjacency(0);
        for (int i = 0; i < 100; ++i) {
            auto sparseRow = weights.getRow(i);

            // fixme:
            // TS_ASSERT_EQUALS(sparseRow.size(), i + 1);
            // fixme: check weights
        }
    }
};

}
