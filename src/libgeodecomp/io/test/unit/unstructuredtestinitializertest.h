#include <libgeodecomp/config.h>
#include <libgeodecomp/io/unstructuredtestinitializer.h>
#include <libgeodecomp/storage/unstructuredgrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class UnstructuredTestInitializerTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
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
            TS_ASSERT_EQUALS(cell.cycleCounter, expectedCycle);
            TS_ASSERT_EQUALS(cell.isValid, true);
            TS_ASSERT_EQUALS(cell.expectedNeighborWeights.size(), i % 20 + 1);

            // we need to insert ID/weight pairs here so can retrieve them sorted by ID below:
            std::map<int, double> weightsReorderBuffer;

            for (int j = i + 1; j < (i + i % 20 + 2); ++j) {
                int neighbor = j % 100;
                weightsReorderBuffer[neighbor] = neighbor + 0.1;
            }

            FixedArray<int,    100> expectedIDs;
            FixedArray<double, 100> expectedWeights;

            for (std::map<int, double>::iterator i = weightsReorderBuffer.begin(); i != weightsReorderBuffer.end(); ++i) {
                expectedIDs << i->first;
                expectedWeights << i->second;
            }

            TS_ASSERT_EQUALS(expectedWeights, cell.expectedNeighborWeights);
        }

        auto weights = grid.getWeights(0);
        for (std::size_t i = 0; i < 100; ++i) {
            auto sparseRow = weights.getRow(i);

            TS_ASSERT_EQUALS(sparseRow.size(), i % 20 + 1);
            std::size_t start = i + 1;
            std::size_t end = start + i % 20 + 1;
            std::vector<std::pair<int, double> > expectedPairs;
            for (std::size_t j = start; j != end; ++j) {
                int neighbor = static_cast<int>(j % 100);
                expectedPairs << std::make_pair(neighbor, neighbor + 0.1);
            }
            std::sort(expectedPairs.begin(), expectedPairs.end(), pairCompareFirst);

            for (std::size_t j = start; j != end; ++j) {
                std::size_t index = j - start;
                TS_ASSERT_EQUALS(sparseRow[index], expectedPairs[index]);
            }
        }
#endif
    }

private:
    static inline bool pairCompareFirst(const std::pair<int, double>& a, const std::pair<int, double>& b)
    {
        return a.first < b.first;
    }

};

}
