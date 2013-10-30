#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/gridvecconv.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class GridVecConvTest : public CxxTest::TestSuite
{
public:
    typedef TestCell<2> TestCellType1;
    typedef TestCellSoA TestCellType2;

    typedef APITraits::SelectTopology<TestCellType1>::Value Topology1;
    typedef APITraits::SelectTopology<TestCellType2>::Value Topology2;

    void testBasic()
    {
        Grid<TestCellType1, Topology1> grid(Coord<2>(20, 10));
        Region<2> region;
        region << Streak<2>(Coord<2>(2, 1), 10)
               << Streak<2>(Coord<2>(1, 2), 11)
               << Streak<2>(Coord<2>(0, 3), 12);
        std::vector<TestCellType1> buffer(region.size());
        int counter = 4711;

        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            grid[*i].testValue = counter;
            ++counter;
        }

        GridVecConv::gridToVector(grid, &buffer, region);

        for (size_t i = 0; i < region.size(); ++i) {
            TS_ASSERT_EQUALS(i + 4711, buffer[i].testValue);
        }

        for (size_t i = 0; i < region.size(); ++i) {
            buffer[i].testValue = 666 + i;
        }

        for (int y = 0; y < grid.getDimensions().y(); ++y) {
            for (int x = 0; x < grid.getDimensions().x(); ++x) {
                grid[Coord<2>(x, y)].testValue = -123;
            }
        }
        GridVecConv::vectorToGrid(buffer, &grid, region);
        counter = 666;

        for (int y = 0; y < grid.getDimensions().y(); ++y) {
            for (int x = 0; x < grid.getDimensions().x(); ++x) {
                Coord<2> c(x, y);

                if (region.count(c)) {
                    TS_ASSERT_EQUALS(counter, grid[c].testValue);
                    ++counter;
                } else {
                    TS_ASSERT_EQUALS(-123, grid[c].testValue);
                }
            }
        }
    }
};

}
