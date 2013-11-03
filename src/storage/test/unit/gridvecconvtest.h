#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/gridvecconv.h>
#include <libgeodecomp/storage/soagrid.h>

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
        DisplacedGrid<TestCellType1, Topology1> grid(CoordBox<2>(Coord<2>(), Coord<2>(20, 10)));
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

    void testSoA()
    {
        SoAGrid<TestCellType2, Topology2> gridA(CoordBox<3>(Coord<3>( 0,  0,  0), Coord<3>(40, 20, 30)));
        SoAGrid<TestCellType2, Topology2> gridB(CoordBox<3>(Coord<3>(11, 12, 13), Coord<3>(20, 25, 10)));

        Region<3> region;
        region << Streak<3>(Coord<3>(15, 12, 13), 31)
               << Streak<3>(Coord<3>(11, 19, 14), 31)
               << Streak<3>(Coord<3>(20, 19, 20), 30);

        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCellType2 cell = gridA.get(*i);
            cell.pos = *i;
            cell.testValue = 12.45;
            gridA.set(*i, cell);
        }

        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCellType2 cell = gridB.get(*i);
            TS_ASSERT_EQUALS(Coord<3>(), cell.pos);
            TS_ASSERT_EQUALS(666,        cell.testValue);
        }

        std::vector<char> buf(sizeof(TestCellType2) * region.size());
        GridVecConv::gridToVector(gridA, &buf, region);
        GridVecConv::vectorToGrid(buf, &gridB, region);

        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCellType2 cell = gridB.get(*i);
            TS_ASSERT_EQUALS(*i,    cell.pos);
            TS_ASSERT_EQUALS(12.45, cell.testValue);
        }
    }
};

}
