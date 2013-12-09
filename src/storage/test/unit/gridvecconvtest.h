#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/gridvecconv.h>
#include <libgeodecomp/storage/soagrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

/**
 * Test model for use with Boost.Serialization
 */
class IrregularCell
{
public:
    class API : public APITraits::HasBoostSerialization
    {};

    IrregularCell(
        int temperature = 0,
        IrregularCell *sublevelNW = 0,
        IrregularCell *sublevelNE = 0,
        IrregularCell *sublevelSW = 0,
        IrregularCell *sublevelSE = 0) :
        sublevelNW(sublevelNW),
        sublevelNE(sublevelNE),
        sublevelSW(sublevelSW),
        sublevelSE(sublevelSE),
        temperature(temperature)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep)
    {
    }

    template<typename ARCHIVE>
    void serialize(ARCHIVE& archive, int version)
    {
        archive & sublevelNW;
        archive & sublevelNE;
        archive & sublevelSW;
        archive & sublevelSE;
        archive & temperature;
    }

    int size() const
    {
        int ret = 1;

        if (sublevelNW) {
            ret += sublevelNW->size();
        }
        if (sublevelNE) {
            ret += sublevelNE->size();
        }
        if (sublevelSW) {
            ret += sublevelSW->size();
        }
        if (sublevelSE) {
            ret += sublevelSE->size();
        }

        return ret;
    }

    IrregularCell *sublevelNW;
    IrregularCell *sublevelNE;
    IrregularCell *sublevelSW;
    IrregularCell *sublevelSE;
    int temperature;
};

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

        std::vector<char> buf(SoAGrid<TestCellType2, Topology2>::AGGREGATED_MEMBER_SIZE * region.size());
        GridVecConv::gridToVector(gridA, &buf, region);
        GridVecConv::vectorToGrid(buf, &gridB, region);

        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            TestCellType2 cell = gridB.get(*i);
            TS_ASSERT_EQUALS(*i,    cell.pos);
            TS_ASSERT_EQUALS(12.45, cell.testValue);
        }
    }

    void testBoostSerialization()
    {
#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
        CoordBox<2> box(Coord<2>(10, 10), Coord<2>(30, 20));
        DisplacedGrid<IrregularCell> gridA(box);
        DisplacedGrid<IrregularCell> gridB(box);

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            gridA[*i].temperature = i->x() * 100 + i->y() + 90000;
        }

        gridA[Coord<2>(12, 11)].sublevelNW = new IrregularCell(1);
        gridA[Coord<2>(12, 11)].sublevelSE = new IrregularCell(
            2,
            0,
            new IrregularCell(3),
            0,
            0);

        gridA[Coord<2>(20, 19)].sublevelNE = new IrregularCell(
            4,
            new IrregularCell(5),
            new IrregularCell(6),
            new IrregularCell(7),
            new IrregularCell(8, new IrregularCell(9)));

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            TS_ASSERT_EQUALS(gridB[*i].temperature, 0);
            TS_ASSERT_EQUALS(gridB[*i].size(), 1);
        }

        Region<2> region;
        region << Streak<2>(Coord<2>(10, 11), 15)
               << Streak<2>(Coord<2>(10, 19), 40);

        std::vector<char> buffer;
        GridVecConv::gridToVector(gridA, &buffer, region);
        GridVecConv::vectorToGrid(buffer, &gridB, region);

        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            int expected = i->x() * 100 + i->y() + 90000;
            TS_ASSERT_EQUALS(gridB[*i].temperature, expected);
        }

        TS_ASSERT_EQUALS(gridB[Coord<2>(12, 11)].size(), 4);
        IrregularCell *null = 0;
        IrregularCell *testCell;

        testCell = &gridB[Coord<2>(12, 11)];
        TS_ASSERT_EQUALS(testCell->temperature, 91211);
        TS_ASSERT_DIFFERS(testCell->sublevelNW, null);
        TS_ASSERT_EQUALS( testCell->sublevelNE, null);
        TS_ASSERT_EQUALS( testCell->sublevelSW, null);
        TS_ASSERT_DIFFERS(testCell->sublevelSE, null);

        testCell = gridB[Coord<2>(12, 11)].sublevelNW;
        TS_ASSERT_EQUALS(testCell->temperature, 1);
        TS_ASSERT_EQUALS( testCell->sublevelNW, null);
        TS_ASSERT_EQUALS( testCell->sublevelNE, null);
        TS_ASSERT_EQUALS( testCell->sublevelSW, null);
        TS_ASSERT_EQUALS( testCell->sublevelSE, null);

        testCell = gridB[Coord<2>(12, 11)].sublevelSE;
        TS_ASSERT_EQUALS(testCell->temperature, 2);
        TS_ASSERT_EQUALS( testCell->sublevelNW, null);
        TS_ASSERT_DIFFERS(testCell->sublevelNE, null);
        TS_ASSERT_EQUALS( testCell->sublevelSW, null);
        TS_ASSERT_EQUALS( testCell->sublevelSE, null);

        testCell = gridB[Coord<2>(12, 11)].sublevelSE->sublevelNE;
        TS_ASSERT_EQUALS(testCell->temperature, 3);
        TS_ASSERT_EQUALS( testCell->sublevelNW, null);
        TS_ASSERT_EQUALS( testCell->sublevelNE, null);
        TS_ASSERT_EQUALS( testCell->sublevelSW, null);
        TS_ASSERT_EQUALS( testCell->sublevelSE, null);

        TS_ASSERT_EQUALS(gridB[Coord<2>(20, 19)].size(), 7);
#endif
    }
};

}
