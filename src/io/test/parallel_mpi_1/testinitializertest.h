#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/misc/testhelper.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class TestTestInitializer : public CxxTest::TestSuite
{
private:
    TestInitializer<TestCell<2> > init;
    unsigned startCycle;

public:

    void setUp()
    {
        init = TestInitializer<TestCell<2> >(Coord<2>(10, 11), 12, 9);
        startCycle = 9 * TestCell<2>::nanoSteps();
    }

    void testGridRectangle()
    {
        TS_ASSERT_EQUALS(init.gridBox(),
                         CoordBox<2>(Coord<2>(0, 0), Coord<2>(10, 11)));
    }

    void testGrid1()
    {
        CoordBox<2> rect(Coord<2>(3, 4), Coord<2>(7, 5));
        DisplacedGrid<TestCell<2> > grid(rect);
        init.grid(&grid);
        TS_ASSERT_EQUALS(grid.getDimensions().x(), rect.dimensions.x());
        TS_ASSERT_EQUALS(grid.getDimensions().y(), rect.dimensions.y());
        TS_ASSERT_EQUALS(grid[Coord<2>(3, 4)].pos, Coord<2>(3, 4));
        TS_ASSERT_TEST_GRID(DisplacedGrid<TestCell<2> >, grid, startCycle);

        DisplacedGrid<TestCell<2> > newGrid = grid;
        for (unsigned x = 1; x < unsigned(rect.dimensions.x() - 1); x++) {
            for (unsigned y = 1; y < unsigned(rect.dimensions.y() - 1); y++) {
                Coord<2> pos(x, y);
                newGrid[pos + rect.origin].update(
                    CoordMap<TestCell<2> >(pos, grid.vanillaGrid()),
                    startCycle % TestCell<2>::nanoSteps());
            }
        }

        for (unsigned x = 1; x < unsigned(rect.dimensions.x() - 1); x++) {
            for (unsigned y = 1; y < unsigned(rect.dimensions.y() - 1); y++) {
                TS_ASSERT(newGrid[Coord<2>(x, y)].valid());
            }
        }
    }

    void testGrid2()
    {
        unsigned width = 23;
        unsigned height = 32;
        Initializer<TestCell<2> > *init =
            new TestInitializer<TestCell<2> >(Coord<2>(width, height));
        Grid<TestCell<2> > gridOld(Coord<2>(width, height));
        init->grid(&gridOld);
        delete init;
        Grid<TestCell<2> > gridNew(Coord<2>(width, height));
        TS_ASSERT_TEST_GRID(Grid<TestCell<2> >, gridOld, 0);


        // update everything...
        for (unsigned x = 0; x < width; x++) {
            for (unsigned y = 0; y < height; y++) {
                Coord<2> pos(x, y);
                gridNew[pos].update(CoordMap<TestCell<2> >(pos, &gridOld), 0);
            }
        }
        gridNew[Coord<2>(-1, -1)] = gridOld[Coord<2>(-1, -1)];
        TS_ASSERT_TEST_GRID(Grid<TestCell<2> >, gridNew, 1);

        // ...twice
        for (unsigned x = 0; x < width; x++) {
            for (unsigned y = 0; y < height; y++) {
                Coord<2> pos(x, y);
                gridOld[pos].update(CoordMap<TestCell<2> >(pos, &gridNew), 1);
            }
        }
        gridOld[Coord<2>(-1, -1)] = gridNew[Coord<2>(-1, -1)];
        TS_ASSERT_TEST_GRID(Grid<TestCell<2> >, gridOld, 2);
    }

    void testGridWidth()
    {
        TS_ASSERT_EQUALS((int)init.gridDimensions().x(), 10);
    }

    void testGridHeight()
    {
        TS_ASSERT_EQUALS((int)init.gridDimensions().y(), 11);
    }

    void testMaxSteps()
    {
        TS_ASSERT_EQUALS((int)init.maxSteps(), 12);
    }

    void testStartStep()
    {
        TS_ASSERT_EQUALS((int)init.startStep(), 9);
    }

    void testDump()
    {
        TS_ASSERT_EQUALS(init.dump(), "foo");
    }
};

};
