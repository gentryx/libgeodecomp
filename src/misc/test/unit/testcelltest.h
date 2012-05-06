#include <cxxtest/TestSuite.h>
#include <sstream>
#include "../../grid.h"
#include "../../testcell.h"
#include "../../testhelper.h"

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class TestCellTest : public CxxTest::TestSuite 
{
private:
    Grid<TestCell<2> > grid;
    int width;
    int height;
    std::ostringstream nirvana;
    std::ostream *oldStream;

public:
    void setUp() 
    {
        oldStream = TestCellBase::stream;
        TestCellBase::stream = &nirvana;
        width = 4;
        height = 3;
        grid = Grid<TestCell<2> >(Coord<2>(width, height));
        grid[Coord<2>(-1, -1)] = TestCell<2>(Coord<2>(-1, -1), 
                                          Coord<2>(width, height));
        for (int x = 0; x < width; x++) 
            for (int y = 0; y < height; y++) 
                grid[Coord<2>(x, y)] = 
                    TestCell<2>(Coord<2>(x, y), Coord<2>(width, height), 0);
    }

    void tearDown()
    {
        TestCellBase::stream = oldStream;
    }

    void testDefaultConstructor()
    {
        TS_ASSERT(!TestCell<2>().valid());
    }

    void testSetUp()
    {
        TS_ASSERT_TEST_GRID(Grid<TestCell<2> >, grid, 0);
    }
    
    void testUpdate()
    {
        TS_ASSERT_TEST_GRID(Grid<TestCell<2> >, grid, 0);
        update();
        TS_ASSERT_TEST_GRID(Grid<TestCell<2> >, grid, 1);
    }

    void testMultipleUpdate()
    {
        for (int i = 0; i < 100; i++) {
            update(i % TestCell<2>::nanoSteps());
            TS_ASSERT_TEST_GRID(Grid<TestCell<2> >, grid, i + 1);
        }
    }

    void testUpdateUnsyncedNanoStep()
    {
        update(3);
        TS_ASSERT(!grid[0][0].isValid);
    }

    void testUpdateOutOfBoundsNanoStep()
    {
        update(100);
        TS_ASSERT(!grid[0][0].isValid);
    }

    void testUpdateBadEdgeCellInner()
    {
        grid[1][1] = TestCell<2>(Coord<2>(-1, -1), Coord<2>(width, height));
        update();
        TS_ASSERT(!grid[0][0].isValid);
    }

    void testUpdateBadEdgeCellOuter()
    {
        grid[Coord<2>(-1, -1)] = 
            TestCell<2>(Coord<2>(1, 1), Coord<2>(width, height));
        update();    
        TS_ASSERT(!grid[0][0].isValid);
    }

    void testUpdateInvalidNeighbor()
    {
        grid[1][0] = TestCell<2>();
        update();
        TS_ASSERT(!grid[0][0].isValid);
    }

    void testUpdateBadCycle()
    {
        grid[0][0].cycleCounter = 1;
        update();
        TS_ASSERT(!grid[0][0].isValid);
    }

    void testUpdateBadRect()
    {
        grid[0][0] = TestCell<2>(Coord<2>(0, 0), Coord<2>(123, 456));
        update();
        TS_ASSERT(!grid[0][0].isValid);
    }

    void testUpdateBadPos()
    {
        grid[0][0].pos = Coord<2>(1, 0);
        update();
        TS_ASSERT(!grid[0][0].isValid);
    }

    typedef Grid<TestCell<3>, TestCell<3>::Topology> Grid3D;

    void test3D1()
    {
        int width = 4;
        int height = 7;
        int depth = 5;
        Coord<3> dim(width, height, depth);

        Grid3D gridA(dim);
        Grid3D gridB(dim);
        gridA.getEdgeCell() = 
            TestCell<3>(Coord<3>::diagonal(-1), dim);
        gridA.getEdgeCell().isEdgeCell = true;

        gridB.getEdgeCell() = 
            gridA.getEdgeCell();

        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); y++) {
                for (int x = 0; x < dim.x(); x++) {
                    Coord<3> pos(x, y, z);
                    gridA[pos] = TestCell<3>(pos, dim, 0);
                }
            }
        }
        TS_ASSERT_TEST_GRID(Grid3D, gridA, 0);

        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); y++) {
                for (int x = 0; x < dim.x(); x++) {
                    Coord<3> pos(x, y, z);
                    gridB[pos].update(gridA.getNeighborhood(pos), 0);
                }
            }
        }
        TS_ASSERT_TEST_GRID(Grid3D, gridB, 1);

        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); y++) {
                for (int x = 0; x < dim.x(); x++) {
                    Coord<3> pos(x, y, z);
                    gridA[pos].update(gridB.getNeighborhood(pos), 1);
                }
            }
        }
        TS_ASSERT_TEST_GRID(Grid3D, gridA, 2);
    }
 
    void test3D2()
    {
        int width = 4;
        int height = 7;
        int depth = 9;
        Coord<3> dim(width, height, depth);

        Grid3D gridA(dim);
        Grid3D gridB(dim);
        gridA.getEdgeCell() = 
            TestCell<3>(Coord<3>::diagonal(-1), dim);
        gridA.getEdgeCell().isEdgeCell = true;
        gridB.getEdgeCell() = 
            gridA.getEdgeCell();

        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); y++) {
                for (int x = 0; x < dim.x(); x++) {
                    Coord<3> pos(x, y, z);
                    gridA[pos] = TestCell<3>(pos, dim, 0);
                }
            }
        }

        TS_ASSERT_TEST_GRID(Grid3D, gridA, 0);

        // use a wrong z-value to freak out the update step
        int badZ = 2;
        int actualZ = 5;
        for (int y = 0; y < dim.y(); y++) 
            for (int x = 0; x < dim.x(); x++) 
                gridA[Coord<3>(x, y, actualZ)] = 
                    TestCell<3>(Coord<3>(x, y, badZ), dim, 0);

        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); y++) {
                for (int x = 0; x < dim.x(); x++) {
                    Coord<3> pos(x, y, z);
                    gridB[pos].update(gridA.getNeighborhood(pos), 0);
                }
            }
        }

        // verify that cells surrounding the rougue cells have been
        // invalidated:
        CoordBox<3> box(Coord<3>(0, 0, actualZ - 1), 
                        Coord<3>(dim.x(), dim.y(), 3));
        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            TS_ASSERT(!gridB[*i].valid());
        }
    }

    void update(unsigned nanoStep = 0)
    {
        Grid<TestCell<2> > newGrid(Coord<2>(width, height));
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                CoordMap<TestCell<2> > map(Coord<2>(x, y), &grid);
                newGrid[Coord<2>(x, y)].update(map, nanoStep);
            }
        }
        newGrid[Coord<2>(-1, -1)] = grid[Coord<2>(-1, -1)];
        grid = newGrid;
    }

};

};
