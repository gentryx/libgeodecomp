#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/misc/linepointerneighborhood.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class LinePointerNeighborhoodTest : public CxxTest::TestSuite 
{
public:
    void testBasic()
    {
        Coord<2> dim(7, 5);
        Grid<int> grid(dim, 4711, -1);
        fillGrid(&grid);

        long x = 0;

        Coord<2> c(1, 3);
        int *pointers[] = {
            &grid[c + Coord<2>(-1, -1)],
            &grid[c + Coord<2>( 0, -1)],
            &grid[c + Coord<2>( 1, -1)],
            &grid[c + Coord<2>(-1,  0)],
            &grid[c + Coord<2>( 0,  0)],
            &grid[c + Coord<2>( 1,  0)],
            &grid[c + Coord<2>(-1,  1)],
            &grid[c + Coord<2>( 0,  1)],
            &grid[c + Coord<2>( 1,  1)]
        };
        LinePointerNeighborhood<int, Stencils::Moore<2, 1>, false, false, false, false, false, false> hood(pointers, &x);

        // CENTER
        TS_ASSERT_EQUALS(1031, (hood[FixedCoord< 0,  0>()]));
        // WEST
        TS_ASSERT_EQUALS(1030, (hood[FixedCoord<-1,  0>()]));
        // TOP
        TS_ASSERT_EQUALS(1021, (hood[FixedCoord< 0, -1>()]));
        // BOTTOM EAST
        TS_ASSERT_EQUALS(1042, (hood[FixedCoord< 1,  1>()]));

        x = 1;
        // CENTER
        TS_ASSERT_EQUALS(1032, (hood[FixedCoord< 0,  0>()]));
        // WEST
        TS_ASSERT_EQUALS(1031, (hood[FixedCoord<-1,  0>()]));
        // TOP
        TS_ASSERT_EQUALS(1022, (hood[FixedCoord< 0, -1>()]));
        // BOTTOM EAST
        TS_ASSERT_EQUALS(1043, (hood[FixedCoord< 1,  1>()]));
    }

    void test2DCubeTopologyMiddle()
    {
        Coord<2> dim(7, 5);
        int endX = dim.x();
        Grid<int> grid(dim, 4711, -1);
        fillGrid(&grid);

        long x = 0;
        
        Coord<2> c(0, 3);
        int *pointers[] = {
            &grid[Coord<2>(c.x() - 1, c.y() - 1)],
            &grid[Coord<2>(c.x() + 0, c.y() - 1)],
            &grid[Coord<2>(endX,      c.y() - 1)],
            &grid[Coord<2>(c.x() - 1, c.y() + 0)],
            &grid[Coord<2>(c.x() + 0, c.y() + 0)],
            &grid[Coord<2>(endX,      c.y() + 0)],
            &grid[Coord<2>(c.x() - 1, c.y() + 1)],
            &grid[Coord<2>(c.x() + 0, c.y() + 1)],
            &grid[Coord<2>(endX,      c.y() + 1)]
        };
        
        // start at western boundary:
        {
            LinePointerNeighborhood<int, Stencils::Moore<2, 1>, true, false, false, false, false, false> hood(pointers, &x);
            
            TS_ASSERT_EQUALS(-1,   (hood[FixedCoord<-1, -1>()]));
            TS_ASSERT_EQUALS(1020, (hood[FixedCoord< 0, -1>()]));
            TS_ASSERT_EQUALS(1021, (hood[FixedCoord< 1, -1>()]));

            TS_ASSERT_EQUALS(-1,   (hood[FixedCoord<-1,  0>()]));
            TS_ASSERT_EQUALS(1030, (hood[FixedCoord< 0,  0>()]));
            TS_ASSERT_EQUALS(1031, (hood[FixedCoord< 1,  0>()]));

            TS_ASSERT_EQUALS(-1,   (hood[FixedCoord<-1,  1>()]));
            TS_ASSERT_EQUALS(1040, (hood[FixedCoord< 0,  1>()]));
            TS_ASSERT_EQUALS(1041, (hood[FixedCoord< 1,  1>()]));
        }

        // sweep through line:
        {
            for (x = 1; x < endX - 1; ++x) {
                LinePointerNeighborhood<int, Stencils::Moore<2, 1>, false, false, false, false, false, false> hood(pointers, &x);
            
                TS_ASSERT_EQUALS(1020 + x - 1, (hood[FixedCoord<-1, -1>()]));
                TS_ASSERT_EQUALS(1020 + x + 0, (hood[FixedCoord< 0, -1>()]));
                TS_ASSERT_EQUALS(1020 + x + 1, (hood[FixedCoord< 1, -1>()]));

                TS_ASSERT_EQUALS(1030 + x - 1, (hood[FixedCoord<-1,  0>()]));
                TS_ASSERT_EQUALS(1030 + x + 0, (hood[FixedCoord< 0,  0>()]));
                TS_ASSERT_EQUALS(1030 + x + 1, (hood[FixedCoord< 1,  0>()]));

                TS_ASSERT_EQUALS(1040 + x - 1, (hood[FixedCoord<-1,  1>()]));
                TS_ASSERT_EQUALS(1040 + x + 0, (hood[FixedCoord< 0,  1>()]));
                TS_ASSERT_EQUALS(1040 + x + 1, (hood[FixedCoord< 1,  1>()]));
            }
        }

        // end at eastern boundary:
        {
            LinePointerNeighborhood<int, Stencils::Moore<2, 1>, false, true, false, false, false, false> hood(pointers, &x);

            TS_ASSERT_EQUALS(1025, (hood[FixedCoord<-1, -1>()]));
            TS_ASSERT_EQUALS(1026, (hood[FixedCoord< 0, -1>()]));
            TS_ASSERT_EQUALS(-1,   (hood[FixedCoord< 1, -1>()]));

            TS_ASSERT_EQUALS(1035, (hood[FixedCoord<-1,  0>()]));
            TS_ASSERT_EQUALS(1036, (hood[FixedCoord< 0,  0>()]));
            TS_ASSERT_EQUALS(-1,   (hood[FixedCoord< 1,  0>()]));

            TS_ASSERT_EQUALS(1045, (hood[FixedCoord<-1,  1>()]));
            TS_ASSERT_EQUALS(1046, (hood[FixedCoord< 0,  1>()]));
            TS_ASSERT_EQUALS(-1,   (hood[FixedCoord< 1,  1>()]));
        }
    }

    void test2DCubeTopologyTotal()
    {
        Coord<2> dim(31, 20);
        int endX = dim.x();

        typedef Grid<TestCell<2> >GridType;
        GridType gridOld(dim);
        GridType gridNew(dim);
        TestInitializer<2> init(dim);
        init.grid(&gridOld);
        init.grid(&gridNew);

        TS_ASSERT_TEST_GRID(GridType, gridOld, 0);
        TS_ASSERT_TEST_GRID(GridType, gridNew, 0);

        for (int y = 0; y < dim.y(); ++y) {
            // fixme: use LinePointerNeighborhood here!
            for (int x = 0; x < dim.x(); ++x) {
                Coord<2> c(x, y);
                gridNew[c].update(gridOld.getNeighborhood(c), 0);
            }
        }

        TS_ASSERT_TEST_GRID(GridType, gridOld, 0);
        TS_ASSERT_TEST_GRID(GridType, gridNew, 1);

        // long x = 0;
        
        // Coord<2> c(0, 0);
        // int *pointers[] = {
        //     &grid[Coord<2>(c.x() - 1, c.y() - 1)],
        //     &grid[Coord<2>(c.x() + 0, c.y() - 1)],
        //     &grid[Coord<2>(endX,      c.y() - 1)],
        //     &grid[Coord<2>(c.x() - 1, c.y() + 0)],
        //     &grid[Coord<2>(c.x() + 0, c.y() + 0)],
        //     &grid[Coord<2>(endX,      c.y() + 0)],
        //     &grid[Coord<2>(c.x() - 1, c.y() + 1)],
        //     &grid[Coord<2>(c.x() + 0, c.y() + 1)],
        //     &grid[Coord<2>(endX,      c.y() + 1)]
        // };
    }
private:
    template<class GRID_TYPE>
    void fillGrid(GRID_TYPE *grid)
    {
        Coord<2> dim = grid->getDimensions();
        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                (*grid)[Coord<2>(x, y)] = 1000 + y * 10 + x;
            }
        }
    }
};

}
