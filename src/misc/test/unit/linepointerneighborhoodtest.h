#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/misc/linepointerneighborhood.h>
#include <libgeodecomp/misc/linepointerupdatefunctor.h>

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
        const int *pointers[] = {
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
        const int *pointers[] = {
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
        long endX = dim.x();

        typedef Grid<TestCell<2>, TestCell<2>::Topology> GridType;
        GridType gridOld(dim);
        GridType gridNew(dim);
        TestInitializer<TestCell<2> > init(dim);
        init.grid(&gridOld);
        init.grid(&gridNew);
        CoordBox<2> box = gridOld.boundingBox();

        TS_ASSERT_TEST_GRID(GridType, gridOld, 0);
        TS_ASSERT_TEST_GRID(GridType, gridNew, 0);

        for (int y = 0; y < dim.y(); ++y) {
            Coord<2> c(0, y);
            const TestCell<2> *pointers[] = {
                &gridOld[Coord<2>(c.x() - 1, c.y() - 1)],
                &gridOld[Coord<2>(c.x() + 0, c.y() - 1)],
                &gridOld[Coord<2>(endX,      c.y() - 1)],
                &gridOld[Coord<2>(c.x() - 1, c.y() + 0)],
                &gridOld[Coord<2>(c.x() + 0, c.y() + 0)],
                &gridOld[Coord<2>(endX,      c.y() + 0)],
                &gridOld[Coord<2>(c.x() - 1, c.y() + 1)],
                &gridOld[Coord<2>(c.x() + 0, c.y() + 1)],
                &gridOld[Coord<2>(endX,      c.y() + 1)]
            };
            LinePointerUpdateFunctor<TestCell<2> >()(Streak<2>(c, endX), box, pointers, &gridNew[c], 0);

        }

        TS_ASSERT_TEST_GRID(GridType, gridOld, 0);
        TS_ASSERT_TEST_GRID(GridType, gridNew, 1);
    }

    void test3DTorusTopologyTotal()
    {
        Coord<3> dim(13, 12, 11);
        long endX = dim.x();

        typedef Grid<TestCell<3>, TestCell<3>::Topology> GridType;
        GridType gridOld(dim);
        GridType gridNew(dim);
        TestInitializer<TestCell<3> > init(dim);
        init.grid(&gridOld);
        init.grid(&gridNew);
        CoordBox<3> box = gridOld.boundingBox();

        TS_ASSERT_TEST_GRID(GridType, gridOld, 0);
        TS_ASSERT_TEST_GRID(GridType, gridNew, 0);

        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                Coord<3> c(0, y, z);
                const TestCell<3> *pointers[] = {
                    &gridOld[Coord<3>(c.x() - 1, c.y() - 1, c.z() - 1)],
                    &gridOld[Coord<3>(c.x() + 0, c.y() - 1, c.z() - 1)],
                    &gridOld[Coord<3>(endX,      c.y() - 1, c.z() - 1)],
                    &gridOld[Coord<3>(c.x() - 1, c.y() + 0, c.z() - 1)],
                    &gridOld[Coord<3>(c.x() + 0, c.y() + 0, c.z() - 1)],
                    &gridOld[Coord<3>(endX,      c.y() + 0, c.z() - 1)],
                    &gridOld[Coord<3>(c.x() - 1, c.y() + 1, c.z() - 1)],
                    &gridOld[Coord<3>(c.x() + 0, c.y() + 1, c.z() - 1)],
                    &gridOld[Coord<3>(endX,      c.y() + 1, c.z() - 1)],

                    &gridOld[Coord<3>(c.x() - 1, c.y() - 1, c.z() + 0)],
                    &gridOld[Coord<3>(c.x() + 0, c.y() - 1, c.z() + 0)],
                    &gridOld[Coord<3>(endX,      c.y() - 1, c.z() + 0)],
                    &gridOld[Coord<3>(c.x() - 1, c.y() + 0, c.z() + 0)],
                    &gridOld[Coord<3>(c.x() + 0, c.y() + 0, c.z() + 0)],
                    &gridOld[Coord<3>(endX,      c.y() + 0, c.z() + 0)],
                    &gridOld[Coord<3>(c.x() - 1, c.y() + 1, c.z() + 0)],
                    &gridOld[Coord<3>(c.x() + 0, c.y() + 1, c.z() + 0)],
                    &gridOld[Coord<3>(endX,      c.y() + 1, c.z() + 0)],

                    &gridOld[Coord<3>(c.x() - 1, c.y() - 1, c.z() + 1)],
                    &gridOld[Coord<3>(c.x() + 0, c.y() - 1, c.z() + 1)],
                    &gridOld[Coord<3>(endX,      c.y() - 1, c.z() + 1)],
                    &gridOld[Coord<3>(c.x() - 1, c.y() + 0, c.z() + 1)],
                    &gridOld[Coord<3>(c.x() + 0, c.y() + 0, c.z() + 1)],
                    &gridOld[Coord<3>(endX,      c.y() + 0, c.z() + 1)],
                    &gridOld[Coord<3>(c.x() - 1, c.y() + 1, c.z() + 1)],
                    &gridOld[Coord<3>(c.x() + 0, c.y() + 1, c.z() + 1)],
                    &gridOld[Coord<3>(endX,      c.y() + 1, c.z() + 1)]
                };
                LinePointerUpdateFunctor<TestCell<3> >()(Streak<3>(c, endX), box, pointers, &gridNew[c], 0);
            }
        }

        TS_ASSERT_TEST_GRID(GridType, gridOld, 0);
        TS_ASSERT_TEST_GRID(GridType, gridNew, 1);
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
