#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/topologies.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/grid.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class TopologiesTest : public CxxTest::TestSuite 
{
public:
    void testNormalizeCoordAndOutOfBoundsAndNormalizeEdges()
    {
        typedef TopologiesHelpers::RawTopology<3, false, false, false> Topo1;
        typedef TopologiesHelpers::RawTopology<3, true,  false, false> Topo2;
        typedef TopologiesHelpers::RawTopology<3, true,  true,  true > Topo3;

        Coord<3> c;
        Coord<3> d;
        Coord<3> dim(10, 20, 30);

        c = Coord<3>(1, 5, 7);
        d = TopologiesHelpers::NormalizeCoord<Topo1>()(c, dim);
        TS_ASSERT_EQUALS(c, d);

        c = Coord<3>(10, 10, 10);
        d = TopologiesHelpers::NormalizeCoord<Topo1>()(c, dim);
        TS_ASSERT_EQUALS(Coord<3>::diagonal(-1), d);

        c = Coord<3>( 5, -1, 10);
        d = TopologiesHelpers::NormalizeCoord<Topo1>()(c, dim);
        TS_ASSERT_EQUALS(Coord<3>::diagonal(-1), d);

        c = Coord<3>( 5,  5, 30);
        d = TopologiesHelpers::NormalizeCoord<Topo1>()(c, dim);
        TS_ASSERT_EQUALS(Coord<3>::diagonal(-1), d);

        c = Coord<3>(1, 5, 7);
        d = TopologiesHelpers::NormalizeCoord<Topo2>()(c, dim);
        TS_ASSERT_EQUALS(c, d);

        c = Coord<3>(-1, 5, 7);
        d = TopologiesHelpers::NormalizeCoord<Topo2>()(c, dim);
        TS_ASSERT_EQUALS(Coord<3>(9, 5, 7), d);

        c = Coord<3>(1, -5, 7);
        d = TopologiesHelpers::NormalizeCoord<Topo2>()(c, dim);
        TS_ASSERT_EQUALS(Coord<3>::diagonal(-1), d);

        c = Coord<3>(-5, -5, -5);
        d = TopologiesHelpers::NormalizeCoord<Topo3>()(c, dim);
        TS_ASSERT_EQUALS(Coord<3>(5, 15, 25), d);

        c = Coord<3>(5, 5, 5);
        d = TopologiesHelpers::NormalizeCoord<Topo3>()(c, dim);
        TS_ASSERT_EQUALS(Coord<3>(5, 5, 5), d);
    }

    void testWrapsAxisClass()
    {
        typedef Topologies::Cube<2>::Topology Cube2;
        typedef Topologies::Cube<3>::Topology Cube3;
        typedef Topologies::Torus<2>::Topology Torus2;
        typedef Topologies::Torus<3>::Topology Torus3;
        typedef Topologies::NDimensional<Topologies::NDimensional<Topologies::ZeroDimensional, true>, false> MyTopo;

        TS_ASSERT_EQUALS((WrapsAxis<0, Cube2>::VALUE), false);
        TS_ASSERT_EQUALS((WrapsAxis<1, Cube2>::VALUE), false);

        TS_ASSERT_EQUALS((WrapsAxis<0, Cube3>::VALUE), false);
        TS_ASSERT_EQUALS((WrapsAxis<1, Cube3>::VALUE), false);
        TS_ASSERT_EQUALS((WrapsAxis<2, Cube3>::VALUE), false);

        TS_ASSERT_EQUALS((WrapsAxis<0, Torus2>::VALUE), true);
        TS_ASSERT_EQUALS((WrapsAxis<1, Torus2>::VALUE), true);

        TS_ASSERT_EQUALS((WrapsAxis<0, Torus3>::VALUE), true);
        TS_ASSERT_EQUALS((WrapsAxis<1, Torus3>::VALUE), true);
        TS_ASSERT_EQUALS((WrapsAxis<2, Torus3>::VALUE), true);

        TS_ASSERT_EQUALS((WrapsAxis<0, MyTopo>::VALUE), true);
        TS_ASSERT_EQUALS((WrapsAxis<1, MyTopo>::VALUE), false);

        TS_ASSERT_EQUALS((WrapsAxis<0, MyTopo>::VALUE), MyTopo::wrapsAxis(0));
        TS_ASSERT_EQUALS((WrapsAxis<1, MyTopo>::VALUE), MyTopo::wrapsAxis(1));
    }

    void testIsOutOfBoundsCube2D()
    {
        Coord<2> dimensions(20, 30);
        Topologies::IsOutOfBoundsHelper<
            1, 
            Coord<2>, 
            Topologies::Cube<2>::Topology > accessor;
        TS_ASSERT_EQUALS(false, accessor(Coord<2>(0,  0), dimensions));
        TS_ASSERT_EQUALS(false, accessor(Coord<2>(2,  2), dimensions));
        TS_ASSERT_EQUALS(true,  accessor(Coord<2>(2, -1), dimensions));
        TS_ASSERT_EQUALS(true,  accessor(Coord<2>(2, 30), dimensions));
        TS_ASSERT_EQUALS(true,  accessor(Coord<2>(-1, 1), dimensions));
        TS_ASSERT_EQUALS(true,  accessor(Coord<2>(30, 1), dimensions));
        TS_ASSERT_EQUALS(true,  accessor(Coord<2>(-1, 100), dimensions));
    }

    void testIsOutOfBoundsCube3D()
    {
        Coord<3> dimensions(20, 30, 40);
        Topologies::IsOutOfBoundsHelper<
            2, 
            Coord<3>, 
            Topologies::Cube<3>::Topology > accessor;
        TS_ASSERT_EQUALS(false, accessor(Coord<3>(0,   0, 10), dimensions));
        TS_ASSERT_EQUALS(false, accessor(Coord<3>(2,   2, 10), dimensions));
        TS_ASSERT_EQUALS(true,  accessor(Coord<3>(2,  -1, 10), dimensions));
        TS_ASSERT_EQUALS(true,  accessor(Coord<3>(2,  30, 10), dimensions));
        TS_ASSERT_EQUALS(true,  accessor(Coord<3>(-1,  1, 10), dimensions));
        TS_ASSERT_EQUALS(true,  accessor(Coord<3>(30,  1, 10), dimensions));
        TS_ASSERT_EQUALS(true,  accessor(Coord<3>(-1, 100, 1), dimensions));
        TS_ASSERT_EQUALS(true,  accessor(Coord<3>(0,   0, -1), dimensions));
        TS_ASSERT_EQUALS(true,  accessor(Coord<3>(0,   0, 40), dimensions));
    }

    void testIsOutOfBoundsTorus()
    {
        Coord<2> dimensions(20, 30);
        Topologies::IsOutOfBoundsHelper<
            1, 
            Coord<2>, 
            Topologies::Torus<2>::Topology > accessor;
        TS_ASSERT_EQUALS(false, accessor(Coord<2>(0,  0), dimensions));
        TS_ASSERT_EQUALS(false, accessor(Coord<2>(2,  2), dimensions));
        TS_ASSERT_EQUALS(false, accessor(Coord<2>(2, -1), dimensions));
        TS_ASSERT_EQUALS(false, accessor(Coord<2>(2, 30), dimensions));
        TS_ASSERT_EQUALS(false, accessor(Coord<2>(-1, 1), dimensions));
        TS_ASSERT_EQUALS(false, accessor(Coord<2>(30, 1), dimensions));
        TS_ASSERT_EQUALS(false, accessor(Coord<2>(-1, 100), dimensions));
    }

    void testLocateHelperLinear()
    {
        int b = -1;

        // in one dimension it's a simple linear indexing using the
        // selected element of the coord (selection via the dimension)
        int x[] = {1, 5, 13, 17};
        b = Topologies::Cube<1>::Topology::LocateHelper<1, int>()(x, Coord<1>(1), Coord<1>(3));
        TS_ASSERT_EQUALS(5, b);
        b = Topologies::Cube<1>::Topology::LocateHelper<1, int>()(x, Coord<1>(3), Coord<1>(3));
        TS_ASSERT_EQUALS(17, b);
        b = Topologies::Torus<1>::Topology::LocateHelper<1, int>()(x, Coord<1>(8), Coord<1>(3));
        TS_ASSERT_EQUALS(13, b);
    }

    void testLocateHelperNonConst()
    {
        // in one dimension it's a simple linear indexing using the
        // selected element of the coord (selection via the dimension)

        int x[] = {1, 5, 13, 17};
        Topologies::Cube<1>::Topology::LocateHelper<1, int>()(x, Coord<1>(1), Coord<1>(3)) = 4711;
        TS_ASSERT_EQUALS(4711, x[1]);

        Topologies::Cube<1>::Topology::LocateHelper<1, int>()(x, Coord<1>(3), Coord<1>(3)) = 4711;
        TS_ASSERT_EQUALS(4711, x[3]);
    }

    void testLocateHelperCube()
    {
        Coord<2> dimensions(7, 9);
        Grid<int, Topologies::Cube<2>::Topology> g(dimensions, 0, -1);
        for (int y = 0; y < 9; ++y)
            for (int x = 0; x < 7; ++x)
                g[Coord<2>(x, y)] = y * 10 + x;
        int actual;
        
        actual = Topologies::Cube<2>::Topology::LocateHelper<
            2,
            int>()(g, Coord<2>(1, 2), Coord<2>(3, 4));
        TS_ASSERT_EQUALS(21, actual);

        // correct since NormalizeCoord is a NOP for the Cube topology
        actual = Topologies::Cube<2>::Topology::LocateHelper<
            2,
            int>()(g, Coord<2>(6, 2), Coord<2>(3, 4));
        TS_ASSERT_EQUALS(26, actual);
    }

    void testLocateHelperCubeNonConst()
    {
        Coord<2> dimensions(7, 9);
        Grid<int, Topologies::Cube<2>::Topology> g(dimensions, 0, -1);
        for (int y = 0; y < 9; ++y)
            for (int x = 0; x < 7; ++x)
                g[Coord<2>(x, y)] = y * 10 + x;

        double foo[] = {1.0, 1.1, 1.2, 1.3};
        Topologies::Cube<1>::Topology::LocateHelper<
            1,
            double>()(foo, Coord<1>(2), Coord<1>(4)) = 4711;
        TS_ASSERT_EQUALS(4711, foo[2]);

        Topologies::Cube<2>::Topology::LocateHelper<
            2,
            int>()(g, Coord<2>(1, 2), Coord<2>(3, 4)) = 4711;
        TS_ASSERT_EQUALS(4711, g[2][1]);

        // correct since NormalizeCoord is a NOP for the Cube topology
        Topologies::Cube<2>::Topology::LocateHelper<
            2,
            int>()(g, Coord<2>(3, 2), Coord<2>(3, 4)) = 4711;
        TS_ASSERT_EQUALS(4711, g[2][3]);
    }

    void testLocateHelperTorus()
    {
        Coord<2> dimensions(7, 9);
        Grid<int, Topologies::Torus<2>::Topology> g(dimensions, 0, -1);
        for (int y = 0; y < 9; ++y)
            for (int x = 0; x < 7; ++x)
                g[Coord<2>(x, y)] = y * 10 + x;
        int actual;

        // y-coordinate is wrapped to 3
        actual = Topologies::Torus<2>::Topology::LocateHelper<
            2,
            int>()(g, Coord<2>(1, -1), Coord<2>(3, 4));
        TS_ASSERT_EQUALS(31, actual);

        // x-coordinate is wrapped to 2
        actual = Topologies::Torus<2>::Topology::LocateHelper<
            2,
            int>()(g, Coord<2>(5, 2), Coord<2>(3, 4));
        TS_ASSERT_EQUALS(22, actual);
    }

    void testLocateCube2D()
    {
        Coord<2> dimensions(7, 9);
        Grid<int, Topologies::Cube<2>::Topology> g(dimensions, 0, -1);
        for (int y = 0; y < 9; ++y)
            for (int x = 0; x < 7; ++x)
                g[Coord<2>(x, y)] = y * 10 + x;
        int actual;

        actual = Topologies::Cube<2>::Topology::locate(g, Coord<2>(1, 2));
        TS_ASSERT_EQUALS(21, actual);

        actual = Topologies::Cube<2>::Topology::locate(g, Coord<2>(7, 2));
        TS_ASSERT_EQUALS(-1, actual);

        actual = Topologies::Cube<2>::Topology::locate(g, Coord<2>(4, -1));
        TS_ASSERT_EQUALS(-1, actual);

        Topologies::Cube<2>::Topology::locate(g, Coord<2>(2, 3)) = 4711;
        TS_ASSERT_EQUALS(4711, g[3][2]);

        Topologies::Cube<2>::Topology::locate(g, Coord<2>(4, -1)) = 4712;
        TS_ASSERT_EQUALS(4712, g.getEdgeCell());
    }

    void testLocateCube3D()
    {
        Coord<3> dimensions(7, 9, 4);
        Grid<int, Topologies::Cube<3>::Topology> g(dimensions, 0, -1);
        for (int z = 0; z < 4; ++z)
            for (int y = 0; y < 9; ++y)
                for (int x = 0; x < 7; ++x)
                    g[Coord<3>(x, y, z)] = z * 100 + y * 10 + x;
        int actual;

        Topologies::Cube<3>::Topology::locate(g, Coord<3>(4, -1, 0)) = 4712;
        TS_ASSERT_EQUALS(4712, g.getEdgeCell());

        actual = Topologies::Cube<3>::Topology::locate(g, Coord<3>(1, 2, 3));
        TS_ASSERT_EQUALS(321, actual);

    }

    void testLocateTorus()
    {
        Coord<2> dimensions(7, 9);
        Grid<int, Topologies::Torus<2>::Topology> g(dimensions, 0, -1);
        for (int y = 0; y < 9; ++y) {
            for (int x = 0; x < 7; ++x) {
                g[Coord<2>(x, y)] = y * 10 + x;
            }
        }
        int actual;

        actual = Topologies::Torus<2>::Topology::locate(g, Coord<2>(1, 2));
        TS_ASSERT_EQUALS(21, actual);

        actual = Topologies::Torus<2>::Topology::locate(g, Coord<2>(7, 2));
        TS_ASSERT_EQUALS(20, actual);

        actual = Topologies::Torus<2>::Topology::locate(g, Coord<2>(4, -1));
        TS_ASSERT_EQUALS(84, actual);

        Topologies::Torus<2>::Topology::locate(g, Coord<2>(4, -1)) = 4711;
        TS_ASSERT_EQUALS(4711, g[8][4]);
    }

    void testNormalize()
    {
        typedef Topologies::Torus<3>::Topology Topo;
        Coord<3> c;

        c = Topo::normalize(Coord<3>(1, 1, 1), Coord<3>(5, 6, 7));
        TS_ASSERT_EQUALS(c, Coord<3>(1, 1, 1));

        c = Topo::normalize(Coord<3>(5, 2, -2), Coord<3>(5, 6, 7));
        TS_ASSERT_EQUALS(c, Coord<3>(0, 2, 5));
    }

    void testWrapsAxis()
    {
        TS_ASSERT_EQUALS(false, Topologies::Cube<3>::Topology::wrapsAxis(0));
        TS_ASSERT_EQUALS(false, Topologies::Cube<3>::Topology::wrapsAxis(1));
        TS_ASSERT_EQUALS(false, Topologies::Cube<3>::Topology::wrapsAxis(2));

        TS_ASSERT_EQUALS(true, Topologies::Torus<3>::Topology::wrapsAxis(0));
        TS_ASSERT_EQUALS(true, Topologies::Torus<3>::Topology::wrapsAxis(1));
        TS_ASSERT_EQUALS(true, Topologies::Torus<3>::Topology::wrapsAxis(2));

        typedef Topologies::NDimensional<Topologies::NDimensional<Topologies::ZeroDimensional, true>, false> MyTopo;
        TS_ASSERT_EQUALS(true,  MyTopo::wrapsAxis(0));
        TS_ASSERT_EQUALS(false, MyTopo::wrapsAxis(1));

        Grid<int, MyTopo> grid(Coord<2>(7, 9), 0, -1);
        for (int y = 0; y < 9; ++y) {
            for (int x = 0; x < 7; ++x) {
                grid[Coord<2>(x, y)] = y * 10 + x;
            }
        }
         
        TS_ASSERT_EQUALS(56, grid[Coord<2>(-1, 5)]);
        TS_ASSERT_EQUALS(50, grid[Coord<2>(7, 5)]);

        TS_ASSERT_EQUALS(-1, grid[Coord<2>(3, -1)]);
        TS_ASSERT_EQUALS(-1, grid[Coord<2>(5,  9)]);
    }
};

}
