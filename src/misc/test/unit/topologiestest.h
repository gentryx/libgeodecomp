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
        typedef TopologiesHelpers::Topology<2, true, false> MyTopo;

        TS_ASSERT_EQUALS((Cube2::WrapsAxis<0>::VALUE), false);
        TS_ASSERT_EQUALS((Cube2::WrapsAxis<1>::VALUE), false);

        TS_ASSERT_EQUALS((Cube3::WrapsAxis<0>::VALUE), false);
        TS_ASSERT_EQUALS((Cube3::WrapsAxis<1>::VALUE), false);
        TS_ASSERT_EQUALS((Cube3::WrapsAxis<2>::VALUE), false);

        TS_ASSERT_EQUALS((Torus2::WrapsAxis<0>::VALUE), true);
        TS_ASSERT_EQUALS((Torus2::WrapsAxis<1>::VALUE), true);

        TS_ASSERT_EQUALS((Torus3::WrapsAxis<0>::VALUE), true);
        TS_ASSERT_EQUALS((Torus3::WrapsAxis<1>::VALUE), true);
        TS_ASSERT_EQUALS((Torus3::WrapsAxis<2>::VALUE), true);

        TS_ASSERT_EQUALS((MyTopo::WrapsAxis<0>::VALUE), true);
        TS_ASSERT_EQUALS((MyTopo::WrapsAxis<1>::VALUE), false);

        TS_ASSERT_EQUALS((MyTopo::WrapsAxis<0>::VALUE), MyTopo::wrapsAxis(0));
        TS_ASSERT_EQUALS((MyTopo::WrapsAxis<1>::VALUE), MyTopo::wrapsAxis(1));
    }

    void testIsOutOfBoundsCube2D()
    {
        Coord<2> dimensions(20, 30);
        typedef Topologies::Cube<2>::Topology Topo;
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<2>(0,  0),   dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<2>(2,  2),   dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<2>(2, -1),   dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<2>(2, 30),   dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<2>(-1, 1),   dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<2>(30, 1),   dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<2>(-1, 100), dimensions));
    }

    void testIsOutOfBoundsCube3D()
    {
        Coord<3> dimensions(20, 30, 40);
        typedef Topologies::Cube<3>::Topology Topo;
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<3>(0,   0, 10), dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<3>(2,   2, 10), dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<3>(2,  -1, 10), dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<3>(2,  30, 10), dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<3>(-1,  1, 10), dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<3>(30,  1, 10), dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<3>(-1, 100, 1), dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<3>(0,   0, -1), dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<3>(0,   0, 40), dimensions));
    }

    void testIsOutOfBoundsTorus()
    {
        Coord<2> dimensions(20, 30);
        typedef Topologies::Torus<2>::Topology Topo;
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<2>(0,  0),   dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<2>(2,  2),   dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<2>(2, -1),   dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<2>(2, 30),   dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<2>(-1, 1),   dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<2>(30, 1),   dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<2>(-1, 100), dimensions));
    }

    void testIsOutOfBoundsSpecial1()
    {
        Coord<2> dimensions(20, 30);
        typedef TopologiesHelpers::Topology<2, false, true> Topo;
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<2>(0,  0), dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<2>(-1, 0), dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<2>(19, 0), dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<2>(20, 0), dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<2>(0, -1), dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<2>(0, 30), dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<2>(0, 32), dimensions));
    }

    void testIsOutOfBoundsSpecial2()
    {
        Coord<3> dimensions(20, 30, 40);
        typedef TopologiesHelpers::Topology<3, true, false, true> Topo;
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<3>(0,  0,  0), dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<3>(-1, 0,  0), dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<3>(20, 0,  0), dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<3>(0, -1,  0), dimensions));
        TS_ASSERT_EQUALS(true,  Topo::isOutOfBounds(Coord<3>(0, 30,  0), dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<3>(0,  0, -1), dimensions));
        TS_ASSERT_EQUALS(false, Topo::isOutOfBounds(Coord<3>(0,  0, 40), dimensions));
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

        typedef TopologiesHelpers::Topology<2, true, false> MyTopo;
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
