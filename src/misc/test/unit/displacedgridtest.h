#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/displacedgrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class DisplacedGridTest : public CxxTest::TestSuite
{
public:

    void testBox()
    {
        CoordBox<2> rect(Coord<2>(10, 11), Coord<2>(12, 13));
        DisplacedGrid<int> grid(rect);
        TS_ASSERT_EQUALS(rect, grid.boundingBox());
        TS_ASSERT_EQUALS(12, grid.getDimensions().x());
        TS_ASSERT_EQUALS(13, grid.getDimensions().y());
    }

    void testPaste()
    {
        DisplacedGrid<double> source(
            CoordBox<2>(Coord<2>( 0,  0), Coord<2>(100, 100)), 47.11);
        DisplacedGrid<double> target(
            CoordBox<2>(Coord<2>(40, 60), Coord<2>( 30,  30)), -1.23);
        DisplacedGrid<double> target2(
            CoordBox<2>(Coord<2>(40, 60), Coord<2>( 30,  30)), -5);

        Region<2> innerSquare;
        Region<2> outerSquare;
        for (int i = 70; i < 80; ++i) {
            innerSquare << Streak<2>(Coord<2>(50, i), 60);
        }
        for (int i = 60; i < 90; ++i) {
            outerSquare << Streak<2>(Coord<2>(40, i), 70);
        }

        Region<2> outerRing = outerSquare - innerSquare;

        for (Region<2>::Iterator i = outerSquare.begin(); i != outerSquare.end(); ++i) {
            TS_ASSERT_EQUALS(-1.23, target[*i]);
        }

        target.paste(source, innerSquare);

        for (Region<2>::Iterator i = outerRing.begin(); i != outerRing.end(); ++i) {
            TS_ASSERT_EQUALS(-1.23, target[*i]);
        }
        for (Region<2>::Iterator i = innerSquare.begin(); i != innerSquare.end(); ++i) {
            TS_ASSERT_EQUALS(47.11, target[*i]);
        }

        target2.paste(target, outerRing);

        for (Region<2>::Iterator i = outerRing.begin(); i != outerRing.end(); ++i) {
            TS_ASSERT_EQUALS(-1.23, target2[*i]);
        }

        for (Region<2>::Iterator i = innerSquare.begin(); i != innerSquare.end(); ++i) {
            TS_ASSERT_EQUALS(-5, target2[*i]);
        }
    }

    void testSetOrigin()
    {
        DisplacedGrid<int> grid(CoordBox<2>(Coord<2>(10, 20), Coord<2>(2, 3)));

        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 2; ++x) {
                grid[Coord<2>(x + 10, y + 20)] = x * 10 + y;
            }
        }

        grid.setOrigin(Coord<2>(0, 50));
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 2; ++x) {
                TS_ASSERT_EQUALS(grid[Coord<2>(x + 0, y + 50)],  x * 10 + y);
            }
        }
    }

    void testResize()
    {
        CoordBox<3> oldBox(Coord<3>(1, 1, 2), Coord<3>(3, 4, 2));
        CoordBox<3> newBox(Coord<3>(4, 4, 4), Coord<3>(3, 3, 3));

        DisplacedGrid<double, Topologies::Torus<3>::Topology> grid(oldBox);
        grid[Coord<3>(1, 1, 2)] = 47;
        grid[Coord<3>(0, 0, 1)] = 11;
        TS_ASSERT_EQUALS(47, grid[Coord<3>(1, 1, 2)]);
        TS_ASSERT_EQUALS(11, grid[Coord<3>(3, 4, 3)]);
        TS_ASSERT_EQUALS(oldBox, grid.boundingBox());

        grid.resize(newBox);
        grid[Coord<3>(4, 4, 4)] = 27;
        grid[Coord<3>(3, 3, 3)] = 1;
        TS_ASSERT_EQUALS(27, grid[Coord<3>(4, 4, 4)]);
        TS_ASSERT_EQUALS(1,  grid[Coord<3>(6, 6, 6)]);
        TS_ASSERT_EQUALS(newBox, grid.boundingBox());
    }

    void testFill3D()
    {
        CoordBox<3> insert(Coord<3>(10, 20, 15), Coord<3>(4, 7, 5));
        Coord<3> origin(10, 10, 10);
        Coord<3> dim(40, 70, 30);

        DisplacedGrid<int, Topologies::Cube<3>::Topology> g(CoordBox<3>(origin, dim), -1);
        g.fill(insert, 2);

        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                for (int x = 0; x < dim.x(); ++x) {
                    Coord<3> c = Coord<3>(x, y, z) + origin;
                    int expected = -1;
                    if (insert.inBounds(c)) {
                        expected = 2;
                    }

                    if (expected != g[c]) {
                        std::cout << c << " expected: " << expected << " actual: " << g[c] << "\n";

                    }
                    // TS_ASSERT_EQUALS(expected, g[c]);
                }
            }
        }
    }

    void testTopologicalNormalizationWithTorus()
    {
        /**
         * This test should check whether it is possible to let a
         * displaced grid act on a 15x10 torus as if it would cover a
         * 8x6 field shifted by an offset of (-3, -2), as outlined
         * below. The code will access the fields marked with a
         * through d.
         *
         *   0123456789abcde
         * 9 XXXXX_______cXX
         * 8 XXXXX_______aXX
         * 7 _______________
         * 6 _______________
         * 5 _______________
         * 4 _______________
         * 3 XXXXX_______XXX
         * 2 XXXXX_______XXX
         * 1 XXXXX_______XXX
         * 0 bXXXX_______XXd
         *
         */

        DisplacedGrid<int, Topologies::Torus<2>::Topology, true> grid(
            CoordBox<2>(Coord<2>(-3, -2),
                        Coord<2>(8, 6)),
            -2,
            -2,
            Coord<2>(15, 10));

        for (int y = -2; y < 4; ++y)
            for (int x = -3; x < 5; ++x)
                grid[Coord<2>(x, y)] = (y+3) * 10 + (x+3);

        TS_ASSERT_EQUALS(10, grid[Coord<2>(-3, -2)]);
        TS_ASSERT_EQUALS(33, grid[Coord<2>( 0,  0)]);
        TS_ASSERT_EQUALS(20, grid[Coord<2>(12, 9)]);
        TS_ASSERT_EQUALS(32, grid[Coord<2>(14, 0)]);

        TS_ASSERT_EQUALS(21, grid.getNeighborhood(
                             Coord<2>(12, 9))[Coord<2>(1, 0)]);
    }
};

}
