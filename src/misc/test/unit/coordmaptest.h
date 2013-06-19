#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/coordmap.h>
#include <libgeodecomp/misc/grid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CoordMapTest : public CxxTest::TestSuite
{
public:

    void testSquareBracketsOperator()
    {
        const int width = 12;
        const int height = 34;
        Grid<double> g(Coord<2>(width, height));
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < width; y++) {
                g[y][x] = y * width + x + 47.11;
            }
        }

        Coord<2> origin(5, 7);
        CoordMap<double, Grid<double> > m(origin, &g);

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                Coord<2> relCoord(i, j);
                TS_ASSERT_EQUALS(g[relCoord + origin], m[relCoord]);
            }
        }

        TS_ASSERT_EQUALS(g[Coord<2>(5, 7)], (m[FixedCoord< 0,  0>()]));
        TS_ASSERT_EQUALS(g[Coord<2>(4, 7)], (m[FixedCoord<-1,  0>()]));
        TS_ASSERT_EQUALS(g[Coord<2>(6, 6)], (m[FixedCoord< 1, -1>()]));
        TS_ASSERT_EQUALS(g[Coord<2>(6, 8)], (m[FixedCoord< 1,  1>()]));
    }


    void testToString()
    {
        Coord<2> origin(23, 42);
        CoordMap<double, Grid<double> > map(origin, 0);
        TS_ASSERT_EQUALS(
                map.toString(),
                "CoordMap origin: " + origin.toString() + "\n");
    }

};

}
