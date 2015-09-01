#include <libgeodecomp/geometry/convexpolytope.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class ConvexPolytopeTest : public CxxTest::TestSuite
{
public:
    void testSquare2D()
    {
        ConvexPolytope<Coord<2> > poly(Coord<2>(200, 100), Coord<2>(300, 300), Coord<2>(1000, 1000), 0.5, 0);

        // create a square:
        poly << std::make_pair(Coord<2>(200,   0), 0)
             << std::make_pair(Coord<2>(300, 100), 1)
             << std::make_pair(Coord<2>(200, 200), 2)
             << std::make_pair(Coord<2>(100, 100), 3);
        poly.updateGeometryData();

        double expectedVolume = 100 * 100;

        TS_ASSERT_EQUALS(Coord<2>(200, 100), poly.getCenter());
        TS_ASSERT_LESS_THAN(0.9 * expectedVolume, poly.getVolume());
        TS_ASSERT_LESS_THAN(poly.getVolume(), 1.1 * expectedVolume);

        TS_ASSERT_EQUALS(100, poly.getDiameter());

        std::vector<Coord<2> > expectedShape;
        expectedShape << Coord<2>(250,  50)
                      << Coord<2>(250, 150)
                      << Coord<2>(150, 150)
                      << Coord<2>(150,  50);

        TS_ASSERT_EQUALS(expectedShape, poly.getShape());

        TS_ASSERT_EQUALS(false, poly.includes(Coord<2>(100,  50)));
        TS_ASSERT_EQUALS(false, poly.includes(Coord<2>(200,  50)));
        TS_ASSERT_EQUALS(false, poly.includes(Coord<2>(300,  50)));

        TS_ASSERT_EQUALS(false, poly.includes(Coord<2>(100, 100)));
        TS_ASSERT_EQUALS(true,  poly.includes(Coord<2>(200, 100)));
        TS_ASSERT_EQUALS(false, poly.includes(Coord<2>(300, 100)));

        TS_ASSERT_EQUALS(false, poly.includes(Coord<2>(100, 150)));
        TS_ASSERT_EQUALS(false, poly.includes(Coord<2>(200, 150)));
        TS_ASSERT_EQUALS(false, poly.includes(Coord<2>(300, 150)));
    }

    void testTriangle2D()
    {
        ConvexPolytope<Coord<2> > triangle1(Coord<2>(110, 10), Coord<2>(300, 300), Coord<2>(1000, 1000), 0.5, 0);
        triangle1 << std::make_pair(Coord<2>(190,  90), 4711)
                  << std::make_pair(Coord<2>( 90,  10), 666)
                  << std::make_pair(Coord<2>( 10, -10), 12345);
        triangle1.updateGeometryData();

        double expectedVolume = 0.5 * 100 * 100;
        TS_ASSERT_LESS_THAN(0.9 * expectedVolume, triangle1.getVolume());
        TS_ASSERT_LESS_THAN(triangle1.getVolume(), 1.1 * expectedVolume);
    }
};

}
