#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/plane.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class PlaneTest : public CxxTest::TestSuite
{
public:
    void testIsOnTop2DSpaceA()
    {
        Plane<Coord<2> > plane(Coord<2>(20, 10), Coord<2>(0, 1));

        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<2>(20, 11)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<2>( 0, 11)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<2>(99, 11)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<2>(20, 100000)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(20,  9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>( 0,  9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(99,  9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(20, -999)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(20, 10)));
    }

    void testIsOnTop2DSpaceB()
    {
        Plane<Coord<2> > plane(Coord<2>(20, 10), Coord<2>(0, -1));

        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(20, 11)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>( 0, 11)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(99, 11)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(20, 100000)));

        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<2>(20,  9)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<2>( 0,  9)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<2>(99,  9)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<2>(20, -999)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(20, 10)));
    }

    void testIsOnTop2DSpaceC()
    {
        Plane<Coord<2> > plane(Coord<2>(20, 10), Coord<2>(1, 0));

        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<2>(21, 10)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<2>(21,  0)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<2>(21, 99)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<2>(10000, 10)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(19, 10)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(19,  0)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(19, 99)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(20, -999)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(20, 10)));
    }

    void testIsOnTop2DSpaceD()
    {
        Plane<Coord<2> > plane(Coord<2>(20, 10), Coord<2>(1, 2));

        TS_ASSERT_EQUALS(true,  plane.isOnTop(Coord<2>(20, 11)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>( 0, 11)));
        TS_ASSERT_EQUALS(true,  plane.isOnTop(Coord<2>(99, 11)));
        TS_ASSERT_EQUALS(true,  plane.isOnTop(Coord<2>(20, 1000)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(20,  9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>( 0,  9)));
        TS_ASSERT_EQUALS(true,  plane.isOnTop(Coord<2>(99,  9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(20, -999)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<2>(20, 10)));
    }

    void testIsOnTopWithFloatCoordAnd2DSpaceA()
    {
        Plane<FloatCoord<2> > plane(FloatCoord<2>(20, 10), FloatCoord<2>(0, 1));

        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<2>(20, 11)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<2>( 0, 11)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<2>(99, 11)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<2>(20, 100000)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(20,  9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>( 0,  9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(99,  9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(20, -999)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(20, 10)));
    }

    void testIsOnTopWithFloatCoordAnd2DSpaceB()
    {
        Plane<FloatCoord<2> > plane(FloatCoord<2>(20, 10), FloatCoord<2>(0, -1));

        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(20, 11)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>( 0, 11)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(99, 11)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(20, 100000)));

        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<2>(20,  9)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<2>( 0,  9)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<2>(99,  9)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<2>(20, -999)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(20, 10)));
    }

    void testIsOnTopWithFloatCoordAnd2DSpaceC()
    {
        Plane<FloatCoord<2> > plane(FloatCoord<2>(20, 10), FloatCoord<2>(1, 0));

        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<2>(21, 10)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<2>(21,  0)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<2>(21, 99)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<2>(10000, 10)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(19, 10)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(19,  0)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(19, 99)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(20, -999)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(20, 10)));
    }

    void testIsOnTopWithFloatCoordAnd2DSpaceD()
    {
        Plane<FloatCoord<2> > plane(FloatCoord<2>(20, 10), FloatCoord<2>(1, 2));

        TS_ASSERT_EQUALS(true,  plane.isOnTop(FloatCoord<2>(20, 11)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>( 0, 11)));
        TS_ASSERT_EQUALS(true,  plane.isOnTop(FloatCoord<2>(99, 11)));
        TS_ASSERT_EQUALS(true,  plane.isOnTop(FloatCoord<2>(20, 1000)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(20,  9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>( 0,  9)));
        TS_ASSERT_EQUALS(true,  plane.isOnTop(FloatCoord<2>(99,  9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(20, -999)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<2>(20, 10)));
    }

    void testIsOnTop3DSpaceA()
    {
        Plane<Coord<3> > plane(Coord<3>(40, 20, 10), Coord<3>(0, 0, 1));

        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<3>(40, 20, 11)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<3>(40, 10, 11)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<3>(40, 30, 11)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<3>(30, 20, 11)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(Coord<3>(50, 20, 11)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<3>(40, 20, 9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<3>(40, 10, 9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<3>(40, 30, 9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<3>(30, 20, 9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<3>(50, 20, 9)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<3>(40, 20, 10)));
    }

    void testIsOnTop3DSpaceB()
    {
        Plane<Coord<3> > plane(Coord<3>(40, 20, 10), Coord<3>(1, 1, 2));

        TS_ASSERT_EQUALS(true,  plane.isOnTop(Coord<3>(40, 20, 11)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<3>(40, 10, 11)));
        TS_ASSERT_EQUALS(true,  plane.isOnTop(Coord<3>(40, 30, 11)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<3>(30, 20, 11)));
        TS_ASSERT_EQUALS(true,  plane.isOnTop(Coord<3>(50, 20, 11)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<3>(40, 20, 9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<3>(40, 10, 9)));
        TS_ASSERT_EQUALS(true,  plane.isOnTop(Coord<3>(40, 30, 9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<3>(30, 20, 9)));
        TS_ASSERT_EQUALS(true,  plane.isOnTop(Coord<3>(50, 20, 9)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(Coord<3>(40, 20, 10)));
    }

    void testIsOnTopWithFloatCoordAnd3DSpaceA()
    {
        Plane<FloatCoord<3> > plane(FloatCoord<3>(40, 20, 10), FloatCoord<3>(0, 0, 1));

        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<3>(40, 20, 11)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<3>(40, 10, 11)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<3>(40, 30, 11)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<3>(30, 20, 11)));
        TS_ASSERT_EQUALS(true, plane.isOnTop(FloatCoord<3>(50, 20, 11)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<3>(40, 20, 9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<3>(40, 10, 9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<3>(40, 30, 9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<3>(30, 20, 9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<3>(50, 20, 9)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<3>(40, 20, 10)));
    }

    void testIsOnTopWithFloatCoordAnd3DSpaceB()
    {
        Plane<FloatCoord<3> > plane(FloatCoord<3>(40, 20, 10), FloatCoord<3>(1, 1, 2));

        TS_ASSERT_EQUALS(true,  plane.isOnTop(FloatCoord<3>(40, 20, 11)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<3>(40, 10, 11)));
        TS_ASSERT_EQUALS(true,  plane.isOnTop(FloatCoord<3>(40, 30, 11)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<3>(30, 20, 11)));
        TS_ASSERT_EQUALS(true,  plane.isOnTop(FloatCoord<3>(50, 20, 11)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<3>(40, 20, 9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<3>(40, 10, 9)));
        TS_ASSERT_EQUALS(true,  plane.isOnTop(FloatCoord<3>(40, 30, 9)));
        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<3>(30, 20, 9)));
        TS_ASSERT_EQUALS(true,  plane.isOnTop(FloatCoord<3>(50, 20, 9)));

        TS_ASSERT_EQUALS(false, plane.isOnTop(FloatCoord<3>(40, 20, 10)));
    }

};

}
