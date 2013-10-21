#include <cxxtest/TestSuite.h>
#include <boost/math/tools/precision.hpp>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/misc/testhelper.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class FloatCoordTest : public CxxTest::TestSuite
{
public:
    void testDefaultConstructors()
    {
        FloatCoord<2> a;
        FloatCoord<2> b(0, 0);
        TS_ASSERT_EQUALS(a, b);

        Coord<3> c(1, 2, 3);
        FloatCoord<3> d(c);
        TS_ASSERT_EQUALS(d, FloatCoord<3>(1, 2, 3));
    }

    void testLength()
    {
        TS_ASSERT_EQUALS_DOUBLE(1.2, FloatCoord<1>(-1.2).length());
        TS_ASSERT_EQUALS_DOUBLE(5.0, FloatCoord<2>(3, -4).length());
        TS_ASSERT_EQUALS_DOUBLE(7.0, FloatCoord<3>(2, 3, 6).length());
    }

    void testSum()
    {
        TS_ASSERT_EQUALS_DOUBLE(2.0, FloatCoord<1>(2.0).sum());
        TS_ASSERT_EQUALS_DOUBLE(3.0, FloatCoord<2>(1.5, 1.5).sum());
        TS_ASSERT_EQUALS_DOUBLE(6.0, FloatCoord<3>(1, 3.3, 1.7).sum());
    }

    void testOperatorPlus()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(3),
                         FloatCoord<1>(1) + FloatCoord<1>(2));
        TS_ASSERT_EQUALS(FloatCoord<2>(4, 6),
                         FloatCoord<2>(1, 2) + FloatCoord<2>(3, 4));
        TS_ASSERT_EQUALS(FloatCoord<3>(5, 7, 9),
                         FloatCoord<3>(1, 2, 3) + FloatCoord<3>(4, 5, 6));

        {
            FloatCoord<1> a(1);
            a += FloatCoord<1>(4);
            TS_ASSERT_EQUALS(FloatCoord<1>(5), a);
        }
        {
            FloatCoord<2> a(1, 2);
            a += FloatCoord<2>(6, 3);
            TS_ASSERT_EQUALS(FloatCoord<2>(7, 5), a);
        }
        {
            FloatCoord<3> a(1, 7, 2);
            a += FloatCoord<3>(6, 1, 7);
            TS_ASSERT_EQUALS(FloatCoord<3>(7, 8, 9), a);
        }
    }

    void testOperatorMinus()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(-1),
                         FloatCoord<1>(1) - FloatCoord<1>(2));
        TS_ASSERT_EQUALS(FloatCoord<2>(-2, -3),
                         FloatCoord<2>(1, 2) - FloatCoord<2>(3, 5));
        TS_ASSERT_EQUALS(FloatCoord<3>(-7, -3, -4),
                         FloatCoord<3>(1, 2, 3) - FloatCoord<3>(8, 5, 7));


        {
            FloatCoord<1> a(5);
            a -= FloatCoord<1>(4);
            TS_ASSERT_EQUALS(FloatCoord<1>(1), a);
        }
        {
            FloatCoord<2> a(7, 5);
            a -= FloatCoord<2>(6, 3);
            TS_ASSERT_EQUALS(FloatCoord<2>(1, 2), a);
        }
        {
            FloatCoord<3> a(7, 8, 9);
            a -= FloatCoord<3>(6, 1, 7);
            TS_ASSERT_EQUALS(FloatCoord<3>(1, 7, 2), a);
        }
    }

    void testOperatorMultiply()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(4),
                         FloatCoord<1>(1) * 4);
        TS_ASSERT_EQUALS(FloatCoord<2>(4, 8),
                         FloatCoord<2>(1, 2) * 4);
        TS_ASSERT_EQUALS(FloatCoord<3>(4, 8, 12),
                         FloatCoord<3>(1, 2, 3) * 4);

        TS_ASSERT_EQUALS(FloatCoord<1>(4),
                         FloatCoord<1>(1) * 4);
        TS_ASSERT_EQUALS(FloatCoord<2>(4, 8),
                         FloatCoord<2>(1, 2) * 4);
        TS_ASSERT_EQUALS(FloatCoord<3>(4, 8, 12),
                         FloatCoord<3>(1, 2, 3) * 4);
        {
            FloatCoord<1> f(3);
            f *= 5;
            TS_ASSERT_EQUALS(FloatCoord<1>(15), f);
        }
        {
            FloatCoord<2> f(3, 5);
            f *= 5;
            TS_ASSERT_EQUALS(FloatCoord<2>(15, 25), f);
        }
        {
            FloatCoord<3> f(3, 5, 2);
            f *= 5;
            TS_ASSERT_EQUALS(FloatCoord<3>(15, 25, 10), f);
        }
    }

    void testOperatorEquals()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(4), FloatCoord<1>(4));
        TS_ASSERT(!(FloatCoord<1>(4) == FloatCoord<1>(5)));

        TS_ASSERT_EQUALS(FloatCoord<2>(4, 1), FloatCoord<2>(4, 1));
        TS_ASSERT(!(FloatCoord<2>(4, 4) == FloatCoord<2>(5, 4)));

        TS_ASSERT_EQUALS(FloatCoord<3>(4, 1, 5), FloatCoord<3>(4, 1, 5));
        TS_ASSERT(!(FloatCoord<3>(4, 4, 4) == FloatCoord<3>(4, 4, 3)));

        TS_ASSERT(FloatCoord<1>(1) != FloatCoord<1>(2));
        TS_ASSERT(!(FloatCoord<1>(1) != FloatCoord<1>(1)));

        TS_ASSERT(FloatCoord<2>(1, 2) != FloatCoord<2>(1, 9));
        TS_ASSERT(!(FloatCoord<2>(1, 2) != FloatCoord<2>(1, 2)));

        TS_ASSERT(FloatCoord<3>(1, 2, 3) != FloatCoord<3>(1, 2, 4));
        TS_ASSERT(!(FloatCoord<3>(1, 2, 3) != FloatCoord<3>(1, 2, 3)));
    }
};

}
