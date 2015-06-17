#include <cxxtest/TestSuite.h>
#include <cuda.h>

#include <libgeodecomp/geometry/coord.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CoordTest2 : public CxxTest::TestSuite
{
public:
    void testConversion1d()
    {
        dim3 d;
        d.x = 6;
        Coord<1> c = d;
        TS_ASSERT_EQUALS(Coord<1>(6), c);

        c.x() = 666;
        d = c;
        TS_ASSERT_EQUALS(666, d.x);
        TS_ASSERT_EQUALS(1,   d.y);
        TS_ASSERT_EQUALS(1,   d.z);
    }

    void testConversion2d()
    {
        dim3 d;
        d.x = 4;
        d.y = 5;
        Coord<2> c = d;
        TS_ASSERT_EQUALS(Coord<2>(4, 5), c);

        c.x() = 19;
        c.y() = 81;
        d = c;
        TS_ASSERT_EQUALS(19, d.x);
        TS_ASSERT_EQUALS(81, d.y);
        TS_ASSERT_EQUALS(1,  d.z);
    }

    void testConversion3d()
    {
        dim3 d;
        d.x = 1;
        d.y = 2;
        d.z = 3;
        Coord<3> c = d;
        TS_ASSERT_EQUALS(Coord<3>(1, 2, 3), c);

        c.x() = 47;
        c.y() = 11;
        c.z() = -1;
        d = c;
        TS_ASSERT_EQUALS(47, d.x);
        TS_ASSERT_EQUALS(11, d.y);
        TS_ASSERT_EQUALS(-1, d.z);
    }

};

}
