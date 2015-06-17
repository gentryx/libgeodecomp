#include <cxxtest/TestSuite.h>
#include <cuda.h>

#include <libgeodecomp/geometry/coord.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CoordTest2 : public CxxTest::TestSuite
{
public:

    void testConversion()
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
