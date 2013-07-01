#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/fixedcoord.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class FixedCoordTest : public CxxTest::TestSuite
{
public:

    void testBasic()
    {
        TS_ASSERT_EQUALS(3, sum(FixedCoord<1, 2>()));
        TS_ASSERT_EQUALS(9, sum(FixedCoord<4, 2, 3>()));
    }

private:
    template<int X, int Y, int Z>
    int sum(FixedCoord<X, Y, Z>)
    {
        return X + Y + Z;
    }
};

}
