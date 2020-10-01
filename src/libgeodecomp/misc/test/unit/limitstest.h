#include <cfloat>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/limits.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class LimitsTest : public CxxTest::TestSuite
{
public:
    void testInt()
    {
        int value0 = Limits<int>::getMin();
        int value1 = Limits<int>::getMax();
        TS_ASSERT(value0 < value1);

        value0 -= 1;
        TS_ASSERT_EQUALS(value0, value1);

        TS_ASSERT_EQUALS(INT_MIN, value0);
        TS_ASSERT_EQUALS(INT_MAX, value1);
    }

    void testFloat()
    {
        float value0 = Limits<float>::getMin();
        float value1 = Limits<float>::getMax();
        TS_ASSERT(value0 < value1);

        TS_ASSERT_EQUALS(FLT_MIN, value0);
        TS_ASSERT_EQUALS(FLT_MAX, value1);
    }

    void testDouble()
    {
        double value0 = Limits<double>::getMin();
        double value1 = Limits<double>::getMax();
        TS_ASSERT(value0 < value1);

        TS_ASSERT_EQUALS(DBL_MIN, value0);
        TS_ASSERT_EQUALS(DBL_MAX, value1);
    }
};

}
