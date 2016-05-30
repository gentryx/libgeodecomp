#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/time.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class TimeTest : public CxxTest::TestSuite
{
public:
    void testRenderISO()
    {
        TS_ASSERT_EQUALS(Time::renderISO(12345.6),           "1970.01.01 04:25:45.600000");
        TS_ASSERT_EQUALS(Time::renderISO(1234567890.000998), "2009.02.14 00:31:30.000998");
    }
};

}
