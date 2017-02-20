#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/time.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class TimeStringConversionTest : public CxxTest::TestSuite
{
public:
    void testRenderISO()
    {
        TS_ASSERT_EQUALS(TimeStringConversion::renderISO(12345.6),           "1970.01.01 03:25:45.600000");
        TS_ASSERT_EQUALS(TimeStringConversion::renderISO(1234567890.000998), "2009.02.13 23:31:30.000998");
        TS_ASSERT_EQUALS(TimeStringConversion::renderISO(1234567890.000000), "2009.02.13 23:31:30.000000");
    }

    void testRenderDuration()
    {
        TS_ASSERT_EQUALS(TimeStringConversion::renderDuration(1.0),        "00:00:01");
        TS_ASSERT_EQUALS(TimeStringConversion::renderDuration(0.25),       "00:00:00.250000");
        TS_ASSERT_EQUALS(TimeStringConversion::renderDuration(36754),      "10:12:34");
        TS_ASSERT_EQUALS(TimeStringConversion::renderDuration(360754.56), "100:12:34.559999");
    }
};

}
