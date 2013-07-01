#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/testplotter.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class TestPlotterTest : public CxxTest::TestSuite
{
public:
    void testPlotCell()
    {
        TestCell<2> c;
        c.testValue = 65;
        Image is(100, 100, Color(100, 100, 100));
        TestPlotter().plotCell(c, &is, Coord<2>(10, 20), 30, 40);

        Image expected(100, 100, Color(100, 100, 100));
        expected.paste(Coord<2>(10, 20), Image(30, 40, Color(65, 47, 11)));

        TS_ASSERT_EQUALS(is, expected);
    }
};

};
