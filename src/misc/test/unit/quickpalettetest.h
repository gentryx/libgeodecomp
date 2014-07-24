#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/quickpalette.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class QuickPaletteTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        QuickPalette<double> pal(0, 1);

        TS_ASSERT_EQUALS(pal[-0.01], Color::BLACK);
        TS_ASSERT_EQUALS(pal[ 0.00], Color::BLUE);
        TS_ASSERT_EQUALS(pal[ 0.25], Color::CYAN);
        TS_ASSERT_EQUALS(pal[ 0.50], Color::GREEN);
        TS_ASSERT_EQUALS(pal[ 0.75], Color::YELLOW);
        TS_ASSERT_EQUALS(pal[ 1.00], Color::RED);
        TS_ASSERT_EQUALS(pal[ 1.01], Color::WHITE);
    }
};

}
