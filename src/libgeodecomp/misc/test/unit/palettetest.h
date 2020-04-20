#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/palette.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class PaletteTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        Palette<double> pal;
        TS_ASSERT_THROWS(pal[0], std::logic_error&);

        pal.addColor(-1, Color(50, 0, 0));
        TS_ASSERT_EQUALS(Color(50, 0, 0), pal[ 0]);
        TS_ASSERT_EQUALS(Color(50, 0, 0), pal[ 0.5]);
        TS_ASSERT_EQUALS(Color(50, 0, 0), pal[ 1]);
        TS_ASSERT_EQUALS(Color(50, 0, 0), pal[-2]);

        pal.addColor( 0, Color( 0, 50, 0));
        TS_ASSERT_EQUALS(Color(50,  0, 0), pal[-1]);
        TS_ASSERT_EQUALS(Color(25, 25, 0), pal[-0.5]);
        TS_ASSERT_EQUALS(Color( 0, 50, 0), pal[ 0]);
        TS_ASSERT_EQUALS(Color(50,  0, 0), pal[-2]);

        pal.addColor( 1, Color( 0, 0, 50));
        TS_ASSERT_EQUALS(Color(50,  0,  0), pal[-1]);
        TS_ASSERT_EQUALS(Color(25, 25,  0), pal[-0.5]);
        TS_ASSERT_EQUALS(Color( 0, 50,  0), pal[ 0]);
        TS_ASSERT_EQUALS(Color( 0, 25, 25), pal[ 0.5]);
        TS_ASSERT_EQUALS(Color( 0,  0, 50), pal[ 1]);
        TS_ASSERT_EQUALS(Color(50,  0,  0), pal[-2]);
    }
};

}
