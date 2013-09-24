#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/misc/regionstreakiterator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class RegionStreakIteratorTest : public CxxTest::TestSuite
{
public:
    // most tests reside in regiontest.h.
    typedef RegionStreakIterator<3, Region<3> > Iterator;

    void testOffsetBasedConstructor3D()
    {
        Region<3> r;
        for (int z = 10; z < 20; ++z) {
            for (int y = 5; y < 15; ++y) {
                for (int x = 0; x < 4; ++x) {
                    r << Streak<3>(Coord<3>(x * 10, y, z), x * 10 + 4);
                }
            }
        }


        Iterator i = r.beginStreak();
        Iterator t = r[Coord<3>(0, 0, 0)];
        TS_ASSERT_EQUALS(*i, *t);

        for (int k = 0; k < 3; ++k) {
            ++i;
        }
        t = r[Coord<3>(3, 0, 0)];
        TS_ASSERT_EQUALS(*i, *t);

        for (int k = 3; k < 48; ++k) {
            ++i;
        }
        t = r[Coord<3>(48, 12, 1)];
        TS_ASSERT_EQUALS(*i, *t);
    }

};

}
