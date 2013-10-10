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

    void testInitForEmptyRegions()
    {
        Region<3> r;

        TS_ASSERT_EQUALS(r.beginStreak(), r[-1]);
        TS_ASSERT_EQUALS(r.beginStreak(), r[0]);

        TS_ASSERT_EQUALS(r.endStreak(), r[0]);
        TS_ASSERT_EQUALS(r.endStreak(), r[1]);
    }

    void testSubstraction2D()
    {
        Region<2> r;

        for (int y = 0; y < 10; ++y) {
            for (int x = 0; x < (5 + y); ++x) {
                r << Streak<2>(Coord<2>(x * 10, y), x * 10 + 4);
            }
        }

        size_t expectedStreaks = 10 * 5 + 10 * (10 - 1) / 2;
        TS_ASSERT_EQUALS(r.numStreaks(), expectedStreaks);

        std::vector<size_t> offsets;
        offsets << 0 << 10 << 15 << 50 << 51 << 90 << 95 << 100;

        std::vector<RegionStreakIterator<2, Region<2> > > iterators;
        for (std::vector<size_t>::iterator i = offsets.begin(); i != offsets.end(); ++i) {
            iterators << r[*i];

            if (*i <= 0) {
                TS_ASSERT_EQUALS(iterators.back(), r.beginStreak());
            }

            if (*i >= r.numStreaks()) {
                TS_ASSERT_EQUALS(iterators.back(), r.endStreak());
            }
        }

        // ensure x-offsets match original offsets
        for (size_t i = 1; i < offsets.size(); ++i) {
            Coord<2> deltaCoord = iterators[i] - iterators[i - 1];
            int deltaOffset = offsets[i] - offsets[i - 1];
            if (offsets[i] > r.numStreaks()) {
                deltaOffset = 0;
            }

            TS_ASSERT_EQUALS(deltaCoord.x(), deltaOffset);
        }

        // ensure result of substration yields the correct iterator after conversion
        for (size_t i = 0; i < offsets.size(); ++i) {
            Coord<2> deltaCoord = iterators[i] - r.beginStreak();
            RegionStreakIterator<2, Region<2> > newIter = r[deltaCoord];
            TS_ASSERT_EQUALS(newIter, iterators[i]);
            TS_ASSERT_EQUALS(*newIter, *iterators[i]);
        }
    }
};

}
