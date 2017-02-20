#include <cxxtest/TestSuite.h>
#include <libgeodecomp/storage/fixedneighborhoodupdatefunctor.h>
#include <libgeodecomp/storage/soagrid.h>
#include <libgeodecomp/storage/updatefunctor.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class StreakTest : public CxxTest::TestSuite
{
public:

    void testAddSub1D()
    {
        Streak<1> streak(Coord<1>(555), 600);

        streak -= Coord<1>(55);
        TS_ASSERT_EQUALS(Streak<1>(Coord<1>(500), 545), streak);

        streak += Coord<1>(166);
        TS_ASSERT_EQUALS(Streak<1>(Coord<1>(666), 711), streak);

        Streak<1> buf = streak - Coord<1>(600);
        TS_ASSERT_EQUALS(Streak<1>(Coord<1>(66), 111), buf);

        buf = streak + Coord<1>(34);
        TS_ASSERT_EQUALS(Streak<1>(Coord<1>(700), 745), buf);
    }

    void testAddSub2D()
    {
        Streak<2> streak(Coord<2>(555, 444), 600);

        streak -= Coord<2>(55, 44);
        TS_ASSERT_EQUALS(Streak<2>(Coord<2>(500, 400), 545), streak);

        streak += Coord<2>(166, 155);
        TS_ASSERT_EQUALS(Streak<2>(Coord<2>(666, 555), 711), streak);

        Streak<2> buf = streak - Coord<2>(600, 500);
        TS_ASSERT_EQUALS(Streak<2>(Coord<2>(66, 55), 111), buf);

        buf = streak + Coord<2>(34, 45);
        TS_ASSERT_EQUALS(Streak<2>(Coord<2>(700, 600), 745), buf);
    }

    void testAddSub3D()
    {
        Streak<3> streak(Coord<3>(555, 444, 222), 600);

        streak -= Coord<3>(55, 44, 22);
        TS_ASSERT_EQUALS(Streak<3>(Coord<3>(500, 400, 200), 545), streak);

        streak += Coord<3>(166, 155, 133);
        TS_ASSERT_EQUALS(Streak<3>(Coord<3>(666, 555, 333), 711), streak);

        Streak<3> buf = streak - Coord<3>(600, 500, 300);
        TS_ASSERT_EQUALS(Streak<3>(Coord<3>(66, 55, 33), 111), buf);

        buf = streak + Coord<3>(34, 45, 67);
        TS_ASSERT_EQUALS(Streak<3>(Coord<3>(700, 600, 400), 745), buf);
    }
};

}
