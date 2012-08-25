#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/misc/newregion.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>

using namespace boost::assign;
using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {

class NewRegionTest : public CxxTest::TestSuite 
{
public:
    void testInsert()
    {
        NewRegion<3> r;
        TS_ASSERT_EQUALS(r.indices[0].size(), 0);
        TS_ASSERT_EQUALS(r.indices[1].size(), 0);
        TS_ASSERT_EQUALS(r.indices[2].size(), 0);

        r << Streak<3>(Coord<3>(10, 20, 30), 40);
        TS_ASSERT_EQUALS(r.indices[0].size(), 1);
        TS_ASSERT_EQUALS(r.indices[1].size(), 1);
        TS_ASSERT_EQUALS(r.indices[2].size(), 1);

        r << Streak<3>(Coord<3>(12, 22, 31), 42);
        TS_ASSERT_EQUALS(r.indices[0].size(), 2);
        TS_ASSERT_EQUALS(r.indices[1].size(), 2);
        TS_ASSERT_EQUALS(r.indices[2].size(), 2);

        r << Streak<3>(Coord<3>(14, 24, 29), 44);
        TS_ASSERT_EQUALS(r.indices[0].size(), 3);
        TS_ASSERT_EQUALS(r.indices[1].size(), 3);
        TS_ASSERT_EQUALS(r.indices[2].size(), 3);

        r << Streak<3>(Coord<3>(16, 21, 30), 46);
        TS_ASSERT_EQUALS(r.indices[0].size(), 4);
        TS_ASSERT_EQUALS(r.indices[1].size(), 4);
        TS_ASSERT_EQUALS(r.indices[2].size(), 3);

        r << Streak<3>(Coord<3>(58, 20, 30), 68);
        TS_ASSERT_EQUALS(r.indices[0].size(), 5);
        TS_ASSERT_EQUALS(r.indices[1].size(), 4);
        TS_ASSERT_EQUALS(r.indices[2].size(), 3);

        r << Streak<3>(Coord<3>(59, 19, 29), 69);
        TS_ASSERT_EQUALS(r.indices[0].size(), 6);
        TS_ASSERT_EQUALS(r.indices[1].size(), 5);
        TS_ASSERT_EQUALS(r.indices[2].size(), 3);

        r << Streak<3>(Coord<3>(38, 20, 30), 60);
        TS_ASSERT_EQUALS(r.indices[0].size(), 5);
        TS_ASSERT_EQUALS(r.indices[1].size(), 5);
        TS_ASSERT_EQUALS(r.indices[2].size(), 3);
    }

    void testRemove()
    {
        NewRegion<3> r;
        r << Streak<3>(Coord<3>(10, 20, 30), 40);
        TS_ASSERT_EQUALS(r.indices[0].size(), 1);
        TS_ASSERT_EQUALS(r.indices[1].size(), 1);
        TS_ASSERT_EQUALS(r.indices[2].size(), 1);

        r >> Streak<3>(Coord<3>(15, 20, 30), 35);
        TS_ASSERT_EQUALS(r.indices[0].size(), 2);
        TS_ASSERT_EQUALS(r.indices[1].size(), 1);
        TS_ASSERT_EQUALS(r.indices[2].size(), 1);

        r >> Streak<3>(Coord<3>(36, 20, 30), 37);
        TS_ASSERT_EQUALS(r.indices[0].size(), 3);
        TS_ASSERT_EQUALS(r.indices[1].size(), 1);
        TS_ASSERT_EQUALS(r.indices[2].size(), 1);

        r >> Streak<3>(Coord<3>(30, 20, 30), 50);
        TS_ASSERT_EQUALS(r.indices[0].size(), 1);
        TS_ASSERT_EQUALS(r.indices[1].size(), 1);
        TS_ASSERT_EQUALS(r.indices[2].size(), 1);

        r << Streak<3>(Coord<3>(40, 21, 29), 60);
        TS_ASSERT_EQUALS(r.indices[0].size(), 2);
        TS_ASSERT_EQUALS(r.indices[1].size(), 2);
        TS_ASSERT_EQUALS(r.indices[2].size(), 2);

        r >> Streak<3>(Coord<3>(50, 21, 29), 55);
        TS_ASSERT_EQUALS(r.indices[0].size(), 3);
        TS_ASSERT_EQUALS(r.indices[1].size(), 2);
        TS_ASSERT_EQUALS(r.indices[2].size(), 2);
    }

    void testStreakIterator()
    {
        SuperVector<Streak<3> > expected;
        SuperVector<Streak<3> > actual;

        NewRegion<3> r;
        TS_ASSERT_EQUALS(r.beginStreak(), r.endStreak());


        Streak<3> newStreak(Coord<3>(10, 10, 10), 20);
        r << newStreak;
        TS_ASSERT_EQUALS(newStreak, *r.beginStreak());
        TS_ASSERT_EQUALS(1, r.size());
        for (NewRegion<3>::StreakIterator i = r.beginStreak(); i != r.endStreak(); ++i) {
            actual << *i;
        }
        expected << newStreak;
        TS_ASSERT_EQUALS(actual, expected);

        newStreak = Streak<3>(Coord<3>(10, 20, 10), 20);
        expected << newStreak;
        r << newStreak;
        TS_ASSERT_EQUALS(2, r.size());
        actual.clear();
        for (NewRegion<3>::StreakIterator i = r.beginStreak(); i != r.endStreak(); ++i) {
            actual << *i;
        }
        TS_ASSERT_EQUALS(actual, expected);

        newStreak = Streak<3>(Coord<3>(30, 20, 10), 40);
        expected << newStreak;
        r << newStreak;
        TS_ASSERT_EQUALS(3, r.size());
        actual.clear();
        for (NewRegion<3>::StreakIterator i = r.beginStreak(); i != r.endStreak(); ++i) {
            actual << *i;
        }
        TS_ASSERT_EQUALS(actual, expected);

        std::cout << "----------------------------------------------------------\n";

        newStreak = Streak<3>(Coord<3>(10, 20, 11), 20);
        expected << newStreak;
        r << newStreak;
        std::cout << "----------------------------------------------------------\n";
        TS_ASSERT_EQUALS(4, r.size());
        actual.clear();
        std::cout << "----------------------------------------------------------\n";
        for (NewRegion<3>::StreakIterator i = r.beginStreak(); i != r.endStreak(); ++i) {
            actual << *i;
            std::cout << "actuak: " << *i << "\n";
        }
        std::cout << "expected: " << expected << "\n";
        std::cout << "actual: " << actual << "\n";
        std::cout << "r: " << r << "\n";
        TS_ASSERT_EQUALS(actual, expected);
    }

};

}
