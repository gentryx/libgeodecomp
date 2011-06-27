#include <cxxtest/TestSuite.h>
#include "../../coordset.h"

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class CoordSetTest : public CxxTest::TestSuite
{
public:

    void testInsert()
    {
        CoordSet set;
        set.insert(Coord<2>(0, 0));
        TS_ASSERT_EQUALS(1, (int)set.size());
        set.insert(CoordBox<2>(Coord<2>(0, 0), Coord<2>(2, 3)));
        TS_ASSERT_EQUALS(6, (int)set.size());
        set.insert(CoordBox<2>(Coord<2>(1, 1), Coord<2>(2, 3)));
        TS_ASSERT_EQUALS(10, (int)set.size());
    }

    void testSequence()
    {
        CoordSet set1, set2;
        set1.insert(CoordBox<2>(Coord<2>(-1, 3), Coord<2>(4, 2)));
        for (CoordSet::Sequence s = set1.sequence(); s.hasNext();)
            set2.insert(s.next());
        TS_ASSERT_EQUALS(set1, set2);
    }
};

};
