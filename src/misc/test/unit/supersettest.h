#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/superset.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class SuperSetTest : public CxxTest::TestSuite
{
public:

    void testInsertMinMaxAndEraseMin()
    {
        SuperSet<int> set;
        set.insert(2);
        set.insert(1);
        set.insert(3);
        set << 0
            << -1;
        TS_ASSERT_EQUALS(set.max(),  3);
        TS_ASSERT_EQUALS(set.min(), -1);

        set.erase_min();
        TS_ASSERT_EQUALS(set.min(), 0);
    }

    void testOperatorAndAnd()
    {
        SuperSet<int> set0;
        set0.insert(2);
        set0.insert(1);
        set0.insert(3);
        set0.insert(0);

        SuperSet<int> set1;
        set1.insert(2);
        set1.insert(4);
        set1.insert(3);
        set1.insert(0);

        SuperSet<int> expected;
        expected.insert(2);
        expected.insert(3);
        expected.insert(0);

        TS_ASSERT_EQUALS(set0 && set1, expected);
    }

    void testOperatorOrOr()
    {
        SuperSet<int> set0;
        set0.insert(2);
        set0.insert(1);
        set0.insert(3);

        SuperSet<int> set1;
        set1.insert(2);
        set1.insert(0);

        SuperSet<int> expected;
        expected.insert(1);
        expected.insert(2);
        expected.insert(3);
        expected.insert(0);

        TS_ASSERT_EQUALS(set0 || set1, expected);
    }
};

}
