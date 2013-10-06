#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/fixedarray.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class FixedArrayTest : public CxxTest::TestSuite
{
public:

    void testInsertDelete()
    {
        FixedArray<int, 20> a;
        TS_ASSERT_EQUALS(0, a.size());

        a << 0
          << 1
          << 2
          << 3;
        TS_ASSERT_EQUALS(4, a.size());
        TS_ASSERT_EQUALS(0, a[0]);
        TS_ASSERT_EQUALS(1, a[1]);
        TS_ASSERT_EQUALS(2, a[2]);
        TS_ASSERT_EQUALS(3, a[3]);

        a.erase(&a[1]);
        TS_ASSERT_EQUALS(3, a.size());
        TS_ASSERT_EQUALS(0, a[0]);
        TS_ASSERT_EQUALS(2, a[1]);
        TS_ASSERT_EQUALS(3, a[2]);

        FixedArray<int, 10> b;
        TS_ASSERT_THROWS(b.reserve(11), std::out_of_range);

        b.reserve(3);
        TS_ASSERT_EQUALS(3, b.size());
        std::copy(a.begin(), a.end(), b.begin());
        TS_ASSERT_EQUALS(0, b[0]);
        TS_ASSERT_EQUALS(2, b[1]);
        TS_ASSERT_EQUALS(3, b[2]);

        a.clear();
        TS_ASSERT_EQUALS(0, a.size());
    }

    void testConstructors()
    {
        FixedArray<int, 4> a(1, 13);
        FixedArray<int, 4> b(4, 47);
        FixedArray<int, 4> c(3, 11);

        TS_ASSERT_EQUALS(a.size(), 1);
        TS_ASSERT_EQUALS(a[0], 13);

        TS_ASSERT_EQUALS(b.size(), 4);
        TS_ASSERT_EQUALS(b[0], 47);
        TS_ASSERT_EQUALS(b[1], 47);
        TS_ASSERT_EQUALS(b[2], 47);
        TS_ASSERT_EQUALS(b[3], 47);

        TS_ASSERT_EQUALS(c.size(), 3);
        TS_ASSERT_EQUALS(c[0], 11);
        TS_ASSERT_EQUALS(c[1], 11);
        TS_ASSERT_EQUALS(c[2], 11);

        a = b;
        TS_ASSERT_EQUALS(a.size(), 4);
        TS_ASSERT_EQUALS(a[0], 47);
        TS_ASSERT_EQUALS(a[1], 47);
        TS_ASSERT_EQUALS(a[2], 47);
        TS_ASSERT_EQUALS(a[3], 47);

        a = c;
        TS_ASSERT_EQUALS(a.size(), 3);
        TS_ASSERT_EQUALS(a[0], 11);
        TS_ASSERT_EQUALS(a[1], 11);
        TS_ASSERT_EQUALS(a[2], 11);
    }
};

}
