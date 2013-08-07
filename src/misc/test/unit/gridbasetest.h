#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/grid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class GridBaseTest : public CxxTest::TestSuite
{
public:
    void testEqualsAndDiffersOperator()
    {
        Grid<int> a(Coord<2>(2, 2),  5, 1);
        Grid<int> b(Coord<2>(2, 2),  5, 2);
        Grid<int> c(Coord<2>(2, 2),  5, 1);
        Grid<int> d(Coord<2>(10, 5), 5, 1);

        GridBase<int, 2>& refA = a;
        GridBase<int, 2>& refB = b;
        GridBase<int, 2>& refC = c;
        GridBase<int, 2>& refD = d;

        TS_ASSERT( (refA == refC));
        c[Coord<2>(0, 1)] = 0;

        TS_ASSERT_DIFFERS(a, b);
        TS_ASSERT_DIFFERS(a, c);
        TS_ASSERT_DIFFERS(a, d);
        // we can't use TS_ASSERT_DIFFERS here, as it won't work with
        // references of abstract types:
        TS_ASSERT( (refA == refA));
        TS_ASSERT(!(refA == refB));
        TS_ASSERT(!(refA == refC));
        TS_ASSERT(!(refA == refD));

        b = a;
        c = a;
        d = a;

        TS_ASSERT_EQUALS(a, b);
        TS_ASSERT_EQUALS(a, c);
        TS_ASSERT_EQUALS(a, d);
        TS_ASSERT(refA == refA);
        TS_ASSERT(refA == refB);
        TS_ASSERT(refA == refC);
        TS_ASSERT(refA == refD);
    }
};

}
