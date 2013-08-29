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

    void testConstIterator()
    {
	Grid<int> grid1(Coord<2>(5, 3));
	for (int y = 0; y < 3; ++y) {
	    for (int x = 0; x < 5; ++x) {
		grid1[Coord<2>(x, y)] = x * 10 + y;
	    }
	}

	const Grid<int>& grid = grid1;

	GridBase<int, 2>::ConstIterator iter = grid.at(Coord<2>(2, 1));
	for (int i = 0; i < 3; ++i) {
	    TS_ASSERT_EQUALS(*iter, (i + 2) * 10 + 1);

	    ++iter;
	}

	iter = grid.at(Coord<2>(2, 0));
	for (int i = 0; i < 3; ++i) {
	    int res;
	    iter >> res;
	    TS_ASSERT_EQUALS(res, (i + 2) * 10 + 0);
	}

	Grid<std::pair<int, char> > grid2(Coord<2>(2, 2), std::make_pair(123, 'a'));
	TS_ASSERT_EQUALS(grid2.at(Coord<2>(1, 1))->first, 123);
	TS_ASSERT_EQUALS(grid2.at(Coord<2>(1, 1))->second, 'a');
    }

    void testIterator()
    {
	Grid<int> grid(Coord<2>(5, 3));
	for (int y = 0; y < 3; ++y) {
	    for (int x = 0; x < 5; ++x) {
		grid[Coord<2>(x, y)] = x * 10 + y;
	    }
	}

	GridBase<int, 2>::Iterator iter = grid.at(Coord<2>(2, 1));
	for (int i = 0; i < 3; ++i) {
	    TS_ASSERT_EQUALS(*iter, (i + 2) * 10 + 1);

	    ++iter;
	}

	iter = grid.at(Coord<2>(2, 0));
	for (int i = 0; i < 3; ++i) {
	    int res;
	    iter >> res;
	    TS_ASSERT_EQUALS(res, (i + 2) * 10 + 0);
	}

	Grid<std::pair<int, char> > grid2(Coord<2>(2, 2), std::make_pair(123, 'a'));
	TS_ASSERT_EQUALS(grid2.at(Coord<2>(1, 1))->first, 123);
	TS_ASSERT_EQUALS(grid2.at(Coord<2>(1, 1))->second, 'a');
    }
};

}
