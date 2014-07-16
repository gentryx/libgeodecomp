#include <libgeodecomp/storage/fixedarray.h>
#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/neighborhooditerator.h>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class NeighborhoodIteratorTest : public CxxTest::TestSuite
{
public:
    typedef FixedArray<double, 10> Container;

    void setUp()
    {
        grid = Grid<Container>(Coord<2>(10, 5));

        {
            Container& c = grid[Coord<2>(0, 0)];
            c << 1;
        }

        {
            Container& c = grid[Coord<2>(1, 0)];
            c << 2
              << 3;
        }

        {
            Container& c = grid[Coord<2>(2, 0)];
            c << 4
              << 5
              << 6;
        }

        {
            Container& c = grid[Coord<2>(0, 1)];
            c << 7
              << 8
              << 9
              << 10;
        }

        {
            Container& c = grid[Coord<2>(1, 1)];
            c << 11
              << 12
              << 13
              << 14
              << 15;
        }

        {
            Container& c = grid[Coord<2>(2, 1)];
            c << 16
              << 17
              << 18
              << 19
              << 20
              << 21;
        }

        {
            Container& c = grid[Coord<2>(0, 2)];
            c << 22
              << 23
              << 24
              << 25
              << 26
              << 27
              << 28;
        }

        {
            Container& c = grid[Coord<2>(1, 2)];
            c << 29
              << 30
              << 31
              << 32
              << 33
              << 34
              << 35
              << 36;
        }

        {
            Container& c = grid[Coord<2>(2, 2)];
            c << 37
              << 38
              << 39
              << 40
              << 41
              << 42
              << 43
              << 44
              << 45;
        }

        TS_ASSERT_EQUALS(grid[Coord<2>(0, 0)].size(),  std::size_t(1));
        TS_ASSERT_EQUALS(grid[Coord<2>(1, 0)].size(),  std::size_t(2));
        TS_ASSERT_EQUALS(grid[Coord<2>(2, 0)].size(),  std::size_t(3));
        TS_ASSERT_EQUALS(grid[Coord<2>(0, 1)].size(),  std::size_t(4));
        TS_ASSERT_EQUALS(grid[Coord<2>(1, 1)].size(),  std::size_t(5));
        TS_ASSERT_EQUALS(grid[Coord<2>(2, 1)].size(),  std::size_t(6));
        TS_ASSERT_EQUALS(grid[Coord<2>(0, 2)].size(),  std::size_t(7));
        TS_ASSERT_EQUALS(grid[Coord<2>(1, 2)].size(),  std::size_t(8));
        TS_ASSERT_EQUALS(grid[Coord<2>(2, 2)].size(),  std::size_t(9));
    }

    void testBasic()
    {
        typedef CoordMap<Container, Grid<Container> > Neighborhood;
        typedef NeighborhoodIterator<Neighborhood, double, 2> HoodIterator;

        Neighborhood hood = grid.getNeighborhood(Coord<2>(1, 1));
        HoodIterator begin = HoodIterator::begin(hood);
        HoodIterator end = HoodIterator::end(hood);
        HoodIterator iter = begin;
        TS_ASSERT_DIFFERS(iter, end);

        // cell -1, -1
        TS_ASSERT_EQUALS(*iter.boxIterator, Coord<2>(-1, -1));
        TS_ASSERT_EQUALS(*iter, 1);

        // cell  0, -1
        ++iter;
        TS_ASSERT_EQUALS(*iter.boxIterator, Coord<2>( 0, -1));
        TS_ASSERT_EQUALS(*iter, 2);
        ++iter;
        TS_ASSERT_EQUALS(*iter.boxIterator, Coord<2>( 0, -1));

        // cell  1, -1
        ++iter;
        TS_ASSERT_EQUALS(*iter.boxIterator, Coord<2>( 1, -1));
        TS_ASSERT_EQUALS(*iter, 4);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 5);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 6);

        // cell -1,  0
        ++iter;
        TS_ASSERT_EQUALS(*iter.boxIterator, Coord<2>(-1,  0));
        TS_ASSERT_EQUALS(*iter, 7);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 8);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 9);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 10);

        // cell  0,  0
        ++iter;
        TS_ASSERT_EQUALS(*iter.boxIterator, Coord<2>( 0,  0));
        TS_ASSERT_EQUALS(*iter, 11);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 12);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 13);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 14);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 15);

        // cell  1,  0
        ++iter;
        TS_ASSERT_EQUALS(*iter.boxIterator, Coord<2>( 1,  0));
        TS_ASSERT_EQUALS(*iter, 16);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 17);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 18);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 19);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 20);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 21);

        // cell  -1,  1
        ++iter;
        TS_ASSERT_EQUALS(*iter.boxIterator, Coord<2>(-1,  1));
        TS_ASSERT_EQUALS(*iter, 22);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 23);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 24);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 25);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 26);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 27);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 28);

        // cell  0,  1
        ++iter;
        TS_ASSERT_EQUALS(*iter.boxIterator, Coord<2>( 0,  1));
        TS_ASSERT_EQUALS(*iter, 29);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 30);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 31);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 32);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 33);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 34);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 35);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 36);

        // cell  1,  1
        ++iter;
        TS_ASSERT_EQUALS(*iter.boxIterator, Coord<2>( 1,  1));
        TS_ASSERT_EQUALS(*iter, 37);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 38);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 39);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 40);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 41);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 42);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 43);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 44);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 45);

        ++iter;
        TS_ASSERT_EQUALS(iter, end);
    }

    void testSkippingOfEmptyCells()
    {
        typedef CoordMap<Container, Grid<Container> > Neighborhood;
        typedef NeighborhoodIterator<Neighborhood, double, 2> HoodIterator;

        grid[Coord<2>(0, 0)].clear();
        grid[Coord<2>(0, 1)].clear();

        Neighborhood hood = grid.getNeighborhood(Coord<2>(1, 1));
        HoodIterator begin = HoodIterator::begin(hood);
        HoodIterator iter = begin;

        // skipping one value from (0, 0)

        TS_ASSERT_EQUALS(*iter, 2);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 3);

        ++iter;
        TS_ASSERT_EQUALS(*iter, 4);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 5);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 6);

        // skipping four values from (0, 1)

        ++iter;
        TS_ASSERT_EQUALS(*iter, 11);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 12);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 13);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 14);
        ++iter;
        TS_ASSERT_EQUALS(*iter, 15);
    }

    void testSkippingOfAllEmptyCells()
    {
        typedef CoordMap<Container, Grid<Container> > Neighborhood;
        typedef NeighborhoodIterator<Neighborhood, double, 2> HoodIterator;

        grid[Coord<2>(0, 0)].clear();
        grid[Coord<2>(1, 0)].clear();
        grid[Coord<2>(2, 0)].clear();
        grid[Coord<2>(0, 1)].clear();
        grid[Coord<2>(1, 1)].clear();
        grid[Coord<2>(2, 1)].clear();
        grid[Coord<2>(0, 2)].clear();
        grid[Coord<2>(1, 2)].clear();
        grid[Coord<2>(2, 2)].clear();


        Neighborhood hood = grid.getNeighborhood(Coord<2>(1, 1));

        HoodIterator begin = HoodIterator::begin(hood);
        HoodIterator end = HoodIterator::end(hood);

        TS_ASSERT_EQUALS(begin, end);
    }

private:
    Grid<Container> grid;
};

}
