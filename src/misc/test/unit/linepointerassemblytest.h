#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/linepointerassembly.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class LinePointerAssemblyTest : public CxxTest::TestSuite 
{
public:
    void testMoore2D()
    {
        Coord<2> dim(7, 5);
        Grid<int> grid(dim, 4711, -1);
        fillGrid(&grid);

        Streak<2> s(Coord<2>(1, 3), 5);
        Coord<2> last(s.endX - 1, s.origin.y());
        int *expected[] = {
            &grid[s.origin + Coord<2>(-1, -1)],
            &grid[s.origin + Coord<2>( 0, -1)],
            &grid[last     + Coord<2>( 1, -1)],
            &grid[s.origin + Coord<2>(-1,  0)],
            &grid[s.origin + Coord<2>( 0,  0)],
            &grid[last     + Coord<2>( 1,  0)],
            &grid[s.origin + Coord<2>(-1,  1)],
            &grid[s.origin + Coord<2>( 0,  1)],
            &grid[last     + Coord<2>( 1,  1)]
        };

        const int *actual[9];
        LinePointerAssembly<Stencils::Moore<2, 1> >()(actual, s, grid);
        for (int i = 0; i < 9; ++i) {
            TS_ASSERT_EQUALS(actual[i], expected[i]);
        }
    }

    // tests of other assemblies intentionally left out as the tests
    // would only clone LinePointerAssembly's behavior. Functionality
    // is covered in LinePointerUpdateFunctor's tests.

private:
    template<class GRID_TYPE>
    void fillGrid(GRID_TYPE *grid)
    {
        Coord<2> dim = grid->getDimensions();
        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                (*grid)[Coord<2>(x, y)] = 1000 + y * 10 + x;
            }
        }
    }
};

}
