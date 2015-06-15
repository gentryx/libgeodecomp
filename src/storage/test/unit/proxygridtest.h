#include <cxxtest/TestSuite.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/proxygrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class ProxyGridTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        std::cout << "============================================================================\n";
        DisplacedGrid<int> mainGrid(CoordBox<2>(Coord<2>(-1, -2), Coord<2>(22, 14)));
        ProxyGrid<int, 2> subGrid(mainGrid, CoordBox<2>(Coord<2>(0, 0), Coord<2>(20, 10)));

        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(0, 0), Coord<2>(20, 10)), subGrid.boundingBox());
        subGrid.setEdge(47);
        TS_ASSERT_EQUALS(47, subGrid.getEdge());
        TS_ASSERT_EQUALS(47, mainGrid.getEdge());
    }
};

}
