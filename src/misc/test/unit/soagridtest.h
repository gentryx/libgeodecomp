#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/soagrid.h>

using namespace LibGeoDecomp;

class SoATestCell
{
public:
    SoATestCell(int v = 0) :
        v(v)
    {}

    bool operator==(const SoATestCell& other)
    {
        return v == other.v;
    }

    int v;
};

LIBFLATARRAY_REGISTER_SOA(SoATestCell, ((int)(v)))

namespace LibGeoDecomp {


class SoAGridTest : public CxxTest::TestSuite
{
public:

    void testBasic()
    {
        CoordBox<3> box(Coord<3>(10, 15, 22), Coord<3>(50, 40, 35));
        SoATestCell defaultCell(1);
        SoATestCell edgeCell(2);

        SoAGrid<SoATestCell, Topologies::Cube<3>::Topology> grid(box, defaultCell, edgeCell);
        grid.set(Coord<3>(1, 1, 1) + box.origin, 3);
        grid.set(Coord<3>(2, 2, 3) + box.origin, 4);

        TS_ASSERT_EQUALS(grid.boundingBox(), box);
        TS_ASSERT_EQUALS(grid.get(Coord<3>(0, 0, 0)), edgeCell);
        TS_ASSERT_EQUALS(grid.get(Coord<3>(0, 0, 0) + box.origin), defaultCell);
        TS_ASSERT_EQUALS(grid.get(Coord<3>(1, 1, 1) + box.origin), 3);
        TS_ASSERT_EQUALS(grid.get(Coord<3>(2, 2, 3) + box.origin), 4);
    }

    void test2d()
    {
    }

    void testDisplacementWithTopologicalCorrectness()
    {
    }

    // fixme: test reset of edge cell
    // fixme: 2d test: check that z-dim == 1
    // fixme: test topological correctness
    // fixme: check that setEdge won't erase interior
    // fixme: check neighborhood may actually access edgecells
    // fixme: cover stencil radius > 1
};

}
