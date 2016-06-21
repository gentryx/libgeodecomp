#include <cxxtest/TestSuite.h>
#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CUDA
#include <cuda.h>
#endif

#include <libgeodecomp/storage/cudasoagrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CUDASoAGridTest : public CxxTest::TestSuite
{
public:
    // fixme: check out-of-range access for cube and torus

    void testScalarGetSet()
    {
        Coord<3> dim(100, 50, 30);
        Coord<3> origin(200, 210, 220);
        CoordBox<3> box(origin, dim);

        CUDASoAGrid<TestCellSoA, Topologies::Cube<3>::Topology> grid(box);

        TestCellSoA expected0(
            Coord<3>(1, 2, 3),
            Coord<3>(4, 5, 6),
            7,
            8);
        TestCellSoA expected1(
            Coord<3>( 9, 10, 11),
            Coord<3>(12, 13, 14),
            15,
            16);
        TestCellSoA expected2(
            Coord<3>(17, 18, 19),
            Coord<3>(20, 21, 22),
            23,
            24);
        grid.set(Coord<3>(200, 210, 220), expected0);
        grid.set(Coord<3>(299, 210, 220), expected1);
        grid.set(Coord<3>(299, 259, 249), expected2);

        TestCellSoA actual;
        actual= grid.get(Coord<3>(200, 210, 220));
        TS_ASSERT_EQUALS(expected0, actual);
        actual= grid.get(Coord<3>(299, 210, 220));
        TS_ASSERT_EQUALS(expected1, actual);
        actual= grid.get(Coord<3>(299, 259, 249));
        TS_ASSERT_EQUALS(expected2, actual);
     }

    void testGetSetMultiple()
    {
        
    }

};

}
