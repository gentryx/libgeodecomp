#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>

#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class RecursiveBisectionPartitionTest : public CxxTest::TestSuite
{
public:
    void testSearchNodeCuboid()
    {
        std::vector<std::size_t> weights;
        weights << 25
                << 25;

        checkCuboid(weights, 0, Coord<2>( 0, 0), Coord<2>(15, 20));
        checkCuboid(weights, 1, Coord<2>(15, 0), Coord<2>(15, 20));

        weights << 50;

        checkCuboid(weights, 1, Coord<2>( 0, 10), Coord<2>(15, 10));
        checkCuboid(weights, 2, Coord<2>(15,  0), Coord<2>(15, 20));

        weights << 200;

        checkCuboid(weights, 3, Coord<2>(10,  0), Coord<2>(20, 20));

        weights << 100;

        checkCuboid(weights, 4, Coord<2>(23,  0), Coord<2>( 7, 20));
    }

    void testGetRegion()
    {
        std::vector<std::size_t> weights;
        weights << 200
                << 100
                <<  50
                <<  25
                <<  25;

        Coord<3> origin(100, 200, 300);
        Coord<3> dimensions(64, 32, 64);
        RecursiveBisectionPartition<3> p(
            origin,
            dimensions,
            0,
            weights,
            RecursiveBisectionPartition<3>::AdjacencyPtr(),
            Coord<3>::diagonal(1));

        TS_ASSERT_EQUALS(
            genRegion(100, 200, 300, 32, 32, 64),
            p.getRegion(0));
        // remainder: (132, 200, 300), (32, 32, 64)

        TS_ASSERT_EQUALS(
            genRegion(132, 200, 300, 32, 32, 32),
            p.getRegion(1));
        // remainder: (132, 200, 332), (32, 32, 32)

        TS_ASSERT_EQUALS(
            genRegion(132, 200, 332, 16, 32, 32),
            p.getRegion(2));
        // remainder: (148, 200, 332), (16, 32, 32)

        TS_ASSERT_EQUALS(
            genRegion(148, 200, 332, 16, 16, 32),
            p.getRegion(3));
        // remainder: (148, 216, 332), (16, 16, 32)

        TS_ASSERT_EQUALS(
            genRegion(148, 216, 332, 16, 16, 32),
            p.getRegion(4));
    }

    void testDimWeights()
    {
        std::vector<std::size_t> weights;
        weights << 16
                << 16
                << 16
                << 16;
        Coord<2> dim(96, 32);
        Coord<2> dimWeights(1, 3);

        checkCuboid(weights, 0, Coord<2>( 0,  0), Coord<2>(48, 16), dim, dimWeights);
        checkCuboid(weights, 1, Coord<2>( 0, 16), Coord<2>(48, 16), dim, dimWeights);
        checkCuboid(weights, 2, Coord<2>(48,  0), Coord<2>(48, 16), dim, dimWeights);
        checkCuboid(weights, 3, Coord<2>(48, 16), Coord<2>(48, 16), dim, dimWeights);
    }

    void testDegradedDimensions()
    {
        std::vector<std::size_t> weights;
        weights << 0
                << 0;

        Coord<3> dim(128, 0, 0);
        TS_ASSERT_THROWS(
            RecursiveBisectionPartition<3> p(
                Coord<3>(),
                dim,
                0,
                weights),
            std::invalid_argument&);
    }

    void checkCuboid(
        std::vector<std::size_t> weights,
        long node,
        Coord<2> expectedOffset,
        Coord<2> expectedDim,
        Coord<2> dimensions=Coord<2>(30, 20),
        Coord<2> dimWeights=Coord<2>::diagonal(1))
    {
        Coord<2> origin(0, 0);
        RecursiveBisectionPartition<2> p(
            origin,
            dimensions,
            0,
            weights,
            RecursiveBisectionPartition<2>::AdjacencyPtr(),
            dimWeights);

        TS_ASSERT_EQUALS(
            CoordBox<2>(expectedOffset, expectedDim),
            p.searchNodeCuboid(
                p.startOffsets.begin(),
                p.startOffsets.end() - 1,
                p.startOffsets.begin() + node,
                CoordBox<2>(origin, dimensions)));
    }

    Region<3> genRegion(int o1, int o2, int o3, int d1, int d2, int d3)
    {
        CoordBox<3> box(Coord<3>(o1, o2, o3), Coord<3>(d1, d2, d3));
        Region<3> r;
        r << box;

        return r;
    }


};

}
