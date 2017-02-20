#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/storage/meshlessadapter.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MeshlessAdapterTest : public CxxTest::TestSuite
{
public:

    void test1d()
    {
        typedef MeshlessAdapter<Topologies::Torus<2>::Topology> AdapterType;

        FloatCoord<2> dim(6.5, 1);
        double boxSize = 2;
        AdapterType adapter(dim, boxSize);

        // test posToCoord()
        Coord<2> actual;
        actual = adapter.posToCoord(FloatCoord<2>(0.5, 0.5));
        TS_ASSERT_EQUALS(Coord<2>(0, 0), actual);
        actual = adapter.posToCoord(FloatCoord<2>(1.5, 0.5));
        TS_ASSERT_EQUALS(Coord<2>(0, 0), actual);
        actual = adapter.posToCoord(FloatCoord<2>(2.5, 0.5));
        TS_ASSERT_EQUALS(Coord<2>(1, 0), actual);
        actual = adapter.posToCoord(FloatCoord<2>(5.5, 0.5));
        TS_ASSERT_EQUALS(Coord<2>(2, 0), actual);
        actual = adapter.posToCoord(FloatCoord<2>(6.1, 0.5));
        TS_ASSERT_EQUALS(Coord<2>(2, 0), actual);

        // test distance2()
        double dist;
        FloatCoord<2> pos1(0.5, 0.5);
        FloatCoord<2> pos2(2.1, 0.5);
        dist = adapter.distance2(pos1, pos2);
        TS_ASSERT_EQUALS_DOUBLE(2.56, dist);

        FloatCoord<2> pos3(6.0, 0.5);
        dist = adapter.distance2(pos1, pos3);
        TS_ASSERT_EQUALS_DOUBLE(1.0, dist);

        FloatCoord<2> pos4(1.0, 0.1);
        FloatCoord<2> pos5(1.0, 0.9);
        dist = adapter.distance2(pos4, pos5);
        TS_ASSERT_EQUALS_DOUBLE(0.04, dist);

        // manhattanDistance()
        int distNY;
        distNY = adapter.manhattanDistance(FloatCoord<2>(0.2, 0.4),
                                           FloatCoord<2>(1.8, 0.9));
        TS_ASSERT_EQUALS(distNY, 0);
        distNY = adapter.manhattanDistance(FloatCoord<2>(0.2, 0.4),
                                           FloatCoord<2>(2.1, 0.9));
        TS_ASSERT_EQUALS(distNY, 1);
        distNY = adapter.manhattanDistance(FloatCoord<2>(0.2, 0.4),
                                           FloatCoord<2>(6.1, 0.9));
        TS_ASSERT_EQUALS(distNY, 1);

        // test grid(), insert() and search()
        AdapterType::CoordListGrid grid = adapter.grid();
        TS_ASSERT_EQUALS(Coord<2>(3, 1), grid.getDimensions());

        for (int i = 0; i < 6; ++i)
            adapter.insert(&grid, FloatCoord<2>(i + 0.5, 0.5), i);

        std::set<int> coords;
        std::set<int> expected;
        expected.insert(0);
        expected.insert(1);
        expected.insert(2);
        expected.insert(3);
        bool res = adapter.search(grid, FloatCoord<2>(2.1, 0.5), &coords);
        TS_ASSERT_EQUALS(true, res);
        TS_ASSERT_EQUALS(expected, coords);
    }

    void test1dCube()
    {
        typedef MeshlessAdapter<Topologies::Cube<2>::Topology> AdapterType;

        FloatCoord<2> dim(6.5, 1);
        double boxSize = 2;
        AdapterType adapter(dim, boxSize);

        double dist;
        FloatCoord<2> pos1(0.5, 0.5);
        FloatCoord<2> pos2(6.0, 0.5);
        dist = adapter.distance2(pos1, pos2);
        TS_ASSERT_EQUALS_DOUBLE(5.5 * 5.5, dist);

        FloatCoord<2> pos3(1.0, 0.1);
        FloatCoord<2> pos4(1.0, 0.9);
        dist = adapter.distance2(pos3, pos4);
        TS_ASSERT_EQUALS_DOUBLE(0.8 * 0.8, dist);

        // manhattanDistance()
        int distNY;
        distNY = adapter.manhattanDistance(FloatCoord<2>(0.2, 0.4),
                                           FloatCoord<2>(1.8, 0.9));
        TS_ASSERT_EQUALS(distNY, 0);
        distNY = adapter.manhattanDistance(FloatCoord<2>(0.2, 0.4),
                                           FloatCoord<2>(2.1, 0.9));
        TS_ASSERT_EQUALS(distNY, 1);
        distNY = adapter.manhattanDistance(FloatCoord<2>(0.2, 0.4),
                                           FloatCoord<2>(6.1, 0.9));
        TS_ASSERT_EQUALS(distNY, 2);


    }

    void testBoxSizeDetermination()
    {
        typedef MeshlessAdapter<Topologies::Torus<2>::Topology> AdapterType;

        int width = 20;
        FloatCoord<2> dim(width * 0.5, 1);
        double boxSize = 0.5;
        AdapterType adapter(dim, boxSize);

        AdapterType::CoordVec positions;
        AdapterType::Graph graph;
        for (int i = 0; i < width; ++i) {
            positions.push_back(std::make_pair(FloatCoord<2>(i * 0.5, 0.5), i));
            std::vector<int> neighbors;
            neighbors << ((i + width - 1) % width)
                      << ((i + width + 1) % width);
            graph << neighbors;
        }

        TS_ASSERT(adapter.checkBoxSize(positions, graph));
        TS_ASSERT_EQUALS_DOUBLE(0.5, adapter.findOptimumBoxSize(positions, graph));

        adapter.resetBoxSize(0.47);
        TS_ASSERT_EQUALS_DOUBLE(0.5, adapter.findOptimumBoxSize(positions, graph));
    }
};

}
