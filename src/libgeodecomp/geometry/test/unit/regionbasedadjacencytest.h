#include <libgeodecomp/geometry/regionbasedadjacency.h>

#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class RegionBasedAdjacencyTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        RegionBasedAdjacency adjacency;
        adjacency.insert(0, 0);
        adjacency.insert(0, 2);
        adjacency.insert(0, 4);
        adjacency.insert(0, 6);

        adjacency.insert(5, 1);
        adjacency.insert(5, 2);
        adjacency.insert(5, 3);

        adjacency.insert(3, 0);
        adjacency.insert(3, 9);

        std::vector<int> expected;
        std::vector<int> actual;
        expected << 0 << 2 << 4 << 6;
        adjacency.getNeighbors(0, &actual);
        TS_ASSERT_EQUALS(expected, actual);

        expected.clear();
        actual.clear();
        adjacency.getNeighbors(1, &actual);
        TS_ASSERT_EQUALS(expected, actual);
        adjacency.getNeighbors(2, &actual);
        TS_ASSERT_EQUALS(expected, actual);
        adjacency.getNeighbors(4, &actual);
        TS_ASSERT_EQUALS(expected, actual);

        expected.clear();
        actual.clear();
        expected << 0 << 9;
        adjacency.getNeighbors(3, &actual);
        TS_ASSERT_EQUALS(expected, actual);

        expected.clear();
        actual.clear();
        expected << 1 << 2 << 3;
        adjacency.getNeighbors(5, &actual);
        TS_ASSERT_EQUALS(expected, actual);
    }

    void testBulkInsert()
    {
        RegionBasedAdjacency adjacency;
        std::vector<int> neighbors;
        int node;

        node = 4;
        neighbors << 9 << 5 << 4 << 3;
        adjacency.insert(node, neighbors);

        node = 2;
        neighbors.clear();
        neighbors << 1 << 5 << 0 << 8;
        adjacency.insert(node, neighbors);

        node = 7;
        neighbors.clear();
        neighbors << 13;
        adjacency.insert(node, neighbors);

        std::vector<int> actual;
        std::vector<int> expected;

        adjacency.getNeighbors(0, &actual);
        TS_ASSERT_EQUALS(expected, actual);

        adjacency.getNeighbors(1, &actual);
        TS_ASSERT_EQUALS(expected, actual);

        expected << 0 << 1 << 5 << 8;
        adjacency.getNeighbors(2, &actual);
        TS_ASSERT_EQUALS(expected, actual);

        actual.clear();
        expected.clear();
        adjacency.getNeighbors(3, &actual);
        TS_ASSERT_EQUALS(expected, actual);

        expected << 3 << 4 << 5 << 9;
        adjacency.getNeighbors(4, &actual);
        TS_ASSERT_EQUALS(expected, actual);

        actual.clear();
        expected.clear();
        adjacency.getNeighbors(5, &actual);
        TS_ASSERT_EQUALS(expected, actual);

        adjacency.getNeighbors(6, &actual);
        TS_ASSERT_EQUALS(expected, actual);

        expected << 13;
        adjacency.getNeighbors(7, &actual);
        TS_ASSERT_EQUALS(expected, actual);

        actual.clear();
        expected.clear();
        adjacency.getNeighbors(8, &actual);
        TS_ASSERT_EQUALS(expected, actual);

    }

    void testSplitLargeAdjacency()
    {
        RegionBasedAdjacency adjacency(10);

        adjacency.insert(0, 1);
        adjacency.insert(1, 1);
        adjacency.insert(2, 1);
        adjacency.insert(3, 1);
        adjacency.insert(4, 1);
        adjacency.insert(5, 1);
        adjacency.insert(6, 1);
        adjacency.insert(7, 1);
        adjacency.insert(8, 1);
        adjacency.insert(9, 1);

        TS_ASSERT_EQUALS(10, adjacency.size());
        TS_ASSERT_EQUALS(1,  adjacency.regions.size());

        // enforce a split...
        adjacency.insert(-1, 1);

        //...and verify the results:
        TS_ASSERT_EQUALS(11, adjacency.size());
        TS_ASSERT_EQUALS(2,  adjacency.regions.size());

        TS_ASSERT_EQUALS(6,  adjacency.regions[0].size());
        TS_ASSERT_EQUALS(5,  adjacency.regions[1].size());

        TS_ASSERT_EQUALS(2, adjacency.limits.size());
        TS_ASSERT_EQUALS(4, adjacency.limits[0]);

        for (int i = -1; i < 10; ++i) {
            std::vector<int> actual;
            std::vector<int> expected(1, 1);

            adjacency.getNeighbors(i, &actual);
            TS_ASSERT_EQUALS(expected, actual);
        }

        // post split inserts at boundary:
        adjacency.insert(4, 3);
        adjacency.insert(5, 7);

        std::vector<int> actual;
        std::vector<int> expected;
        adjacency.getNeighbors(4, &actual);
        expected << 1 << 3;

        TS_ASSERT_EQUALS(expected, actual);

        actual.clear();
        expected.clear();
        adjacency.getNeighbors(5, &actual);
        expected << 1 << 7;
        TS_ASSERT_EQUALS(expected, actual);

        TS_ASSERT_EQUALS(13, adjacency.size());
        TS_ASSERT_EQUALS( 7, adjacency.regions[0].size());
        TS_ASSERT_EQUALS( 6, adjacency.regions[1].size());

        // provoke another split in regions[0]:
        adjacency.insert(3, -1);
        adjacency.insert(3,  3);
        adjacency.insert(3, -3);
        adjacency.insert(3, -5);

        //...and verify the results:
        TS_ASSERT_EQUALS(17, adjacency.size());
        TS_ASSERT_EQUALS( 3, adjacency.regions.size());

        // split doesn't yield perfect balance as node 3 has more
        // edges than the other nodes. that's still ok as any
        // remaining imbalance can be reduced by further splits.
        TS_ASSERT_EQUALS(4,  adjacency.regions[0].size());
        TS_ASSERT_EQUALS(7,  adjacency.regions[1].size());
        TS_ASSERT_EQUALS(6,  adjacency.regions[2].size());

        TS_ASSERT_EQUALS(3, adjacency.limits.size());
        TS_ASSERT_EQUALS(2, adjacency.limits[0]);
        TS_ASSERT_EQUALS(4, adjacency.limits[1]);

        actual.clear();
        expected.clear();
        adjacency.getNeighbors(3, &actual);
        expected << -5 << -3 << -1 << 1 << 3;
        TS_ASSERT_EQUALS(expected, actual);

        actual.clear();
        expected.clear();
        adjacency.getNeighbors(4, &actual);
        expected << 1 << 3;
        TS_ASSERT_EQUALS(expected, actual);

        actual.clear();
        expected.clear();
        adjacency.getNeighbors(5, &actual);
        expected << 1 << 7;
        TS_ASSERT_EQUALS(expected, actual);

        // finally: force split of regions[2]
        std::vector<int> neighbors;
        neighbors << 10 << 12 << 14 << 16 << 18;
        adjacency.insert(20, neighbors);

        //...and verify the results:
        TS_ASSERT_EQUALS(22, adjacency.size());
        TS_ASSERT_EQUALS( 4, adjacency.regions.size());
        TS_ASSERT_EQUALS(4,  adjacency.regions[0].size());
        TS_ASSERT_EQUALS(7,  adjacency.regions[1].size());
        TS_ASSERT_EQUALS(5,  adjacency.regions[2].size());
        TS_ASSERT_EQUALS(6,  adjacency.regions[3].size());

        actual.clear();
        adjacency.getNeighbors(20, &actual);
        TS_ASSERT_EQUALS(neighbors, actual);

    }
};

}
