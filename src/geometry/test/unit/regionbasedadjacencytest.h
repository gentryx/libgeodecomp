#include <libgeodecomp/geometry/regionbasedadjacency.h>

#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>

using namespace boost::assign;
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
        adjacency.getNeightbors(0, &actual);
        TS_ASSERT_EQUALS(expected, actual);

        expected.clear();
        actual.clear();
        adjacency.getNeightbors(1, &actual);
        TS_ASSERT_EQUALS(expected, actual);
        adjacency.getNeightbors(2, &actual);
        TS_ASSERT_EQUALS(expected, actual);
        adjacency.getNeightbors(4, &actual);
        TS_ASSERT_EQUALS(expected, actual);

        expected.clear();
        actual.clear();
        expected << 0 << 9;
        adjacency.getNeightbors(3, &actual);
        TS_ASSERT_EQUALS(expected, actual);

        expected.clear();
        actual.clear();
        expected << 1 << 2 << 3;
        adjacency.getNeightbors(5, &actual);
        TS_ASSERT_EQUALS(expected, actual);
    }
};

}
