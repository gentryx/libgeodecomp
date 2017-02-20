#include <libgeodecomp/geometry/partitions/hilbertpartition.h>

#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class HilbertPartitionTest : public CxxTest::TestSuite
{
public:
    typedef std::vector<Coord<2> > CoordVector;

    void setUp()
    {
        partition = HilbertPartition(Coord<2>(10, 20), Coord<2>(4, 4));
        expected.clear();
        expected << Coord<2>(10, 23)
                 << Coord<2>(11, 23)
                 << Coord<2>(11, 22)
                 << Coord<2>(10, 22)
                 << Coord<2>(10, 21)
                 << Coord<2>(10, 20)
                 << Coord<2>(11, 20)
                 << Coord<2>(11, 21)
                 << Coord<2>(12, 21)
                 << Coord<2>(12, 20)
                 << Coord<2>(13, 20)
                 << Coord<2>(13, 21)
                 << Coord<2>(13, 22)
                 << Coord<2>(12, 22)
                 << Coord<2>(12, 23)
                 << Coord<2>(13, 23);
        actual.clear();
    }

    void testFillRectangles()
    {
        CoordVector actual;
        for (int i = 0; i < 16; ++i) {
            actual <<
                *HilbertPartition::Iterator(
                    Coord<2>(10, 20),
                    Coord<2>(4, 4),
                    i);
        }
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testOperatorInc()
    {
        HilbertPartition::Iterator i(Coord<2>(10, 10), Coord<2>(4, 4), 10);
        TS_ASSERT_EQUALS(Coord<2>(13, 10), *i);
        ++i;
        TS_ASSERT_EQUALS(Coord<2>(13, 11), *i);
        ++i;
        TS_ASSERT_EQUALS(Coord<2>(13, 12), *i);
        ++i;
        TS_ASSERT_EQUALS(Coord<2>(12, 12), *i);
        ++i;
        TS_ASSERT_EQUALS(Coord<2>(12, 13), *i);
        ++i;
        TS_ASSERT_EQUALS(Coord<2>(13, 13), *i);
    }

    void testLoop()
    {
        CoordVector actual;
        for (HilbertPartition::Iterator i = partition.begin(); i != partition.end(); ++i)
            actual.push_back(*i);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testAsymmetric()
    {
        partition = HilbertPartition(Coord<2>(10, 22), Coord<2>(5, 3));
        // 45678
        // 32b9a
        // 01cde
        expected.clear();
        expected << Coord<2>(10, 24)
                 << Coord<2>(11, 24)
                 << Coord<2>(11, 23)
                 << Coord<2>(10, 23)
                 << Coord<2>(10, 22)
                 << Coord<2>(11, 22)
                 << Coord<2>(12, 22)
                 << Coord<2>(13, 22)
                 << Coord<2>(14, 22)
                 << Coord<2>(13, 23)
                 << Coord<2>(14, 23)
                 << Coord<2>(12, 23)
                 << Coord<2>(12, 24)
                 << Coord<2>(13, 24)
                 << Coord<2>(14, 24);

        for (HilbertPartition::Iterator i = partition.begin();
             i != partition.end();
             ++i) {
            actual.push_back(*i);
        }

        TS_ASSERT_EQUALS(expected, actual);
    }

    void testSquareBracketsOperatorVersusIteration()
    {
        Coord<2> offset(10, 20);
        Coord<2> dimensions(6, 35);
        partition = HilbertPartition(offset, dimensions);

        CoordVector expected;
        for (int i = 0; i < (dimensions.x()*dimensions.y()); ++i) {
            expected << *partition[i];
        }

        CoordVector actual;
        for (HilbertPartition::Iterator i = partition.begin();
             i != partition.end(); ++i) {
            actual << *i;
        }
        TS_ASSERT_EQUALS(expected, actual);
    }

    void testLarge()
    {
        Coord<2> offset(10, 20);
        Coord<2> dimensions(600, 3500);
        partition = HilbertPartition(offset, dimensions);
        CoordVector expectedSorted;
        for (int x = offset.x(); x < (offset.x() + dimensions.x()); ++x) {
            for (int y = offset.y(); y < (offset.y() + dimensions.y()); ++y) {
                expectedSorted << Coord<2>(x, y);
            }
        }

        sort(expectedSorted);

        CoordVector actual;
        for (HilbertPartition::Iterator i = partition.begin(); i != partition.end(); ++i) {
            actual << *i;
        }

        sort(actual);
        TS_ASSERT_EQUALS(expectedSorted, actual);
    }

private:
    HilbertPartition partition;
    CoordVector expected, actual;
};

}
