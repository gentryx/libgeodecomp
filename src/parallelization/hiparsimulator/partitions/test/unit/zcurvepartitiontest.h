#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/zcurvepartition.h>

using namespace boost::assign;
using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

class ZCurvePartitionTest : public CxxTest::TestSuite 
{
public:
    typedef SuperVector<Coord<2> > CoordVector;

    void setUp()
    {
        partition = ZCurvePartition<2>(Coord<2>(10, 20), Coord<2>(4, 4));
        expected.clear();
        expected += 
            Coord<2>(10, 20), Coord<2>(11, 20), Coord<2>(10, 21), Coord<2>(11, 21), 
            Coord<2>(12, 20), Coord<2>(13, 20), Coord<2>(12, 21), Coord<2>(13, 21), 
            Coord<2>(10, 22), Coord<2>(11, 22), Coord<2>(10, 23), Coord<2>(11, 23), 
            Coord<2>(12, 22), Coord<2>(13, 22), Coord<2>(12, 23), Coord<2>(13, 23);
        actual.clear();
    }

    void testFillRectangles()
    {
        CoordVector actual;
        for (int i = 0; i < 16; ++i) 
            actual.push_back(*ZCurvePartition<2>::Iterator(Coord<2>(10, 20), Coord<2>(4, 4), i));
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testOperatorInc()
    {
        ZCurvePartition<2>::Iterator i(Coord<2>(10, 10), Coord<2>(4, 4), 10);
        TS_ASSERT_EQUALS(Coord<2>(10, 13), *i);
        ++i;
        TS_ASSERT_EQUALS(Coord<2>(11, 13), *i);
        ++i;
        TS_ASSERT_EQUALS(Coord<2>(12, 12), *i);
        ++i;
        TS_ASSERT_EQUALS(Coord<2>(13, 12), *i);
        ++i;
        TS_ASSERT_EQUALS(Coord<2>(12, 13), *i);
        ++i;
        TS_ASSERT_EQUALS(Coord<2>(13, 13), *i);
    }

    void testLoop()
    {
        CoordVector actual;
        for (ZCurvePartition<2>::Iterator i = partition.begin(); i != partition.end(); ++i) 
            actual.push_back(*i);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testAsymmetric()
    {
        partition = ZCurvePartition<2>(Coord<2>(10, 20), Coord<2>(5, 3));
        // 01234
        // 569ab
        // 78cde
        expected.clear();
        expected += 
            Coord<2>(10, 20), Coord<2>(11, 20), 
            Coord<2>(12, 20), Coord<2>(13, 20), Coord<2>(14, 20), 
            Coord<2>(10, 21), Coord<2>(11, 21), Coord<2>(10, 22), Coord<2>(11, 22), 
            Coord<2>(12, 21), Coord<2>(13, 21), Coord<2>(14, 21), Coord<2>(12, 22), Coord<2>(13, 22), Coord<2>(14, 22);
        for (ZCurvePartition<2>::Iterator i = partition.begin(); i != partition.end(); ++i) 
            actual.push_back(*i);
        TS_ASSERT_EQUALS(expected, actual);
    }

    void testSquareBracketsOperatorVersusIteration()
    {
        Coord<2> offset(10, 20);
        Coord<2> dimensions(6, 35);
        partition = ZCurvePartition<2>(offset, dimensions);
        CoordVector expected;
        for (int i = 0; i < (dimensions.x() * dimensions.y()); ++i)
            expected += *partition[i];
        CoordVector actual;

        for (ZCurvePartition<2>::Iterator i = partition.begin(); i != partition.end(); ++i) {
            actual.push_back(*i);
        }

        TS_ASSERT_EQUALS(expected, actual);
    }

    void testLarge()
    {
        partition = ZCurvePartition<2>(Coord<2>(10, 20), Coord<2>(600, 3500));
        CoordVector expectedSorted;
        for (int x = 10; x < 610; ++x)
            for (int y = 20; y < 3520; ++y)
                expectedSorted += Coord<2>(x, y);
        expectedSorted.sort();
        CoordVector actual;
        for (ZCurvePartition<2>::Iterator i = partition.begin(); i != partition.end(); ++i) 
            actual.push_back(*i);
        actual.sort();
        TS_ASSERT_EQUALS(expectedSorted, actual);
    }

    void test3dSimple()
    {
        ZCurvePartition<3> partition(Coord<3>(1, 2, 3), Coord<3>(2, 2, 2));

        SuperVector<Coord<3> > actual1;
        for (int i = 0; i < 8; ++i)
            actual1 << *partition[i];

        SuperVector<Coord<3> > actual2;
        for (ZCurvePartition<3>::Iterator i = partition.begin();
             i != partition.end();
             ++i)
            actual2 << *i;

        SuperVector<Coord<3> > expected;
        expected << Coord<3>(1, 2, 3)
                 << Coord<3>(2, 2, 3)
                 << Coord<3>(1, 3, 3)
                 << Coord<3>(2, 3, 3)
                 << Coord<3>(1, 2, 4)
                 << Coord<3>(2, 2, 4)
                 << Coord<3>(1, 3, 4)
                 << Coord<3>(2, 3, 4);

        TS_ASSERT_EQUALS(actual1, expected);
        TS_ASSERT_EQUALS(actual2, expected);
    }

    void largeTest(Coord<3> dimensions)
    {
        Coord<3> offset(10, 20, 30);
        ZCurvePartition<3> partition(offset, dimensions);

        SuperVector<Coord<3> > actual1;
        SuperVector<Coord<3> > actual2;
        SuperVector<Coord<3> > expected;

        CoordBox<3> box(offset, dimensions);
        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            expected << *i;
        }
        
        for (int i = 0; i < dimensions.prod(); ++i)
            actual1 << *partition[i];

        for (ZCurvePartition<3>::Iterator i = partition.begin();
             i != partition.end();
             ++i)
            actual2 << *i;

        actual1.sort();
        actual2.sort();
        expected.sort();

        TS_ASSERT_EQUALS(actual1, expected);
        TS_ASSERT_EQUALS(actual2, expected);
    }

    void test3dLarge2()
    {
        largeTest(Coord<3>(5, 7, 20));
        largeTest(Coord<3>(5, 40, 4));
        largeTest(Coord<3>(50, 8, 8));
    }


private:
    ZCurvePartition<2> partition;
    CoordVector expected, actual;
};

}
}
