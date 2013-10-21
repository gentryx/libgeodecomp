#include <boost/assign/std/vector.hpp>
#include <boost/shared_ptr.hpp>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/geometry/partitions/hindexingpartition.h>
#include <libgeodecomp/geometry/partitions/stripingpartition.h>

using namespace boost::assign;
using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class HIndexingPartitionTest : public CxxTest::TestSuite
{
public:
    typedef std::vector<Coord<2> > CoordVector;

    void testFillRectangles()
    {
        std::map<Coord<2>, CoordVector> expected;
        expected[Coord<2>(4, 1)] += Coord<2>(0, 0), Coord<2>(1, 0), Coord<2>(2, 0), Coord<2>(3, 0);
        expected[Coord<2>(1, 5)] += Coord<2>(0, 0), Coord<2>(0, 1), Coord<2>(0, 2), Coord<2>(0, 3), Coord<2>(0, 4);
        expected[Coord<2>(2, 2)] += Coord<2>(0, 0), Coord<2>(1, 0), Coord<2>(1, 1), Coord<2>(0, 1);
        expected[Coord<2>(3, 2)] += Coord<2>(0, 0), Coord<2>(1, 0), Coord<2>(2, 0), Coord<2>(1, 1), Coord<2>(2, 1), Coord<2>(0, 1);
        expected[Coord<2>(2, 3)] += Coord<2>(0, 0), Coord<2>(1, 0), Coord<2>(1, 1), Coord<2>(1, 2), Coord<2>(0, 2), Coord<2>(0, 1);
        expected[Coord<2>(3, 3)] +=
            // 1st half
            Coord<2>(0, 0),
            Coord<2>(1, 0), Coord<2>(2, 0),
            Coord<2>(1, 1), Coord<2>(2, 1), Coord<2>(2, 2),
            // 2nd half
            Coord<2>(1, 2),
            Coord<2>(0, 2), Coord<2>(0, 1);
        expected[Coord<2>(4, 4)] +=
            // 1st half
            Coord<2>(0, 0), Coord<2>(1, 0), Coord<2>(1, 1),
            Coord<2>(2, 1), Coord<2>(2, 0), Coord<2>(3, 0),
            Coord<2>(3, 1),
            Coord<2>(2, 2), Coord<2>(3, 2), Coord<2>(3, 3),
            // 2nd half
            Coord<2>(2, 3),
            Coord<2>(1, 3),
            Coord<2>(0, 3), Coord<2>(0, 2), Coord<2>(1, 2),
            Coord<2>(0, 1);

        for (std::map<Coord<2>, CoordVector>::iterator i = expected.begin();
             i != expected.end();
             i++) {
            Coord<2> origin(47, 11);
            Coord<2> dimension(i->first);
            CoordVector coords = i->second;

            unsigned c = 0;
            HIndexingPartition h(origin, dimension);
            for (HIndexingPartition::Iterator j = h.begin(); j != h.end(); ++j)
                TS_ASSERT_EQUALS(origin + coords[c++], *j);

            TS_ASSERT_EQUALS(coords.size(), c);
        }
    }

    void testBeginEnd()
    {
        HIndexingPartition h(Coord<2>(10, 20), Coord<2>(3, 5));
        CoordVector expected, actual;
        // expected traversal order:
        // 034
        // 125
        // e67
        // db8
        // ca9
        expected +=
            Coord<2>(10, 20),
            Coord<2>(10, 21),
            Coord<2>(11, 21),
            Coord<2>(11, 20),
            Coord<2>(12, 20),
            Coord<2>(12, 21),
            Coord<2>(11, 22),
            Coord<2>(12, 22),
            Coord<2>(12, 23),
            Coord<2>(12, 24),
            Coord<2>(11, 24),
            Coord<2>(11, 23),
            Coord<2>(10, 24),
            Coord<2>(10, 23),
            Coord<2>(10, 22);
        for (HIndexingPartition::Iterator i = h.begin(); i != h.end(); ++i)
            actual += *i;
        TS_ASSERT_EQUALS(expected, actual);
    }

    void testTriangleLengthTrivial()
    {
        TS_ASSERT_EQUALS(unsigned(1), HIndexingPartition::Iterator::triangleLength(Coord<2>(1, 1), 0));
        TS_ASSERT_EQUALS(unsigned(0), HIndexingPartition::Iterator::triangleLength(Coord<2>(1, 1), 1));
        TS_ASSERT_EQUALS(unsigned(1), HIndexingPartition::Iterator::triangleLength(Coord<2>(1, 1), 2));
        TS_ASSERT_EQUALS(unsigned(0), HIndexingPartition::Iterator::triangleLength(Coord<2>(1, 1), 3));

        TS_ASSERT_EQUALS(unsigned(8), HIndexingPartition::Iterator::triangleLength(Coord<2>(1, 8), 0));
        TS_ASSERT_EQUALS(unsigned(0), HIndexingPartition::Iterator::triangleLength(Coord<2>(1, 8), 1));
        TS_ASSERT_EQUALS(unsigned(8), HIndexingPartition::Iterator::triangleLength(Coord<2>(1, 8), 2));
        TS_ASSERT_EQUALS(unsigned(0), HIndexingPartition::Iterator::triangleLength(Coord<2>(1, 8), 3));

        TS_ASSERT_EQUALS(unsigned(9), HIndexingPartition::Iterator::triangleLength(Coord<2>(9, 1), 0));
        TS_ASSERT_EQUALS(unsigned(0), HIndexingPartition::Iterator::triangleLength(Coord<2>(9, 1), 1));
        TS_ASSERT_EQUALS(unsigned(9), HIndexingPartition::Iterator::triangleLength(Coord<2>(9, 1), 2));
        TS_ASSERT_EQUALS(unsigned(0), HIndexingPartition::Iterator::triangleLength(Coord<2>(9, 1), 3));
    }

    void testTriangleLengthRecursive()
    {
        TS_ASSERT_EQUALS(unsigned(7), HIndexingPartition::Iterator::triangleLength(Coord<2>(4, 3), 0));
        TS_ASSERT_EQUALS(unsigned(5), HIndexingPartition::Iterator::triangleLength(Coord<2>(4, 3), 1));
        TS_ASSERT_EQUALS(unsigned(7), HIndexingPartition::Iterator::triangleLength(Coord<2>(4, 3), 2));
        TS_ASSERT_EQUALS(unsigned(5), HIndexingPartition::Iterator::triangleLength(Coord<2>(4, 3), 3));

        // repetition to check if cache returns correct results
        TS_ASSERT_EQUALS(unsigned(7), HIndexingPartition::Iterator::triangleLength(Coord<2>(4, 3), 0));

        TS_ASSERT_EQUALS(unsigned(123 * 456),
                         HIndexingPartition::Iterator::triangleLength(Coord<2>(123, 456), 0) +
                         HIndexingPartition::Iterator::triangleLength(Coord<2>(123, 456), 3));
    }

    void testSquareBracketsOperatorSimple()
    {
        Coord<2> dimensions(3, 5);
        for (int prolog = 0; prolog < (dimensions.x() * dimensions.y()); ++prolog) {
            HIndexingPartition h(Coord<2>(10, 20), dimensions);
            HIndexingPartition::Iterator testIter = h[prolog];
            HIndexingPartition::Iterator normalIter = h.begin();
            for (int i = 0; i < prolog; ++i)
                ++normalIter;
            while (normalIter != h.end()) {
                TS_ASSERT_EQUALS(*normalIter, *testIter);
                TS_ASSERT_EQUALS(normalIter, testIter);
                ++testIter;
                ++normalIter;
            }
            TS_ASSERT_EQUALS(h.end(), testIter);
        }
    }

    void testSquareBracketsOperatorLarge()
    {
        Coord<2> dimensions(123, 234);
        // don't check ALL possible skip values, but just some, with special emphasis on the lower ones
        for (int prolog = 0; prolog <= (dimensions.x() * dimensions.y()); prolog = prolog * 3 + 1) {
            HIndexingPartition h(Coord<2>(10, 20), dimensions);
            HIndexingPartition::Iterator testIter = h[prolog];
            HIndexingPartition::Iterator normalIter = h.begin();
            for (int i = 0; i < prolog; ++i)
                ++normalIter;
            while (normalIter != h.end()) {
                TS_ASSERT_EQUALS(*normalIter, *testIter);
                TS_ASSERT_EQUALS(normalIter, testIter);
                ++testIter;
                ++normalIter;
            }
            TS_ASSERT_EQUALS(h.end(), testIter);
        }
    }

    void testSquareBracketsOperatorForPartialIteration()
    {
        CoordVector expected, actual, buffer;
        unsigned start = 11;
        unsigned end = 47;

        HIndexingPartition h(Coord<2>(10, 20), Coord<2>(30, 20));
        for (HIndexingPartition::Iterator i = h.begin(); i != h.end(); ++i)
            buffer += *i;
        for (unsigned i = start; i < end; ++i)
            expected += buffer[i];

        for (HIndexingPartition::Iterator i = h[start]; i != h[end]; ++i)
            actual += *i;

        TS_ASSERT_EQUALS(expected, actual);
    }
};

}
