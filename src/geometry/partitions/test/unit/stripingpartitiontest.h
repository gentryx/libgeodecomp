#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/partitions/stripingpartition.h>

#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class StripingPartitionTest : public CxxTest::TestSuite
{
public:
    typedef std::vector<Coord<2> > CoordVector;

    void setUp()
    {
        expected.clear();
    }

    void check(const StripingPartition<2>& p, const CoordVector& expected)
    {
        CoordVector actual;
        StripingPartition<2>::Iterator end = p.end();

        for (StripingPartition<2>::Iterator i = p.begin(); i != end; ++i) {
            actual << *i;
        }

        TS_ASSERT_EQUALS(actual, expected);
    }

    void testSimple()
    {
        expected << Coord<2>(0, 0);
        check(StripingPartition<2>(Coord<2>(0, 0), Coord<2>(1, 1)), expected);
    }

    void testVertical()
    {
        expected << Coord<2>(0, 0) << Coord<2>(0, 1) << Coord<2>(0, 2) << Coord<2>(0, 3);
        check(StripingPartition<2>(Coord<2>(0, 0), Coord<2>(1, 4)), expected);
    }

    void testHorizontal()
    {
        expected << Coord<2>(0, 0) << Coord<2>(1, 0) << Coord<2>(2, 0) << Coord<2>(3, 0) << Coord<2>(4, 0);
        check(StripingPartition<2>(Coord<2>(0, 0), Coord<2>(5, 1)), expected);
    }

    void testNormal()
    {
        expected << Coord<2>(0, 0) << Coord<2>(1, 0) << Coord<2>(2, 0)
                 << Coord<2>(0, 1) << Coord<2>(1, 1) << Coord<2>(2, 1)
                 << Coord<2>(0, 2) << Coord<2>(1, 2) << Coord<2>(2, 2)
                 << Coord<2>(0, 3) << Coord<2>(1, 3) << Coord<2>(2, 3);
        check(StripingPartition<2>(Coord<2>(0, 0), Coord<2>(3, 4)), expected);
    }

    void testOffset()
    {
        expected << Coord<2>(12, 34)
                 << Coord<2>(13, 34)
                 << Coord<2>(14, 34)
                 << Coord<2>(15, 34)
                 << Coord<2>(16, 34);
        check(StripingPartition<2>(Coord<2>(12, 34), Coord<2>(5, 1)), expected);
    }

    void testSquareBracketsOperator()
    {
        StripingPartition<2> s(Coord<2>(), Coord<2>(3, 3));
        std::vector<StripingPartition<2>::Iterator> expected, actual;
        expected << s[0] << s[1] << s[2] << s[3];
        StripingPartition<2>::Iterator i = s.begin();
        for (int c = 0; c < 4; ++c) {
            actual.push_back(i);
            ++i;
        }

        TS_ASSERT_EQUALS(expected, actual);
    }

    void test3D()
    {
        std::vector<Coord<3> > expected;
        std::vector<Coord<3> > actual;

        Coord<3> offset(4, 3, 5);
        Coord<3> dim(4, 3, 5);
        CoordBox<3> box(offset, dim);

        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            expected << *i;
        }

        StripingPartition<3> p(offset, dim);

        for (StripingPartition<3>::Iterator i = p.begin(); i != p.end(); ++i) {
            actual << *i;
        }

        TS_ASSERT_EQUALS(expected, actual);
    }

private:
    CoordVector  expected;
};

}
