#include <libgeodecomp/geometry/partitions/stripingpartition.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/misc/chronometer.h>

#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>

using namespace boost::assign;
using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class RegionTest : public CxxTest::TestSuite
{
public:
    typedef std::vector<Coord<2> > CoordVector;
    typedef Region<1>::IntPair IntPair;
    typedef Region<1>::IndexVectorType IndexVectorType;

    void setUp()
    {
        c = Region<2>();

        std::vector<std::string> s;
        s +=
            "               ",
            "               ",
            "               ",
            "               ",
            "               ",
            "               ",
            "          X    ",
            "          X    ",
            "   X X X XXXXXX",
            " XXXXXXXXXXXXX ",
            "  XXXXXXXXXX   ",
            "  XXXXXXXXXX   ",
            "  XXXXXXXXXX   ",
            "  XXXXXXXXXX   ",
            "  XXXXXXXXXX   ",
            "  XXXXXXXXXX   ",
            "  XXXXXXXXXX   ",
            "  XXXXXXXXXX   ",
            "  XXX    XXX   ",
            "  XXX    XXX   ",
            "  XXX    XXX   ",
            "  XXX    XXX   ",
            "  XXXXXXXXXX   ",
            "  XXX    XXX   ",
            "  XXX    XXX   ",
            "  XXX    XXX   ",
            "  XXX    XXX   ";
        bigInsertOrdered = transform(s);
        bigInsertShuffled = bigInsertOrdered;
        std::random_shuffle(bigInsertShuffled.begin(), bigInsertShuffled.end());
    }

    void testIntersectOrTouch()
    {
        RegionHelpers::RegionInsertHelper<0> h;
        TS_ASSERT_EQUALS(
            true, h.intersectOrTouch(IntPair(10, 30),
                                     IntPair(30, 40)));
        TS_ASSERT_EQUALS(
            true, h.intersectOrTouch(IntPair(30, 40),
                                     IntPair(10, 30)));
        TS_ASSERT_EQUALS(
            true, h.intersectOrTouch(IntPair(10, 40),
                                     IntPair(20, 30)));
        TS_ASSERT_EQUALS(
            true, h.intersectOrTouch(IntPair(20, 30),
                                     IntPair(10, 40)));
        TS_ASSERT_EQUALS(
            true, h.intersectOrTouch(IntPair(10, 30),
                                     IntPair(20, 40)));
        TS_ASSERT_EQUALS(
            true, h.intersectOrTouch(IntPair(20, 40),
                                     IntPair(10, 30)));
        TS_ASSERT_EQUALS(
            true, h.intersectOrTouch(IntPair(20, 40),
                                     IntPair(30, 30)));
        TS_ASSERT_EQUALS(
            true, h.intersectOrTouch(IntPair(30, 30),
                                     IntPair(20, 40)));

        TS_ASSERT_EQUALS(
            false, h.intersectOrTouch(IntPair(20, 30),
                                      IntPair(40, 50)));
        TS_ASSERT_EQUALS(
            false, h.intersectOrTouch(IntPair(40, 50),
                                      IntPair(20, 30)));
    }

    void testIntersect()
    {
        RegionHelpers::RegionRemoveHelper<0> h;

        TS_ASSERT_EQUALS(
            false, h.intersect(IntPair(10, 30),
                               IntPair(30, 40)));
        TS_ASSERT_EQUALS(
            false, h.intersect(IntPair(30, 40),
                               IntPair(10, 30)));

        TS_ASSERT_EQUALS(
            true, h.intersect(IntPair(10, 40),
                              IntPair(20, 30)));
        TS_ASSERT_EQUALS(
            true, h.intersect(IntPair(20, 30),
                              IntPair(10, 40)));
        TS_ASSERT_EQUALS(
            true, h.intersect(IntPair(10, 30),
                              IntPair(20, 40)));
        TS_ASSERT_EQUALS(
            true, h.intersect(IntPair(20, 40),
                              IntPair(10, 30)));
        TS_ASSERT_EQUALS(
            true, h.intersect(IntPair(20, 40),
                              IntPair(30, 30)));
        TS_ASSERT_EQUALS(
            true, h.intersect(IntPair(30, 30),
                              IntPair(20, 40)));

        TS_ASSERT_EQUALS(
            false, h.intersect(IntPair(20, 30),
                               IntPair(40, 50)));
        TS_ASSERT_EQUALS(
            false, h.intersect(IntPair(40, 50),
                               IntPair(20, 30)));
    }

    void testFuse()
    {
        RegionHelpers::RegionInsertHelper<0> h;

        TS_ASSERT_EQUALS(
            IntPair(10, 30),
            h.fuse(IntPair(10, 15),
                   IntPair(15, 30)));
        TS_ASSERT_EQUALS(
            IntPair(10, 30),
            h.fuse(IntPair(15, 15),
                   IntPair(10, 30)));
        TS_ASSERT_EQUALS(
            IntPair(10, 30),
            h.fuse(IntPair(20, 30),
                   IntPair(10, 25)));
    }

    void testSubstract()
    {
        RegionHelpers::RegionRemoveHelper<0> h;
        Region<2>::IndexVectorType expected;

        TS_ASSERT_EQUALS(
            expected, h.substract(IntPair(40, 50),
                           IntPair(20, 60)));
        TS_ASSERT_EQUALS(
            expected, h.substract(IntPair(40, 50),
                           IntPair(40, 50)));

        expected += IntPair(40, 42);
        TS_ASSERT_EQUALS(
            expected, h.substract(IntPair(40, 50),
                           IntPair(42, 60)));

        expected += IntPair(49, 50);
        TS_ASSERT_EQUALS(
            expected, h.substract(IntPair(40, 50),
                                  IntPair(42, 49)));

        expected.erase(expected.begin());
        TS_ASSERT_EQUALS(
            expected, h.substract(IntPair(40, 50),
                                  IntPair(30, 49)));
    }

    void testInsert1a()
    {
        c << Coord<2>(10, 10)
          << Coord<2>(12, 10)
          << Coord<2>(11, 10)
          << Coord<2>(14, 10);

        TS_ASSERT_EQUALS(unsigned(1), c.indices[1].size());
        TS_ASSERT_EQUALS(unsigned(2), c.indices[0].size());
    }

    void testInsert1b()
    {
        c << Coord<2>(12, 10)
          << Coord<2>(14, 10)
          << Coord<2>(10, 10);

        TS_ASSERT_EQUALS(unsigned(1), c.indices[1].size());
        TS_ASSERT_EQUALS(unsigned(3), c.indices[0].size());
        TS_ASSERT_EQUALS(10, c.indices[0][0].first);
    }


    void testInsert2()
    {
        CoordVector expected;
        expected +=
            Coord<2>(17, 22),
            Coord<2>(18, 22),
            Coord<2>(20, 11),
            Coord<2>(20, 11),
            Coord<2>(-100, 33),
            Coord<2>(11, 33),
            Coord<2>(12, 33),
            Coord<2>(10, 33),
            Coord<2>(12, 33),
            Coord<2>(20, 33),
            Coord<2>(49, 11),
            Coord<2>(48, 11),
            Coord<2>(47, 11),
            Coord<2>(48, 11),
            Coord<2>(40, 44),
            Coord<2>(43, 44),
            Coord<2>(41, 44),
            Coord<2>(42, 44);

        for (CoordVector::iterator i = expected.begin();
             i != expected.end(); i++) {
            c << *i;
        }

        TS_ASSERT_EQUALS(std::size_t(4), c.indices[1].size());
        TS_ASSERT_EQUALS(std::size_t(7), c.indices[0].size());
    }

    void testInsertCoordBox()
    {
        Region<3> expected, actual;

        for (int x = 0; x < 10; ++x) {
            for (int y = 5; y < 12; ++y) {
                for (int z = 3; z < 17; ++z) {
                    expected << Coord<3>(x, y, z);
                }
            }
        }

        actual << CoordBox<3>(Coord<3>(0, 5, 3), Coord<3>(10, 7, 14));
        TS_ASSERT_EQUALS(expected, actual);
    }

    void testInsert3D()
    {
        Region<3> r;
        TS_ASSERT_EQUALS(r.indices[0].size(), unsigned(0));
        TS_ASSERT_EQUALS(r.indices[1].size(), unsigned(0));
        TS_ASSERT_EQUALS(r.indices[2].size(), unsigned(0));

        r << Streak<3>(Coord<3>(10, 20, 30), 40);
        TS_ASSERT_EQUALS(r.indices[0].size(), unsigned(1));
        TS_ASSERT_EQUALS(r.indices[1].size(), unsigned(1));
        TS_ASSERT_EQUALS(r.indices[2].size(), unsigned(1));

        r << Streak<3>(Coord<3>(12, 22, 31), 42);
        TS_ASSERT_EQUALS(r.indices[0].size(), unsigned(2));
        TS_ASSERT_EQUALS(r.indices[1].size(), unsigned(2));
        TS_ASSERT_EQUALS(r.indices[2].size(), unsigned(2));

        r << Streak<3>(Coord<3>(14, 24, 29), 44);
        TS_ASSERT_EQUALS(r.indices[0].size(), unsigned(3));
        TS_ASSERT_EQUALS(r.indices[1].size(), unsigned(3));
        TS_ASSERT_EQUALS(r.indices[2].size(), unsigned(3));

        r << Streak<3>(Coord<3>(16, 21, 30), 46);
        TS_ASSERT_EQUALS(r.indices[0].size(), unsigned(4));
        TS_ASSERT_EQUALS(r.indices[1].size(), unsigned(4));
        TS_ASSERT_EQUALS(r.indices[2].size(), unsigned(3));

        r << Streak<3>(Coord<3>(58, 20, 30), 68);
        TS_ASSERT_EQUALS(r.indices[0].size(), unsigned(5));
        TS_ASSERT_EQUALS(r.indices[1].size(), unsigned(4));
        TS_ASSERT_EQUALS(r.indices[2].size(), unsigned(3));

        r << Streak<3>(Coord<3>(59, 19, 29), 69);
        TS_ASSERT_EQUALS(r.indices[0].size(), unsigned(6));
        TS_ASSERT_EQUALS(r.indices[1].size(), unsigned(5));
        TS_ASSERT_EQUALS(r.indices[2].size(), unsigned(3));

        r << Streak<3>(Coord<3>(38, 20, 30), 60);
        TS_ASSERT_EQUALS(r.indices[0].size(), unsigned(5));
        TS_ASSERT_EQUALS(r.indices[1].size(), unsigned(5));
        TS_ASSERT_EQUALS(r.indices[2].size(), unsigned(3));
    }

    void testInsertVsOperator()
    {
        Region<2> a;
        Region<2> b;
        TS_ASSERT_EQUALS(a, b);

        a << Coord<2>(1, 2)
          << Coord<2>(1, 4);
        b.insert(Coord<2>(1, 2));
        b.insert(Coord<2>(1, 4));
        TS_ASSERT_EQUALS(a, b);
    }

    void testCount()
    {
        Region<2> r1;
        TS_ASSERT_EQUALS(0, r1.count(Streak<2>(Coord<2>(0, 0), 1)));

        r1 << Coord<2>(1, 1);
        TS_ASSERT_EQUALS(0, r1.count(Streak<2>(Coord<2>(0, 0), 1)));

        r1 << Coord<2>(-1, -1);
        TS_ASSERT_EQUALS(0, r1.count(Streak<2>(Coord<2>(0, 0), 1)));

        r1 << Coord<2>(-1, 0);
        TS_ASSERT_EQUALS(0, r1.count(Streak<2>(Coord<2>(0, 0), 1)));

        r1 << Coord<2>(1, 0);
        TS_ASSERT_EQUALS(0, r1.count(Streak<2>(Coord<2>(0, 0), 1)));

        r1 << Coord<2>(0, 0);
        TS_ASSERT_EQUALS(1, r1.count(Streak<2>(Coord<2>(0, 0), 1)));

        Region<2> r2;
        r2 << Coord<2>(1, 1);
        TS_ASSERT_EQUALS(1, r2.count(Streak<2>(Coord<2>(1, 1), 2)));

        Region<3> r3;
        TS_ASSERT_EQUALS(0, r3.count(Streak<3>(Coord<3>(1, 2, 3), 2)));
        r3 << Coord<3>(1, 2, 3);
        TS_ASSERT_EQUALS(1, r3.count(Streak<3>(Coord<3>(1, 2, 3), 2)));
    }

    void testCountCoord()
    {
        Region<2> r1;
        TS_ASSERT_EQUALS(0, r1.count(Coord<2>(0, 0)));

        r1 << Coord<2>(1, 1);
        TS_ASSERT_EQUALS(0, r1.count(Coord<2>(0, 0)));

        r1 << Coord<2>(-1, -1);
        TS_ASSERT_EQUALS(0, r1.count(Coord<2>(0, 0)));

        r1 << Coord<2>(-1, 0);
        TS_ASSERT_EQUALS(0, r1.count(Coord<2>(0, 0)));

        r1 << Coord<2>(1, 0);
        TS_ASSERT_EQUALS(0, r1.count(Coord<2>(0, 0)));

        r1 << Coord<2>(0, 0);
        TS_ASSERT_EQUALS(1, r1.count(Coord<2>(0, 0)));

        Region<2> r2;
        r2 << Coord<2>(1, 1);
        TS_ASSERT_EQUALS(1, r2.count(Coord<2>(1, 1)));

        Region<3> r3;
        TS_ASSERT_EQUALS(0, r3.count(Coord<3>(1, 2, 3)));
        r3 << Coord<3>(1, 2, 3);
        TS_ASSERT_EQUALS(1, r3.count(Coord<3>(1, 2, 3)));
    }

    void testCountStreak()
    {
        Region<2> r1;
        TS_ASSERT_EQUALS(0, r1.count(Streak<2>(Coord<2>(3, 1), 5)));

        r1 << Coord<2>(2, 1);
        TS_ASSERT_EQUALS(0, r1.count(Streak<2>(Coord<2>(3, 1), 5)));

        r1 << Coord<2>(3, 1);
        TS_ASSERT_EQUALS(0, r1.count(Streak<2>(Coord<2>(3, 1), 5)));

        r1 << Coord<2>(5, 1);
        TS_ASSERT_EQUALS(0, r1.count(Streak<2>(Coord<2>(3, 1), 5)));

        r1 << Coord<2>(4, 1);
        TS_ASSERT_EQUALS(1, r1.count(Streak<2>(Coord<2>(3, 1), 5)));
        TS_ASSERT_EQUALS(1, r1.count(Streak<2>(Coord<2>(3, 1), 6)));
    }

    void testStreakIteration()
    {
        std::vector<Streak<2> > actual, expected;
        for (Region<2>::StreakIterator i = c.beginStreak();
             i != c.endStreak(); ++i) {
            actual += *i;
        }
        TS_ASSERT_EQUALS(actual, expected);

        c << Streak<2>(Coord<2>(10, 10), 20)
          << Streak<2>(Coord<2>(10, 20), 30)
          << Streak<2>(Coord<2>(25, 10), 40)
          << Streak<2>(Coord<2>(15, 30), 30)
          << Streak<2>(Coord<2>(15, 10), 30)
          << Streak<2>(Coord<2>( 5, 30), 60);

        for (Region<2>::StreakIterator i = c.beginStreak();
             i != c.endStreak(); ++i) {
            actual += *i;
        }

        expected +=
            Streak<2>(Coord<2>(10, 10), 40),
            Streak<2>(Coord<2>(10, 20), 30),
            Streak<2>(Coord<2>( 5, 30), 60);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testEmptyStreakIteration()
    {
        Region<2> region1;
        Region<2> region2;
        Region<2> region;

        region1 << Streak<2>(Coord<2>(0, 0), 10);
        region2 << Streak<2>(Coord<2>(0, 5), 10);
        region = region1 & region2;

        for (Region<2>::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            TS_FAIL("loop should not execute!");
        }
    }

    void testUnorderedInsert()
    {
        c << Coord<2>(7, 8);
        TS_ASSERT_EQUALS(std::size_t(1), c.indices[0].size());
        c << Coord<2>(6, 8);
        TS_ASSERT_EQUALS(std::size_t(1), c.indices[0].size());
        c << Coord<2>(9, 8);
        TS_ASSERT_EQUALS(std::size_t(2), c.indices[0].size());
        c << Coord<2>(4, 8);
        TS_ASSERT_EQUALS(std::size_t(3), c.indices[0].size());
        c << Coord<2>(8, 8);
        TS_ASSERT_EQUALS(std::size_t(2), c.indices[0].size());
        TS_ASSERT_EQUALS(IntPair(4,  5), c.indices[0][0]);
        TS_ASSERT_EQUALS(IntPair(6, 10), c.indices[0][1]);

        c << Coord<2>(3, 8);
        TS_ASSERT_EQUALS(std::size_t(2), c.indices[0].size());
        c << Coord<2>(2, 8);
        TS_ASSERT_EQUALS(std::size_t(2), c.indices[0].size());
        c << Coord<2>(11, 8);
        TS_ASSERT_EQUALS(std::size_t(3), c.indices[0].size());
        c << Coord<2>(5, 8);
        TS_ASSERT_EQUALS(std::size_t(2), c.indices[0].size());
        c << Coord<2>(10, 8);
        TS_ASSERT_EQUALS(std::size_t(1), c.indices[0].size());
        TS_ASSERT_EQUALS(IntPair(2, 12), c.indices[0][0]);
    }

    void testBigInsert()
    {
        CoordVector res;
        for (CoordVector::iterator i = bigInsertShuffled.begin();
             i != bigInsertShuffled.end(); i++) {
            c << *i;
        }

        for (Region<2>::Iterator i = c.begin(); i != c.end(); ++i) {
            res << *i;
        }
        TS_ASSERT_EQUALS(res, bigInsertOrdered);

        TS_ASSERT_EQUALS(0,  c.indices[1][ 0].second);
        TS_ASSERT_EQUALS(1,  c.indices[1][ 1 ].second);
        TS_ASSERT_EQUALS(2,  c.indices[1][ 2].second);
        TS_ASSERT_EQUALS(6,  c.indices[1][ 3].second);

        TS_ASSERT_EQUALS(7,  c.indices[1][ 4].second);
        TS_ASSERT_EQUALS(8,  c.indices[1][ 5].second);
        TS_ASSERT_EQUALS(9,  c.indices[1][ 6].second);
        TS_ASSERT_EQUALS(10, c.indices[1][ 7].second);
        TS_ASSERT_EQUALS(11, c.indices[1][ 8].second);
        TS_ASSERT_EQUALS(12, c.indices[1][ 9].second);
        TS_ASSERT_EQUALS(13, c.indices[1][10].second);
        TS_ASSERT_EQUALS(14, c.indices[1][11].second);
        TS_ASSERT_EQUALS(15, c.indices[1][12].second);

        TS_ASSERT_EQUALS(17, c.indices[1][13].second);
        TS_ASSERT_EQUALS(19, c.indices[1][14].second);
        TS_ASSERT_EQUALS(21, c.indices[1][15].second);
        TS_ASSERT_EQUALS(23, c.indices[1][16].second);

        TS_ASSERT_EQUALS(24, c.indices[1][17].second);

        TS_ASSERT_EQUALS(26, c.indices[1][18].second);
        TS_ASSERT_EQUALS(28, c.indices[1][19].second);
        TS_ASSERT_EQUALS(30, c.indices[1][20].second);
    }

    void testEmpty()
    {
        Region<2> c;
        TS_ASSERT_EQUALS(c.empty(), true);
        c << Coord<2>(1, 2);
        TS_ASSERT_EQUALS(c.empty(), false);
    }

    void testBoundingBox()
    {
        Region<2> c;
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(0, 0),  Coord<2>(0, 0)),
                         c.boundingBox());

        c << Coord<2>(4, 8);
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(4, 8),  Coord<2>(1, 1)),
                         c.boundingBox());

        c << Coord<2>(10, 13);
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(4, 8),  Coord<2>(7, 6)),
                         c.boundingBox());

        c << Streak<2>(Coord<2>(2, 3), 20);
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(2, 3),  Coord<2>(18, 11)),
                         c.boundingBox());

        c << Streak<2>(Coord<2>(5, 5), 30);
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(2, 3),  Coord<2>(28, 11)),
                         c.boundingBox());

        c >> Streak<2>(Coord<2>(6, 5), 29);
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(2, 3),  Coord<2>(28, 11)),
                         c.boundingBox());

        c >> Streak<2>(Coord<2>(5, 5), 30);
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(2, 3),  Coord<2>(18, 11)),
                         c.boundingBox());

        c >> Streak<2>(Coord<2>(500, 500), 1000);
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(2, 3),  Coord<2>(18, 11)),
                         c.boundingBox());

        c << Streak<2>(Coord<2>(-5, 33), 95);
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(-5, 3), Coord<2>(100, 31)),
                         c.boundingBox());

        c << Streak<2>(Coord<2>(-5, 42), 95);
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(-5, 3), Coord<2>(100, 40)),
                         c.boundingBox());

        c >> Streak<2>(Coord<2>(-10, 42),  20);
        c >> Streak<2>(Coord<2>( 50, 42), 100);
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(-5, 3), Coord<2>(100, 40)),
                         c.boundingBox());

        c >> Streak<2>(Coord<2>(-5, 42), 95);
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(-5, 3), Coord<2>(100, 31)),
                         c.boundingBox());
    }

    void testSize()
    {
        Region<2> c;
        TS_ASSERT_EQUALS(std::size_t( 0), c.size());

        c << Streak<2>(Coord<2>(10, 10), 20);
        TS_ASSERT_EQUALS(std::size_t(10), c.size());

        c << Streak<2>(Coord<2>(17, 20), 98);
        TS_ASSERT_EQUALS(std::size_t(91), c.size());

        c >> Streak<2>(Coord<2>(15, 10), 18);
        TS_ASSERT_EQUALS(std::size_t(88), c.size());
    }

    void testDimension()
    {
        Region<3> r;
        TS_ASSERT_EQUALS(Coord<3>(), r.dimension());

        r << Streak<3>(Coord<3>(-100, -90, -80), 200);
        TS_ASSERT_EQUALS(Coord<3>(300, 1, 1), r.dimension());

        r << Coord<3>(10, 20, 30);
        TS_ASSERT_EQUALS(Coord<3>(300, 111, 111), r.dimension());
    }

    void testExpand1()
    {
        //
        // XXXXX
        // X   X
        // X   X
        // X   X
        // XXXXX
        //
        //
        c << Coord<2>(2, 2)
          << Coord<2>(5, 6)
          << Coord<2>(3, 6)
          << Coord<2>(4, 6)
          << Coord<2>(6, 6)
          << Coord<2>(2, 6)
          << Coord<2>(3, 2)
          << Coord<2>(4, 2)
          << Coord<2>(5, 2)
          << Coord<2>(6, 2)
          << Coord<2>(6, 3)
          << Coord<2>(2, 3)
          << Coord<2>(2, 4)
          << Coord<2>(6, 4)
          << Coord<2>(2, 5)
          << Coord<2>(6, 5);
        TS_ASSERT_EQUALS(0, c.indices[1][0].second);
        TS_ASSERT_EQUALS(1, c.indices[1][1].second);
        TS_ASSERT_EQUALS(3, c.indices[1][2].second);
        TS_ASSERT_EQUALS(5, c.indices[1][3].second);
        TS_ASSERT_EQUALS(7, c.indices[1][4].second);

        Region<2> c1 = c.expand();
        TS_ASSERT_EQUALS(0, c1.indices[1][0].second);
        TS_ASSERT_EQUALS(1, c1.indices[1][1].second);
        TS_ASSERT_EQUALS(2, c1.indices[1][2].second);
        TS_ASSERT_EQUALS(3, c1.indices[1][3].second);
        TS_ASSERT_EQUALS(5, c1.indices[1][4].second);
        TS_ASSERT_EQUALS(6, c1.indices[1][5].second);
        TS_ASSERT_EQUALS(7, c1.indices[1][6].second);

        Region<2> c2 = c1.expand();
        for (int i = 0; i < 9; ++i) {
            TS_ASSERT_EQUALS(IntPair(0, 9), c2.indices[0][i]);
        }
    }

    void testExpand2()
    {
        for(int x = 0; x < 10; ++x) {
            for(int y = 0; y < 20; ++y) {
                c << Coord<2>(x, y);
            }
        }

        Region<2> actual = c.expand(20);
        Region<2> expected;

        for(int x = -20; x < 30; ++x) {
            for(int y = -20; y < 40; ++y) {
                expected << Coord<2>(x, y);
            }
        }

        TS_ASSERT_EQUALS(actual, expected);
    }

    void testDelete()
    {
        Region<2>::IndexVectorType expected;
        expected << IntPair(0, 10)  // 0
                                    // 1
                 << IntPair(0, 10)  // 2
                 << IntPair(0,  5)  // 3
                 << IntPair(0, 10)  // 4
                 << IntPair(0,  5)  // 5
                 << IntPair(6, 10)  // 5
                 << IntPair(0, 10)  // 6
                 << IntPair(0, 10)  // 7
                 << IntPair(0, 10)  // 8
                 << IntPair(5, 10); // 9

        for(int x = 0; x < 10; ++x) {
            for(int y = 0; y < 10; ++y) {
                c << Coord<2>(x, y);
            }
        }

        c >> Streak<2>(Coord<2>(-10, 1), 20);
        c >> Streak<2>(Coord<2>(  5, 3), 20);
        c >> Coord<2>(5, 5);
        c >> Streak<2>(Coord<2>(  5, 7),  5);
        c >> Streak<2>(Coord<2>(-20, 9),  5);

        TS_ASSERT_EQUALS(expected, c.indices[0]);
    }

    void testAndNot1()
    {
        Region<2> minuend, expected;
        for(int x = 0; x < 10; ++x) {
            for(int y = 0; y < 10; ++y) {
                c << Coord<2>(x, y);
            }
        }

        for(int x = 5; x < 20; ++x) {
            for(int y = 5; y < 20; ++y) {
                minuend << Coord<2>(x, y);
            }
        }

        for(int x = 0; x < 10; ++x) {
            for(int y = 0; y < 5; ++y) {
                expected << Coord<2>(x, y);
            }
        }

        for(int x = 0; x < 5; ++x) {
            for(int y = 5; y < 10; ++y) {
                expected << Coord<2>(x, y);
            }
        }

        TS_ASSERT_EQUALS(expected, c - minuend);
    }

    void testAndNot2()
    {
        Region<2> expanded, expected;
        for(int x = 0; x < 10; ++x) {
            for(int y = 0; y < 10; ++y) {
                c << Coord<2>(x, y);
            }
        }

        for(int i = -1; i < 11; ++i) {
            expected << Coord<2>( i, -1);
            expected << Coord<2>( i, 10);
            expected << Coord<2>(10,  i);
            expected << Coord<2>(-1,  i);
        }

        expanded = c.expand();
        Region<2> ring = expanded - c;
        TS_ASSERT_EQUALS(expected, ring);
    }

    void testAndAssignmentOperator()
    {
        Region<2> original, mask, expected;
        for(int x = 0; x < 10; ++x) {
            for(int y = 0; y < 10; ++y) {
                original << Coord<2>(x, y);
            }
        }

        for(int x = 3; x < 15; ++x) {
            for(int y = 5; y < 15; ++y) {
                mask << Coord<2>(x, y);
            }
        }

        for(int x = 3; x < 10; ++x) {
            for(int y = 5; y < 10; ++y) {
                expected << Coord<2>(x, y);
            }
        }

        original &= mask;
        TS_ASSERT_EQUALS(original, expected);
    }

    void testAndOperator()
    {
        Region<2> original, mask, expected;
        for(int x = 0; x < 10; ++x) {
            for(int y = 0; y < 10; ++y) {
                original << Coord<2>(x, y);
            }
        }

        for(int x = 3; x < 15; ++x) {
            for(int y = 5; y < 15; ++y) {
                mask << Coord<2>(x, y);
            }
        }

        for(int x = 3; x < 10; ++x) {
            for(int y = 5; y < 10; ++y) {
                expected << Coord<2>(x, y);
            }
        }

        TS_ASSERT_EQUALS(original & mask, expected);
    }

    void testAddAssignmentOperator()
    {
        Region<2> original, addent, expected;
        for(int x = 0; x < 10; ++x) {
            for(int y = 0; y < 10; ++y) {
                original << Coord<2>(x, y);
            }
        }

        for(int x = 3; x < 15; ++x) {
            for(int y = 0; y < 10; ++y) {
                addent << Coord<2>(x, y);
            }
        }

        for(int x = 0; x < 15; ++x) {
            for(int y = 0; y < 10; ++y) {
                expected << Coord<2>(x, y);
            }
        }

        original += addent;
        TS_ASSERT_EQUALS(original, expected);
    }

    void testAddOperator()
    {
        Region<2> original, addent, expected;
        for(int x = 0; x < 10; ++x) {
            for(int y = 0; y < 10; ++y) {
                original << Coord<2>(x, y);
            }
        }

        for(int x = 3; x < 15; ++x) {
            for(int y = 0; y < 10; ++y) {
                addent << Coord<2>(x, y);
            }
        }

        for(int x = 0; x < 15; ++x) {
            for(int y = 0; y < 10; ++y) {
                expected << Coord<2>(x, y);
            }
        }

        TS_ASSERT_EQUALS(original + addent, expected);
    }

    void testEqualsOperator()
    {
        Region<2> a, b;
        TS_ASSERT_EQUALS(a, b);
        a << Coord<2>(10, 10);
        b << Coord<2>(20, 20);
        TS_ASSERT_DIFFERS(a, b);
        b << Coord<2>(10, 10);
        a << Coord<2>(20, 20);
        TS_ASSERT_EQUALS(a, b);
        TS_ASSERT_EQUALS(true, a.geometryCacheTainted);
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(10, 10), Coord<2>(11, 11)),
                         a.boundingBox());
        TS_ASSERT_EQUALS(false, a.geometryCacheTainted);
        TS_ASSERT_EQUALS(true, b.geometryCacheTainted);
        TS_ASSERT_EQUALS(a, b);
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(10, 10), Coord<2>(11, 11)),
                         b.boundingBox());
        TS_ASSERT_EQUALS(false, b.geometryCacheTainted);
        TS_ASSERT_EQUALS(a, b);
    }

    void testNumStreaks()
    {
        Region<2> a;
        TS_ASSERT_EQUALS(unsigned(0), a.numStreaks());
        a << Streak<2>(Coord<2>(10, 10), 20);
        TS_ASSERT_EQUALS(unsigned(1), a.numStreaks());
        a << Streak<2>(Coord<2>(30, 10), 60);
        TS_ASSERT_EQUALS(unsigned(2), a.numStreaks());
        a << Streak<2>(Coord<2>(10, 20), 20);
        TS_ASSERT_EQUALS(unsigned(3), a.numStreaks());
        a << Streak<2>(Coord<2>(15, 10), 25);
        TS_ASSERT_EQUALS(unsigned(3), a.numStreaks());
        a >> Streak<2>(Coord<2>(15, 10), 17);
        TS_ASSERT_EQUALS(unsigned(4), a.numStreaks());
        a << Streak<2>(Coord<2>(12, 10), 30);
        TS_ASSERT_EQUALS(unsigned(2), a.numStreaks());
    }

    void testToVector()
    {
        Region<2> a;
        TS_ASSERT_EQUALS(unsigned(0), a.toVector().size());
        a << Streak<2>(Coord<2>(10, 10), 20);
        TS_ASSERT_EQUALS(unsigned(1), a.toVector().size());
        a << Streak<2>(Coord<2>(30, 10), 60);
        TS_ASSERT_EQUALS(unsigned(2), a.toVector().size());
        a << Streak<2>(Coord<2>(10, 20), 20);
        TS_ASSERT_EQUALS(unsigned(3), a.toVector().size());
        a << Streak<2>(Coord<2>(15, 10), 25);
        TS_ASSERT_EQUALS(unsigned(3), a.toVector().size());
        a >> Streak<2>(Coord<2>(15, 10), 17);
        TS_ASSERT_EQUALS(unsigned(4), a.toVector().size());
        std::vector<Streak<2> > vec = a.toVector();
        TS_ASSERT_EQUALS(a, Region<2>(vec.begin(), vec.end()));
        a << Streak<2>(Coord<2>(12, 10), 30);
        TS_ASSERT_EQUALS(unsigned(2), a.toVector().size());
        vec = a.toVector();
        TS_ASSERT_EQUALS(a, Region<2>(vec.begin(), vec.end()));
    }

    void testIteratorInsertConstructor()
    {
        Coord<2> origin(123, 456);
        Coord<2> dimensions(200, 100);
        StripingPartition<2> s(origin, dimensions);
        Region<2> actual(s.begin(), s.end());
        Region<2> expected;
        for (int y = origin.y(); y != (origin.y() + dimensions.y()); ++y) {
            expected << Streak<2>(Coord<2>(origin.x(), y),
                                  origin.x() + dimensions.x());
        }
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testClear()
    {
        Region<2> a;
        a << Streak<2>(Coord<2>(10, 10), 20);
        TS_ASSERT(!a.empty());

        a.clear();
        TS_ASSERT(a.empty());
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(0, 0), Coord<2>(0, 0)),
                         a.boundingBox());
    }

    void test3DSimple1()
    {
        Region<3> a;
        std::vector<Streak<3> > streaks;
        std::vector<Streak<3> > retrievedStreaks;
        streaks << Streak<3>(Coord<3>(10, 10, 10), 20)
                << Streak<3>(Coord<3>(10, 11, 10), 20)
                << Streak<3>(Coord<3>(10, 12, 10), 20)
                << Streak<3>(Coord<3>(10, 10, 11), 20)
                << Streak<3>(Coord<3>(10, 10, 12), 20);
        a << streaks[0]
          << streaks[1]
          << streaks[2]
          << streaks[3]
          << streaks[4];

        for (Region<3>::StreakIterator i = a.beginStreak(); i != a.endStreak(); ++i) {
            retrievedStreaks << *i;
        }

        TS_ASSERT_EQUALS(retrievedStreaks.size(), streaks.size());
        TS_ASSERT_EQUALS(retrievedStreaks, streaks);
    }

    void test3DSimple2()
    {
        Region<3> a;
        std::vector<Streak<3> > streaks;
        std::vector<Streak<3> > retrievedStreaks;
        streaks << Streak<3>(Coord<3>(10, 10, 10), 20)
                << Streak<3>(Coord<3>(11, 10, 10), 20)
                << Streak<3>(Coord<3>(12, 10, 10), 20);
        a << streaks[0]
          << streaks[1]
          << streaks[2];

        for (Region<3>::StreakIterator i = a.beginStreak(); i != a.endStreak(); ++i) {
            retrievedStreaks << *i;
        }

        TS_ASSERT_EQUALS(retrievedStreaks.size(), std::size_t(1));
        TS_ASSERT_EQUALS(retrievedStreaks[0], Streak<3>(Coord<3>(10, 10, 10), 20));
    }

    void test3DSimple3()
    {
        Region<3> a;
        std::vector<Streak<3> > retrievedStreaks;
        a << Streak<3>(Coord<3>(10, 10, 10), 20);
        a >> Streak<3>(Coord<3>(15, 10, 10), 16);

        for (Region<3>::StreakIterator i = a.beginStreak(); i != a.endStreak(); ++i) {
            retrievedStreaks << *i;
        }

        TS_ASSERT_EQUALS(retrievedStreaks.size(), std::size_t(2));
        TS_ASSERT_EQUALS(retrievedStreaks[0], Streak<3>(Coord<3>(10, 10, 10), 15));
        TS_ASSERT_EQUALS(retrievedStreaks[1], Streak<3>(Coord<3>(16, 10, 10), 20));
    }

    void test3DSimple4()
    {
        Region<3> a;
        std::set<Coord<3> > expected;
        std::set<Coord<3> > actual;

        for (int x = 0; x < 10; ++x) {
            for (int y = 0; y < 10; ++y) {
                for (int z = 0; z < 10; ++z) {
                    Coord<3> c(x, y, z);
                    expected.insert(c);
                    a << c;
                }
            }
        }

        for (Region<3>::Iterator i = a.begin(); i != a.end(); ++i) {
            actual.insert(*i);
        }

        TS_ASSERT_EQUALS(expected, actual);
        TS_ASSERT_EQUALS(a.boundingBox(), CoordBox<3>(Coord<3>(), Coord<3>(10, 10, 10)));
    }

    void testExpand3D()
    {
        Region<3> actual, expected, base;

        for (int z = -3; z < 13; ++z) {
            for (int y = -3; y < 13; ++y) {
                for (int x = -3; x < 28; ++x) {
                    expected << Coord<3>(x, y, z);
                }
            }
        }

        for (int z = 0; z < 10; ++z) {
            for (int y = 0; y < 10; ++y) {
                for (int x = 0; x < 10; ++x) {
                    base << Coord<3>(x, y, z)
                         << Coord<3>(x + 15, y, z);
                }
            }
        }

        actual = base.expand(3);
        TS_ASSERT_EQUALS(expected, actual);
    }

    void testExpandWithTopology1()
    {
        Region<2> region;
        region << Streak<2>(Coord<2>(0, 0), 15)
               << Streak<2>(Coord<2>(0, 1), 20)
               << Streak<2>(Coord<2>(0, 2), 20);

        Region<2> actual = region.expandWithTopology(
            2,
            Coord<2>(20, 20),
            Topologies::Torus<2>::Topology());

        Region<2> expected;
        expected << Streak<2>(Coord<2>(0,   0), 20)
                 << Streak<2>(Coord<2>(0,   1), 20)
                 << Streak<2>(Coord<2>(0,   2), 20)
                 << Streak<2>(Coord<2>(0,   3), 20)
                 << Streak<2>(Coord<2>(0,   4), 20)
                 << Streak<2>(Coord<2>(0,  18), 17)
                 << Streak<2>(Coord<2>(18, 18), 20)
                 << Streak<2>(Coord<2>(0,  19), 20);

        TS_ASSERT_EQUALS(actual, expected);
    }

    void testExpandWithTopology2()
    {
        Region<2> region;
        region << Streak<2>(Coord<2>(0, 0), 15)
               << Streak<2>(Coord<2>(0, 1), 20)
               << Streak<2>(Coord<2>(0, 2), 20);

        Region<2> actual = region.expandWithTopology(
            2,
            Coord<2>(20, 20),
            Topologies::Cube<2>::Topology());

        Region<2> expected;
        expected << Streak<2>(Coord<2>(0,   0), 20)
                 << Streak<2>(Coord<2>(0,   1), 20)
                 << Streak<2>(Coord<2>(0,   2), 20)
                 << Streak<2>(Coord<2>(0,   3), 20)
                 << Streak<2>(Coord<2>(0,   4), 20);

        TS_ASSERT_EQUALS(actual, expected);
    }

    void testExpandWithRadius1D()
    {
        Region<1> r;
        r << Streak<1>(Coord<1>( 10),  30)
          << Streak<1>(Coord<1>( 45),  55)
          << Streak<1>(Coord<1>(100), 130);

        Region<1> actual = r.expand(Coord<1>(10));

        Region<1> expected;
        expected << Streak<1>(Coord<1>( 0),  65)
                 << Streak<1>(Coord<1>(90), 140);

        TS_ASSERT_EQUALS(expected, actual);
    }

    void testExpandWithRadius2D()
    {
        Region<2> r;
        r << CoordBox<2>(Coord<2>(10, 20), Coord<2>(5, 6));

        Region<2> actual = r.expand(Coord<2>(2, 1));

        Region<2> expected;
        expected << CoordBox<2>(Coord<2>(8, 19), Coord<2>(9, 8));

        TS_ASSERT_EQUALS(expected, actual);
    }

    void testExpandWithRadius3D()
    {
        Region<3> r;
        r << CoordBox<3>(Coord<3>(100, 120, 140), Coord<3>(50, 20, 10))
          << CoordBox<3>(Coord<3>(300, 300, 300), Coord<3>(20, 40, 30));

        Region<3> actual = r.expand(Coord<3>(5, 6, 7));

        Region<3> expected;
        expected << CoordBox<3>(Coord<3>( 95, 114, 133), Coord<3>(60, 32, 24))
                 << CoordBox<3>(Coord<3>(295, 294, 293), Coord<3>(30, 52, 44));

        TS_ASSERT_EQUALS(actual, expected);
    }

    void testBoolean3D()
    {
        Region<3> leftCube, rightCube, frontCube, mergerLR, mergerFL;

        for (int z = 0; z < 10; ++z) {
            for (int y = 0; y < 10; ++y) {
                for (int x = 0; x < 10; ++x) {
                    Coord<3> c1(x, y, z);
                    Coord<3> c2(x + 30, y, z);
                    Coord<3> c3(x, y, z - 30);
                    leftCube << c1;
                    rightCube << c2;
                    frontCube << c3;
                    mergerLR << c1
                             << c2;
                    mergerFL << c1
                             << c3;
                }
            }
        }

        TS_ASSERT_EQUALS(leftCube + rightCube, mergerLR);
        TS_ASSERT_EQUALS(mergerLR - leftCube,  rightCube);
        TS_ASSERT_EQUALS(mergerLR - rightCube, leftCube);

        TS_ASSERT_EQUALS(leftCube + frontCube, mergerFL);
        TS_ASSERT_EQUALS(mergerFL - frontCube, leftCube);
        TS_ASSERT_EQUALS(mergerFL - leftCube,  frontCube);
    }

    void testSwap()
    {
        // fixme
    }

    void testRemove3D()
    {
        Region<3> r;
        TS_ASSERT_EQUALS(r.size(), std::size_t(0));
        TS_ASSERT_EQUALS(r.boundingBox(), CoordBox<3>(Coord<3>(), Coord<3>()));

        r << Streak<3>(Coord<3>(10, 20, 30), 40);
        TS_ASSERT_EQUALS(r.indices[0].size(), std::size_t(1));
        TS_ASSERT_EQUALS(r.indices[1].size(), std::size_t(1));
        TS_ASSERT_EQUALS(r.indices[2].size(), std::size_t(1));
        TS_ASSERT_EQUALS(r.size(), std::size_t(30));
        TS_ASSERT_EQUALS(
            r.boundingBox(),
            CoordBox<3>(Coord<3>(10, 20, 30), Coord<3>(30, 1, 1)));

        r >> Streak<3>(Coord<3>(15, 20, 30), 35);
        TS_ASSERT_EQUALS(r.indices[0].size(), std::size_t(2));
        TS_ASSERT_EQUALS(r.indices[1].size(), std::size_t(1));
        TS_ASSERT_EQUALS(r.indices[2].size(), std::size_t(1));

        r >> Streak<3>(Coord<3>(36, 20, 30), 37);
        TS_ASSERT_EQUALS(r.indices[0].size(), std::size_t(3));
        TS_ASSERT_EQUALS(r.indices[1].size(), std::size_t(1));
        TS_ASSERT_EQUALS(r.indices[2].size(), std::size_t(1));

        r >> Streak<3>(Coord<3>(30, 20, 30), 50);
        TS_ASSERT_EQUALS(r.indices[0].size(), std::size_t(1));
        TS_ASSERT_EQUALS(r.indices[1].size(), std::size_t(1));
        TS_ASSERT_EQUALS(r.indices[2].size(), std::size_t(1));

        r << Streak<3>(Coord<3>(40, 21, 29), 60);
        TS_ASSERT_EQUALS(r.indices[0].size(), std::size_t(2));
        TS_ASSERT_EQUALS(r.indices[1].size(), std::size_t(2));
        TS_ASSERT_EQUALS(r.indices[2].size(), std::size_t(2));

        r >> Streak<3>(Coord<3>(50, 21, 29), 55);
        TS_ASSERT_EQUALS(r.indices[0].size(), std::size_t(3));
        TS_ASSERT_EQUALS(r.indices[1].size(), std::size_t(2));
        TS_ASSERT_EQUALS(r.indices[2].size(), std::size_t(2));

        r >> Streak<3>(Coord<3>(35, 21, 29), 60);
        TS_ASSERT_EQUALS(r.indices[0].size(), std::size_t(1));
        TS_ASSERT_EQUALS(r.indices[1].size(), std::size_t(1));
        TS_ASSERT_EQUALS(r.indices[2].size(), std::size_t(1));

        TS_ASSERT_EQUALS(r.indices[1][0].second, 0);
        TS_ASSERT_EQUALS(r.indices[2][0].second, 0);

        r >> Streak<3>(Coord<3>(10, 20, 30), 15);

        TS_ASSERT_EQUALS(r.indices[0].size(), std::size_t(0));
        TS_ASSERT_EQUALS(r.indices[1].size(), std::size_t(0));
        TS_ASSERT_EQUALS(r.indices[2].size(), std::size_t(0));
    }

    void testStreakIterator()
    {
        std::vector<Streak<3> > expected;
        std::vector<Streak<3> > actual;

        Region<3> r;
        TS_ASSERT_EQUALS(r.beginStreak(), r.endStreak());

        Streak<3> newStreak(Coord<3>(10, 10, 10), 20);
        r << newStreak;
        TS_ASSERT_EQUALS(newStreak, *r.beginStreak());
        TS_ASSERT_EQUALS(std::size_t(1), r.numStreaks());
        TS_ASSERT_EQUALS(std::size_t(10), r.size());
        for (Region<3>::StreakIterator i = r.beginStreak(); i != r.endStreak(); ++i) {
            actual << *i;
        }
        expected << newStreak;
        TS_ASSERT_EQUALS(actual, expected);

        newStreak = Streak<3>(Coord<3>(10, 20, 10), 20);
        expected << newStreak;
        r << newStreak;
        TS_ASSERT_EQUALS(std::size_t(2), r.numStreaks());
        TS_ASSERT_EQUALS(std::size_t(20), r.size());
        actual.clear();
        for (Region<3>::StreakIterator i = r.beginStreak(); i != r.endStreak(); ++i) {
            actual << *i;
        }
        TS_ASSERT_EQUALS(actual, expected);

        newStreak = Streak<3>(Coord<3>(30, 20, 10), 40);
        expected << newStreak;
        r << newStreak;
        TS_ASSERT_EQUALS(std::size_t(3), r.numStreaks());
        TS_ASSERT_EQUALS(std::size_t(30), r.size());
        actual.clear();
        for (Region<3>::StreakIterator i = r.beginStreak(); i != r.endStreak(); ++i) {
            actual << *i;
        }
        TS_ASSERT_EQUALS(actual, expected);

        newStreak = Streak<3>(Coord<3>(10, 20, 11), 20);
        expected << newStreak;
        r << newStreak;
        TS_ASSERT_EQUALS(std::size_t(4), r.numStreaks());
        TS_ASSERT_EQUALS(std::size_t(40), r.size());
        actual.clear();
        for (Region<3>::StreakIterator i = r.beginStreak(); i != r.endStreak(); ++i) {
            actual << *i;
        }
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testNormalIterator()
    {
        std::vector<Coord<3> > expected;
        std::vector<Coord<3> > actual;

        Region<3> r;
        r << Streak<3>(Coord<3>(10, 20, 30), 40);
        r << Streak<3>(Coord<3>(50, 60, 70), 80);
        for (Region<3>::Iterator i = r.begin(); i != r.end(); ++i) {
            actual << *i;
        }

        for (int i = 10; i < 40; ++i) {
            expected << Coord<3>(i, 20, 30);
        }
        for (int i = 50; i < 80; ++i) {
            expected << Coord<3>(i, 60, 70);
        }

        TS_ASSERT_EQUALS(actual, expected);
    }

    void testOffsetBasedStreakIteratorAccess()
    {
        Region<3> r;
        for (int z = 0; z < 5; ++z) {
            for (int y = 0; y < 4; ++y) {
                for (int x = 0; x < (2 + y); ++x) {
                    r << Streak<3>(Coord<3>(10 * x, y, z), 10 * x + 4);
                }
            }
        }

        Region<3>::StreakIterator expected = r.beginStreak();
        // cursor will provide us with the initial offsets from which
        // we'll construct the iterators. we need to drag it along
        // with the loops b/c otherwise these is no knowing of how to
        // set the individual offsets.
        Coord<3> cursor(0, 0, 0);

        for (int z = 0; z < 5; ++z) {
            for (int y = 0; y < 4; ++y) {
                for (int x = 0; x < (2 + y); ++x) {
                    Region<3>::StreakIterator actual = r[cursor];
                    TS_ASSERT_EQUALS(*actual, *expected);

                    ++expected;

                    ++actual;
                    TS_ASSERT_EQUALS(*actual, *expected);

                    ++cursor.x();
                }
                ++cursor.y();
            }
            ++cursor.z();
        }
    }

    void testRandomAccessIteratorAccess()
    {
        Region<3> r;
        for (int z = 0; z < 5; ++z) {
            for (int y = 0; y < 4; ++y) {
                for (int x = 0; x < (2 + y); ++x) {
                    r << Streak<3>(Coord<3>(10 * x, y, z), 10 * x + 4);
                }
            }
        }

        Region<3>::StreakIterator expected = r.beginStreak();

        for (unsigned i = 0; i < r.numStreaks(); ++i) {
            Region<3>::StreakIterator actual = r[i];
            TS_ASSERT_EQUALS(*actual, *expected);

            ++expected;

            ++actual;
            TS_ASSERT_EQUALS(*actual, *expected);
        }
    }

    void testToString()
    {
        std::vector<Streak<2> > streaks;
        streaks << Streak<2>(Coord<2>( 1,  2), 10)
                << Streak<2>(Coord<2>( 2,  2), 15)
                << Streak<2>(Coord<2>( 3,  8), 20)
                << Streak<2>(Coord<2>(40,  8), 50)
                << Streak<2>(Coord<2>(80,  9), 90);

        std::ostringstream buf;
        buf << "Region<2>(\n"
            << "  indices[0] = [(1, 15), (3, 20), (40, 50), (80, 90)]\n"
            << "  indices[1] = [(2, 0), (8, 1), (9, 3)]\n"
            << ")\n";

        Region<2> region;
        region << streaks[0]
               << streaks[1]
               << streaks[2]
               << streaks[3]
               << streaks[4];

        std::string expected = buf.str();
        std::string actual = region.toString();

        TS_ASSERT_EQUALS(expected, actual);
    }

    void testPrettyPrint()
    {
        std::vector<Streak<2> > streaks;
        streaks << Streak<2>(Coord<2>(1,  2), 10)
                << Streak<2>(Coord<2>(2,  4), 15)
                << Streak<2>(Coord<2>(3,  8), 20)
                << Streak<2>(Coord<2>(4, 16), 25);

        std::ostringstream buf;
        buf << "Region<2>(\n"
            << "  " << streaks[0] << "\n"
            << "  " << streaks[1] << "\n"
            << "  " << streaks[2] << "\n"
            << "  " << streaks[3] << "\n"
            << ")\n";

        Region<2> region;
        region << streaks[0]
               << streaks[1]
               << streaks[2]
               << streaks[3];

        std::string expected = buf.str();
        std::string actual = region.prettyPrint();

        TS_ASSERT_EQUALS(expected, actual);
    }

    void testPrintToBOV()
    {
        // fixme
    }

private:
    Region<2> c;
    CoordVector bigInsertOrdered;
    CoordVector bigInsertShuffled;

    CoordVector transform(const std::vector<std::string>& shape)
    {
        CoordVector ret;
        for (int y = 0; y < (int)shape.size(); y++) {
            std::string line = shape[y];
            for (int x = 0; x < (int)line.size(); x++) {
                if (line[x] != ' ')
                    ret.push_back(Coord<2>(x, y));
            }
        }
        return ret;
    }

    int bongo(const Coord<2>& c) const
    {
        return c.x() % 13 + c.y() % 17;
    }

    Coord<2> benchDim() const
    {
        return Coord<2>(1000, 50);
    }
};

}
