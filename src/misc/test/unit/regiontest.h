#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/parallelization/chronometer.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>

using namespace boost::assign;
using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {

class RegionTest : public CxxTest::TestSuite 
{
public:
    typedef Region<2>::Line Line;
    typedef Region<2>::StreakMap StreakMap;

    void setUp()
    {
        c = Region<2>();

        SuperVector<std::string> s;
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
        TS_ASSERT_EQUALS(
            true, c.intersectOrTouch(Streak<2>(Coord<2>(10, 20), 30), 
                                     Streak<2>(Coord<2>(30, 20), 40)));
        TS_ASSERT_EQUALS(
            true, c.intersectOrTouch(Streak<2>(Coord<2>(30, 20), 40), 
                                     Streak<2>(Coord<2>(10, 20), 30)));
        TS_ASSERT_EQUALS(
            true, c.intersectOrTouch(Streak<2>(Coord<2>(10, 20), 40), 
                                     Streak<2>(Coord<2>(20, 20), 30)));
        TS_ASSERT_EQUALS(
            true, c.intersectOrTouch(Streak<2>(Coord<2>(20, 20), 30), 
                                     Streak<2>(Coord<2>(10, 20), 40)));
        TS_ASSERT_EQUALS(
            true, c.intersectOrTouch(Streak<2>(Coord<2>(10, 20), 30), 
                                     Streak<2>(Coord<2>(20, 20), 40)));
        TS_ASSERT_EQUALS(
            true, c.intersectOrTouch(Streak<2>(Coord<2>(20, 20), 40), 
                                     Streak<2>(Coord<2>(10, 20), 30)));
        TS_ASSERT_EQUALS(
            true, c.intersectOrTouch(Streak<2>(Coord<2>(20, 20), 40), 
                                     Streak<2>(Coord<2>(30, 20), 30)));
        TS_ASSERT_EQUALS(
            true, c.intersectOrTouch(Streak<2>(Coord<2>(30, 20), 30), 
                                     Streak<2>(Coord<2>(20, 20), 40)));

        TS_ASSERT_EQUALS(
            false, c.intersectOrTouch(Streak<2>(Coord<2>(20, 20), 30), 
                                      Streak<2>(Coord<2>(40, 20), 50)));
        TS_ASSERT_EQUALS(
            false, c.intersectOrTouch(Streak<2>(Coord<2>(40, 20), 50), 
                                      Streak<2>(Coord<2>(20, 20), 30)));
    }

    void testIntersect()
    {
        TS_ASSERT_EQUALS(
            false, c.intersect(Streak<2>(Coord<2>(10, 20), 30), 
                               Streak<2>(Coord<2>(30, 20), 40)));
        TS_ASSERT_EQUALS(
            false, c.intersect(Streak<2>(Coord<2>(30, 20), 40), 
                               Streak<2>(Coord<2>(10, 20), 30)));

        TS_ASSERT_EQUALS(
            true, c.intersect(Streak<2>(Coord<2>(10, 20), 40), 
                              Streak<2>(Coord<2>(20, 20), 30)));
        TS_ASSERT_EQUALS(
            true, c.intersect(Streak<2>(Coord<2>(20, 20), 30), 
                              Streak<2>(Coord<2>(10, 20), 40)));
        TS_ASSERT_EQUALS(
            true, c.intersect(Streak<2>(Coord<2>(10, 20), 30), 
                              Streak<2>(Coord<2>(20, 20), 40)));
        TS_ASSERT_EQUALS(
            true, c.intersect(Streak<2>(Coord<2>(20, 20), 40), 
                              Streak<2>(Coord<2>(10, 20), 30)));
        TS_ASSERT_EQUALS(
            true, c.intersect(Streak<2>(Coord<2>(20, 20), 40), 
                              Streak<2>(Coord<2>(30, 20), 30)));
        TS_ASSERT_EQUALS(
            true, c.intersect(Streak<2>(Coord<2>(30, 20), 30), 
                              Streak<2>(Coord<2>(20, 20), 40)));

        TS_ASSERT_EQUALS(
            false, c.intersect(Streak<2>(Coord<2>(20, 20), 30), 
                               Streak<2>(Coord<2>(40, 20), 50)));
        TS_ASSERT_EQUALS(
            false, c.intersect(Streak<2>(Coord<2>(40, 20), 50), 
                               Streak<2>(Coord<2>(20, 20), 30)));
    }

    void testFuse()
    {
        TS_ASSERT_EQUALS(
            Streak<2>(Coord<2>(10, 20), 30), 
            c.fuse(Streak<2>(Coord<2>(10, 20), 15), 
                   Streak<2>(Coord<2>(15, 20), 30)));
        TS_ASSERT_EQUALS(
            Streak<2>(Coord<2>(10, 20), 30), 
            c.fuse(Streak<2>(Coord<2>(15, 20), 15), 
                   Streak<2>(Coord<2>(10, 20), 30)));
        TS_ASSERT_EQUALS(
            Streak<2>(Coord<2>(10, 20), 30), 
            c.fuse(Streak<2>(Coord<2>(20, 20), 30), 
                   Streak<2>(Coord<2>(10, 20), 25)));
    }

    void testSubstract()
    {
        SuperVector<Streak<2> > expected;
        TS_ASSERT_EQUALS(
            expected, c.substract(Streak<2>(Coord<2>(40, 20), 50), 
                                  Streak<2>(Coord<2>(20, 20), 60)));
        TS_ASSERT_EQUALS(
            expected, c.substract(Streak<2>(Coord<2>(40, 20), 50), 
                                  Streak<2>(Coord<2>(40, 20), 50)));
        
        expected += Streak<2>(Coord<2>(40, 20), 42);
        TS_ASSERT_EQUALS(
            expected, c.substract(Streak<2>(Coord<2>(40, 20), 50), 
                                  Streak<2>(Coord<2>(42, 20), 60)));

        expected += Streak<2>(Coord<2>(49, 20), 50);
        TS_ASSERT_EQUALS(
            expected, c.substract(Streak<2>(Coord<2>(40, 20), 50), 
                                  Streak<2>(Coord<2>(42, 20), 49)));

        expected.erase(expected.begin());
        TS_ASSERT_EQUALS(
            expected, c.substract(Streak<2>(Coord<2>(40, 20), 50), 
                                  Streak<2>(Coord<2>(30, 20), 49)));
    }

    void testInsert1()
    {
        c << Coord<2>(10, 10)
          << Coord<2>(12, 10)
          << Coord<2>(11, 10)
          << Coord<2>(14, 10);

        TS_ASSERT_EQUALS(1, c.streaks.size());
        TS_ASSERT_EQUALS(2, c.streaks[10].size());
    }

    void testInsert2()
    {
        SuperVector<Coord<2> > expected;
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
            
        for (SuperVector<Coord<2> >::iterator i = expected.begin(); 
             i != expected.end(); i++) 
            c << *i;

        TS_ASSERT_EQUALS((unsigned)4, c.streaks.size());
        TS_ASSERT_EQUALS((unsigned)1, c.streaks[22].size());
        TS_ASSERT_EQUALS((unsigned)3, c.streaks[33].size());
        TS_ASSERT_EQUALS((unsigned)2, c.streaks[11].size());
        TS_ASSERT_EQUALS((unsigned)1, c.streaks[44].size());
    }

    void testInsertCoordBox()
    {
        Region<3> expected, actual;
        
        for (int x = 0; x < 10; ++x) 
            for (int y = 5; y < 12; ++y) 
                for (int z = 3; z < 17; ++z) 
                    expected << Coord<3>(x, y, z);
        actual << CoordBox<3>(Coord<3>(0, 5, 3), Coord<3>(10, 7, 14));
        TS_ASSERT_EQUALS(expected, actual);
    }

    void testStreakIteration()
    {
        SuperVector<Streak<2> > actual, expected;
        for (StreakIterator<2> i = c.beginStreak(); 
             i != c.endStreak(); ++i)
            actual += *i;
        TS_ASSERT_EQUALS(actual, expected);

        c << Streak<2>(Coord<2>(10, 10), 20) 
          << Streak<2>(Coord<2>(10, 20), 30)
          << Streak<2>(Coord<2>(25, 10), 40)
          << Streak<2>(Coord<2>(15, 30), 30)
          << Streak<2>(Coord<2>(15, 10), 30)
          << Streak<2>(Coord<2>( 5, 30), 60);
        for (StreakIterator<2> i = c.beginStreak(); 
             i != c.endStreak(); ++i)
            actual += *i;
        expected +=
            Streak<2>(Coord<2>(10, 10), 40),
            Streak<2>(Coord<2>(10, 20), 30),
            Streak<2>(Coord<2>( 5, 30), 60);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testUnorderedInsert()
    {
        c << Coord<2>(7, 8);
        TS_ASSERT_EQUALS((unsigned)1, c.streaks[8].size());
        c << Coord<2>(6, 8);
        TS_ASSERT_EQUALS((unsigned)1, c.streaks[8].size());
        c << Coord<2>(9, 8);
        TS_ASSERT_EQUALS((unsigned)2, c.streaks[8].size());
        c << Coord<2>(4, 8);
        TS_ASSERT_EQUALS((unsigned)3, c.streaks[8].size());
        c << Coord<2>(8, 8);
        TS_ASSERT_EQUALS((unsigned)2, c.streaks[8].size());
        TS_ASSERT_EQUALS(Streak<2>(Coord<2>(4, 8),  5), c.streaks[8][4]);
        TS_ASSERT_EQUALS(Streak<2>(Coord<2>(6, 8), 10), c.streaks[8][6]);
        c << Coord<2>(3, 8);
        TS_ASSERT_EQUALS((unsigned)2, c.streaks[8].size());
        c << Coord<2>(2, 8);
        TS_ASSERT_EQUALS((unsigned)2, c.streaks[8].size());
        c << Coord<2>(11, 8);
        TS_ASSERT_EQUALS((unsigned)3, c.streaks[8].size());
        c << Coord<2>(5, 8);
        TS_ASSERT_EQUALS((unsigned)2, c.streaks[8].size());
        c << Coord<2>(10, 8);
        TS_ASSERT_EQUALS((unsigned)1, c.streaks[8].size());
        TS_ASSERT_EQUALS(Streak<2>(Coord<2>(2, 8), 12), c.streaks[8][2]);
    }

    void testBigInsert()
    {
        SuperVector<Coord<2> > res;
        for (SuperVector<Coord<2> >::iterator i = bigInsertShuffled.begin(); i != bigInsertShuffled.end(); i++) 
            c << *i; 

        for (Region<2>::Iterator i = c.begin(); i != c.end(); ++i) 
            res.push_back(*i);
        TS_ASSERT_EQUALS(res, bigInsertOrdered);

        TS_ASSERT_EQUALS(0, (int)c.streaks[ 5].size());

        TS_ASSERT_EQUALS(1, (int)c.streaks[ 6].size());
        TS_ASSERT_EQUALS(1, (int)c.streaks[ 7].size());
        TS_ASSERT_EQUALS(4, (int)c.streaks[ 8].size());
        TS_ASSERT_EQUALS(1, (int)c.streaks[ 9].size());

        TS_ASSERT_EQUALS(1, (int)c.streaks[10].size());
        TS_ASSERT_EQUALS(1, (int)c.streaks[11].size());
        TS_ASSERT_EQUALS(1, (int)c.streaks[12].size());
        TS_ASSERT_EQUALS(1, (int)c.streaks[13].size());
        TS_ASSERT_EQUALS(1, (int)c.streaks[14].size());
        TS_ASSERT_EQUALS(1, (int)c.streaks[15].size());
        TS_ASSERT_EQUALS(1, (int)c.streaks[16].size());
        TS_ASSERT_EQUALS(1, (int)c.streaks[17].size());

        TS_ASSERT_EQUALS(2, (int)c.streaks[18].size());
        TS_ASSERT_EQUALS(2, (int)c.streaks[19].size());
        TS_ASSERT_EQUALS(2, (int)c.streaks[20].size());
        TS_ASSERT_EQUALS(2, (int)c.streaks[21].size());

        TS_ASSERT_EQUALS(1, (int)c.streaks[22].size());

        TS_ASSERT_EQUALS(2, (int)c.streaks[23].size());
        TS_ASSERT_EQUALS(2, (int)c.streaks[24].size());
        TS_ASSERT_EQUALS(2, (int)c.streaks[25].size());
        TS_ASSERT_EQUALS(2, (int)c.streaks[26].size());

        TS_ASSERT_EQUALS(0, (int)c.streaks[27].size());
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
        TS_ASSERT_EQUALS(1, c.streaks[2].size());
        TS_ASSERT_EQUALS(2, c.streaks[3].size());
        TS_ASSERT_EQUALS(2, c.streaks[4].size());
        TS_ASSERT_EQUALS(2, c.streaks[5].size());
        TS_ASSERT_EQUALS(1, c.streaks[6].size());

        Region<2> c1 = c.expand();
        TS_ASSERT_EQUALS(1, c1.streaks[1].size());
        TS_ASSERT_EQUALS(1, c1.streaks[2].size());
        TS_ASSERT_EQUALS(1, c1.streaks[3].size());
        TS_ASSERT_EQUALS(2, c1.streaks[4].size());
        TS_ASSERT_EQUALS(1, c1.streaks[5].size());
        TS_ASSERT_EQUALS(1, c1.streaks[6].size());
        TS_ASSERT_EQUALS(1, c1.streaks[7].size());

        Region<2> c2 = c1.expand();
        for (int i = 0; i < 9; ++i) {
            TS_ASSERT_EQUALS(1, c2.streaks[i].size());
            TS_ASSERT_EQUALS(Streak<2>(Coord<2>(0, i), 9), c2.streaks[i].begin()->second);
        }
    }

    void testExpand2()
    {
        for(int x = 0; x < 10; ++x)
            for(int y = 0; y < 20; ++y)
                c << Coord<2>(x, y);
        Region<2> actual = c.expand(20);
        Region<2> expected;
        for(int x = -20; x < 30; ++x)
            for(int y = -20; y < 40; ++y)
                expected << Coord<2>(x, y);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testDelete()
    {
        Region<2>::StreakMap expected;
        expected[1].clear();
        expected[3][0] = Streak<2>(Coord<2>(0, 3),  5);
        expected[5][0] = Streak<2>(Coord<2>(0, 5),  5);
        expected[5][6] = Streak<2>(Coord<2>(6, 5), 10);
        expected[7][0] = Streak<2>(Coord<2>(0, 7), 10);
        expected[9][5] = Streak<2>(Coord<2>(5, 9), 10);

        for(int x = 0; x < 10; ++x)
            for(int y = 0; y < 10; ++y)
                c << Coord<2>(x, y);
        c >> Streak<2>(Coord<2>(-10, 1), 20);
        c >> Streak<2>(Coord<2>(  5, 3), 20);
        c >> Coord<2>(5, 5);
        c >> Streak<2>(Coord<2>(  5, 7),  5);
        c >> Streak<2>(Coord<2>(-20, 9),  5);

        TS_ASSERT_EQUALS(1, c.streaks[0].size());
        TS_ASSERT_EQUALS(1, c.streaks[2].size());
        TS_ASSERT_EQUALS(1, c.streaks[4].size());
        TS_ASSERT_EQUALS(1, c.streaks[6].size());
        TS_ASSERT_EQUALS(1, c.streaks[8].size());
        for (Region<2>::StreakMap::iterator i = expected.begin(); i != expected.end(); ++i) 
            TS_ASSERT_EQUALS(i->second, c.streaks[i->first]);
    }

    void testAndNot1()
    {
        Region<2> minuend, expected;
        for(int x = 0; x < 10; ++x)
            for(int y = 0; y < 10; ++y)
                c << Coord<2>(x, y);
        for(int x = 5; x < 20; ++x)
            for(int y = 5; y < 20; ++y)
                minuend << Coord<2>(x, y);
        for(int x = 0; x < 10; ++x)
            for(int y = 0; y < 5; ++y)
                expected << Coord<2>(x, y);
        for(int x = 0; x < 5; ++x)
            for(int y = 5; y < 10; ++y)
                expected << Coord<2>(x, y);
        TS_ASSERT_EQUALS(expected, c - minuend);
    }

    void testAndNot2()
    {
        Region<2> expanded, expected;
        for(int x = 0; x < 10; ++x)
            for(int y = 0; y < 10; ++y)
                c << Coord<2>(x, y);
        for(int i = -1; i < 11; ++i) {
            expected << Coord<2>( i, -1);
            expected << Coord<2>( i, 10);
            expected << Coord<2>(10,  i);
            expected << Coord<2>(-1,  i);
        }
        expanded = c.expand();
        TS_ASSERT_EQUALS(expected, expanded -c);
    }

    void testAndAssignmentOperator()
    {
        Region<2> original, mask, expected;
        for(int x = 0; x < 10; ++x)
            for(int y = 0; y < 10; ++y)
                original << Coord<2>(x, y);
        for(int x = 3; x < 15; ++x)
            for(int y = 5; y < 15; ++y)
                mask << Coord<2>(x, y);
        for(int x = 3; x < 10; ++x)
            for(int y = 5; y < 10; ++y)
                expected << Coord<2>(x, y);
        original &= mask;
        TS_ASSERT_EQUALS(original, expected);
    }
    
    void testAndOperator()
    {
        Region<2> original, mask, expected;
        for(int x = 0; x < 10; ++x)
            for(int y = 0; y < 10; ++y)
                original << Coord<2>(x, y);
        for(int x = 3; x < 15; ++x)
            for(int y = 5; y < 15; ++y)
                mask << Coord<2>(x, y);
        for(int x = 3; x < 10; ++x)
            for(int y = 5; y < 10; ++y)
                expected << Coord<2>(x, y);
        TS_ASSERT_EQUALS(original & mask, expected);
    }

    void testAddAssignmentOperator()
    {
        Region<2> original, addent, expected;
        for(int x = 0; x < 10; ++x)
            for(int y = 0; y < 10; ++y)
                original << Coord<2>(x, y);
        for(int x = 3; x < 15; ++x)
            for(int y = 0; y < 10; ++y)
                addent << Coord<2>(x, y);
        for(int x = 0; x < 15; ++x)
            for(int y = 0; y < 10; ++y)
                expected << Coord<2>(x, y);
        original += addent;
        TS_ASSERT_EQUALS(original, expected);
    }

    void testAddOperator()
    {
        Region<2> original, addent, expected;
        for(int x = 0; x < 10; ++x)
            for(int y = 0; y < 10; ++y)
                original << Coord<2>(x, y);
        for(int x = 3; x < 15; ++x)
            for(int y = 0; y < 10; ++y)
                addent << Coord<2>(x, y);
        for(int x = 0; x < 15; ++x)
            for(int y = 0; y < 10; ++y)
                expected << Coord<2>(x, y);
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
        TS_ASSERT_EQUALS(0, a.numStreaks());
        a << Streak<2>(Coord<2>(10, 10), 20);
        TS_ASSERT_EQUALS(1, a.numStreaks());
        a << Streak<2>(Coord<2>(30, 10), 60);
        TS_ASSERT_EQUALS(2, a.numStreaks());
        a << Streak<2>(Coord<2>(10, 20), 20);
        TS_ASSERT_EQUALS(3, a.numStreaks());
        a << Streak<2>(Coord<2>(15, 10), 25);
        TS_ASSERT_EQUALS(3, a.numStreaks());
        a >> Streak<2>(Coord<2>(15, 10), 17);
        TS_ASSERT_EQUALS(4, a.numStreaks());
        a << Streak<2>(Coord<2>(12, 10), 30);
        TS_ASSERT_EQUALS(2, a.numStreaks());
    }

    void testToVector()
    {
        Region<2> a;
        TS_ASSERT_EQUALS(0, a.toVector().size());
        a << Streak<2>(Coord<2>(10, 10), 20);
        TS_ASSERT_EQUALS(1, a.toVector().size());
        a << Streak<2>(Coord<2>(30, 10), 60);
        TS_ASSERT_EQUALS(2, a.toVector().size());
        a << Streak<2>(Coord<2>(10, 20), 20);
        TS_ASSERT_EQUALS(3, a.toVector().size());
        a << Streak<2>(Coord<2>(15, 10), 25);
        TS_ASSERT_EQUALS(3, a.toVector().size());
        a >> Streak<2>(Coord<2>(15, 10), 17);
        TS_ASSERT_EQUALS(4, a.toVector().size());
        SuperVector<Streak<2> > vec = a.toVector();
        TS_ASSERT_EQUALS(a, Region<2>(vec.begin(), vec.end()));
        a << Streak<2>(Coord<2>(12, 10), 30);        
        TS_ASSERT_EQUALS(2, a.toVector().size());
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
        for (int y = origin.y(); y != (origin.y() + dimensions.y()); ++y)
            expected << Streak<2>(Coord<2>(origin.x(), y), origin.x() + dimensions.x());
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testClear()
    {
        Region<2> a;
        a << Streak<2>(Coord<2>(10, 10), 20);
        TS_ASSERT(!a.empty());

        a.clear();
        TS_ASSERT(a.empty());
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(0, 0), Coord<2>(0, 0)), a.boundingBox());
    }

    void test3DSimple1()
    {
        Region<3> a;
        SuperVector<Streak<3> > streaks;
        SuperVector<Streak<3> > retrievedStreaks;
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
        
        for (StreakIterator<3> i = a.beginStreak(); i != a.endStreak(); ++i)
            retrievedStreaks << *i;

        TS_ASSERT_EQUALS(retrievedStreaks.size(), streaks.size());
        TS_ASSERT_EQUALS(retrievedStreaks, streaks);
    }

    void test3DSimple2()
    {
        Region<3> a;
        SuperVector<Streak<3> > streaks;
        SuperVector<Streak<3> > retrievedStreaks;
        streaks << Streak<3>(Coord<3>(10, 10, 10), 20)
                << Streak<3>(Coord<3>(11, 10, 10), 20)
                << Streak<3>(Coord<3>(12, 10, 10), 20);
        a << streaks[0]
          << streaks[1]
          << streaks[2];
        
        for (StreakIterator<3> i = a.beginStreak(); i != a.endStreak(); ++i)
            retrievedStreaks << *i;

        TS_ASSERT_EQUALS(retrievedStreaks.size(), 1);
        TS_ASSERT_EQUALS(retrievedStreaks[0], Streak<3>(Coord<3>(10, 10, 10), 20));
    }

    void test3DSimple3()
    {
        Region<3> a;
        SuperVector<Streak<3> > retrievedStreaks;
        a << Streak<3>(Coord<3>(10, 10, 10), 20);
        a >> Streak<3>(Coord<3>(15, 10, 10), 16);
        
        for (StreakIterator<3> i = a.beginStreak(); i != a.endStreak(); ++i)
            retrievedStreaks << *i;

        TS_ASSERT_EQUALS(retrievedStreaks.size(), 2);
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

        for (Region<3>::Iterator i = a.begin(); i != a.end(); ++i)
            actual.insert(*i);

        TS_ASSERT_EQUALS(expected, actual);
        TS_ASSERT_EQUALS(a.boundingBox(), CoordBox<3>(Coord<3>(), Coord<3>(10, 10, 10)));
    }

    void testExpand3D()
    {
        Region<3> actual, expected, base;

        for (int z = -3; z < 13; ++z) 
            for (int y = -3; y < 13; ++y) 
                for (int x = -3; x < 28; ++x) 
                    expected << Coord<3>(x, y, z);

        for (int z = 0; z < 10; ++z) 
            for (int y = 0; y < 10; ++y) 
                for (int x = 0; x < 10; ++x) 
                    base << Coord<3>(x, y, z)
                         << Coord<3>(x + 15, y, z);

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

    // void testBenchStriping()
    // {
    //     int maxX = benchDim().x();
    //     int maxY = benchDim().y();
    //     int res = 0;

    //     long long tStart = Chronometer::timeUSec();
    //     for (int c = 0; c < 2048; ++c) {
    //         for (int y = 0; y < maxY; ++y)
    //             for (int x = 0; x < maxX; ++x) {
    //                 res += bongo(Coord<2>(x, y));
    //                 res %= 13;
    //             }
            
    //     }
    //     long long tEnd = Chronometer::timeUSec();

    //     std::cout << "\n  striping:        " << (tEnd - tStart) << " res: " << res << "\n";
    // }

    // void testBenchRegionVanilla()
    // {        
    //     int res = 0;
    //     Region<2> r;
    //     for (int y = 0; y < benchDim().y(); ++y)
    //         r << Streak<2>(Coord<2>(0, y), benchDim().x());

    //     long long tStart = Chronometer::timeUSec();
    //     for (int c = 0; c < 2048; ++c) {
    //         for (Region<2>::Iterator i = r.begin(); i != r.end(); ++i) {
    //             res += bongo(*i);
    //             res %= 13;
    //         }
    //     }
    //     long long tEnd = Chronometer::timeUSec();

    //     std::cout << " region vanilla:  " << (tEnd - tStart) << " res: " << res << "\n";
    // }

    // void testBenchRegionBetter()
    // {        
    //     int res = 0;
    //     Region<2> r;
    //     for (int y = 0; y < benchDim().y(); ++y)
    //         r << Streak<2>(Coord<2>(0, y), benchDim().x());

    //     long long tStart = Chronometer::timeUSec();
    //     Region<2>::Iterator end = r.end();
    //     for (int c = 0; c < 2048; ++c) {
    //         Coord<2> end = benchDim() - Coord<2>(1, 1);
    //         for (Region<2>::Iterator i = r.begin(); *i != end; ++i) {
    //             res += bongo(*i);
    //             res %= 13;
    //         }
    //     }
    //     long long tEnd = Chronometer::timeUSec();

    //     std::cout << " region better:   " << (tEnd - tStart) << " res: " << res << "\n";
    // }

    // void testBenchRegionBest()
    // {        
    //     int res = 0;
    //     Region<2> r;
    //     for (int y = 0; y < benchDim().y(); ++y)
    //         r << Streak<2>(Coord<2>(0, y), benchDim().x());

    //     long long tStart = Chronometer::timeUSec();
    //     Region<2>::Iterator end = r.end();
    //     for (int c = 0; c < 2048; ++c) {
    //         Coord<2> end = benchDim();
    //         StreakIterator<2> endStreak = r.endStreak();
    //         for (StreakIterator<2> s = r.beginStreak(); s != endStreak; ++s) {
    //             int y = s->origin.y();
    //             int endX = s->endX;
    //             for (int x = s->origin.x(); x < endX; ++x) {
    //                 res += bongo(Coord<2>(x, y));
    //                 res %= 13;
    //             }
    //         }
    //     }
    //     long long tEnd = Chronometer::timeUSec();

    //     std::cout << " region best:     " << (tEnd - tStart) << " res: " << res << "\n";
    // }

    // void testBenchRegionBest2()
    // {        
    //     int res = 0;
    //     Region<2> r;
    //     for (int y = 0; y < benchDim().y(); ++y)
    //         r << Streak<2>(Coord<2>(0, y), benchDim().x());

    //     long long tStart = Chronometer::timeUSec();
    //     Region<2>::Iterator end = r.end();
    //     for (int c = 0; c < 2048; ++c) {
    //         Coord<2> end = benchDim();
    //         for (StreakIterator<2> s = r.beginStreak(); !s.endReached(); ++s) {
    //             int y = s->origin.y();
    //             int endX = s->endX;
    //             for (int x = s->origin.x(); x < endX; ++x) {
    //                 res += bongo(Coord<2>(x, y));
    //                 res %= 13;
    //             }
    //         }
    //     }
    //     long long tEnd = Chronometer::timeUSec();

    //     std::cout << " region best2:    " << (tEnd - tStart) << " res: " << res << "\n";
    // }

private:
    Region<2> c;
    Coord<2>::Vector bigInsertOrdered;
    Coord<2>::Vector bigInsertShuffled;        

    Coord<2>::Vector transform(const SuperVector<std::string>& shape)
    {
        Coord<2>::Vector ret;
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
