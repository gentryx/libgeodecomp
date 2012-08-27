#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/misc/newregion.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/stripingpartition.h>

using namespace boost::assign;
using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {

class NewRegionTest : public CxxTest::TestSuite 
{
public:
    typedef SuperVector<Coord<2> > CoordVector;
    typedef std::pair<int, int> IntPair;

    void setUp()
    {
        c = NewRegion<2>();

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
        NewRegionInsertHelper<0> h;
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
        NewRegionRemoveHelper<0> h;
        
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
        NewRegionInsertHelper<0> h;
        
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
        NewRegionRemoveHelper<0> h;
        NewRegion<2>::VecType expected;

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

        TS_ASSERT_EQUALS(1, c.indices[1].size());
        TS_ASSERT_EQUALS(2, c.indices[0].size());
    }
 
    void testInsert1b()
    {
        c << Coord<2>(12, 10)
          << Coord<2>(14, 10)
          << Coord<2>(10, 10);

        TS_ASSERT_EQUALS(1, c.indices[1].size());
        TS_ASSERT_EQUALS(3, c.indices[0].size());
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

        TS_ASSERT_EQUALS(4, c.indices[1].size());
        TS_ASSERT_EQUALS(7, c.indices[0].size());
    }

    void testInsertCoordBox()
    {
        NewRegion<3> expected, actual;
        
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
        NewRegion<3> r;
        TS_ASSERT_EQUALS(r.indices[0].size(), 0);
        TS_ASSERT_EQUALS(r.indices[1].size(), 0);
        TS_ASSERT_EQUALS(r.indices[2].size(), 0);

        r << Streak<3>(Coord<3>(10, 20, 30), 40);
        TS_ASSERT_EQUALS(r.indices[0].size(), 1);
        TS_ASSERT_EQUALS(r.indices[1].size(), 1);
        TS_ASSERT_EQUALS(r.indices[2].size(), 1);

        r << Streak<3>(Coord<3>(12, 22, 31), 42);
        TS_ASSERT_EQUALS(r.indices[0].size(), 2);
        TS_ASSERT_EQUALS(r.indices[1].size(), 2);
        TS_ASSERT_EQUALS(r.indices[2].size(), 2);

        r << Streak<3>(Coord<3>(14, 24, 29), 44);
        TS_ASSERT_EQUALS(r.indices[0].size(), 3);
        TS_ASSERT_EQUALS(r.indices[1].size(), 3);
        TS_ASSERT_EQUALS(r.indices[2].size(), 3);

        r << Streak<3>(Coord<3>(16, 21, 30), 46);
        TS_ASSERT_EQUALS(r.indices[0].size(), 4);
        TS_ASSERT_EQUALS(r.indices[1].size(), 4);
        TS_ASSERT_EQUALS(r.indices[2].size(), 3);

        r << Streak<3>(Coord<3>(58, 20, 30), 68);
        TS_ASSERT_EQUALS(r.indices[0].size(), 5);
        TS_ASSERT_EQUALS(r.indices[1].size(), 4);
        TS_ASSERT_EQUALS(r.indices[2].size(), 3);

        r << Streak<3>(Coord<3>(59, 19, 29), 69);
        TS_ASSERT_EQUALS(r.indices[0].size(), 6);
        TS_ASSERT_EQUALS(r.indices[1].size(), 5);
        TS_ASSERT_EQUALS(r.indices[2].size(), 3);

        r << Streak<3>(Coord<3>(38, 20, 30), 60);
        TS_ASSERT_EQUALS(r.indices[0].size(), 5);
        TS_ASSERT_EQUALS(r.indices[1].size(), 5);
        TS_ASSERT_EQUALS(r.indices[2].size(), 3);
    }

    void testStreakIteration()
    {
        SuperVector<Streak<2> > actual, expected;
        for (NewRegion<2>::StreakIterator i = c.beginStreak(); 
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

        for (NewRegion<2>::StreakIterator i = c.beginStreak(); 
             i != c.endStreak(); ++i) {
            actual += *i;
        }

        expected +=
            Streak<2>(Coord<2>(10, 10), 40),
            Streak<2>(Coord<2>(10, 20), 30),
            Streak<2>(Coord<2>( 5, 30), 60);
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testUnorderedInsert()
    {
        c << Coord<2>(7, 8);
        TS_ASSERT_EQUALS(1, c.indices[0].size());
        c << Coord<2>(6, 8);
        TS_ASSERT_EQUALS(1, c.indices[0].size());
        c << Coord<2>(9, 8);
        TS_ASSERT_EQUALS(2, c.indices[0].size());
        c << Coord<2>(4, 8);
        TS_ASSERT_EQUALS(3, c.indices[0].size());
        c << Coord<2>(8, 8);
        TS_ASSERT_EQUALS(2, c.indices[0].size());
        TS_ASSERT_EQUALS(IntPair(4,  5), c.indices[0][0]);
        TS_ASSERT_EQUALS(IntPair(6, 10), c.indices[0][1]);

        c << Coord<2>(3, 8);
        TS_ASSERT_EQUALS(2, c.indices[0].size());
        c << Coord<2>(2, 8);
        TS_ASSERT_EQUALS(2, c.indices[0].size());
        c << Coord<2>(11, 8);
        TS_ASSERT_EQUALS(3, c.indices[0].size());
        c << Coord<2>(5, 8);
        TS_ASSERT_EQUALS(2, c.indices[0].size());
        c << Coord<2>(10, 8);
        TS_ASSERT_EQUALS(1, c.indices[0].size());
        TS_ASSERT_EQUALS(IntPair(2, 12), c.indices[0][0]);
    }

    void testBigInsert()
    {
        CoordVector res;
        for (CoordVector::iterator i = bigInsertShuffled.begin(); 
             i != bigInsertShuffled.end(); i++) {
            c << *i; 
        }

        for (NewRegion<2>::Iterator i = c.begin(); i != c.end(); ++i) {
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
        NewRegion<2> c;
        TS_ASSERT_EQUALS(c.empty(), true);
        c << Coord<2>(1, 2);
        TS_ASSERT_EQUALS(c.empty(), false);
    }

    void testBoundingBox()
    {
        NewRegion<2> c;
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
        NewRegion<2> c;
        TS_ASSERT_EQUALS(0, c.size());
        c << Streak<2>(Coord<2>(10, 10), 20);
        TS_ASSERT_EQUALS(10, c.size());
        c << Streak<2>(Coord<2>(17, 20), 98);
        TS_ASSERT_EQUALS(91, c.size());
        c >> Streak<2>(Coord<2>(15, 10), 18);
        TS_ASSERT_EQUALS(88, c.size());
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

        NewRegion<2> c1 = c.expand();
        TS_ASSERT_EQUALS(0, c1.indices[1][0].second);
        TS_ASSERT_EQUALS(1, c1.indices[1][1].second);
        TS_ASSERT_EQUALS(2, c1.indices[1][2].second);
        TS_ASSERT_EQUALS(3, c1.indices[1][3].second);
        TS_ASSERT_EQUALS(5, c1.indices[1][4].second);
        TS_ASSERT_EQUALS(6, c1.indices[1][5].second);
        TS_ASSERT_EQUALS(7, c1.indices[1][6].second);
        TS_ASSERT_EQUALS(8, c1.indices[1][7].second);

        NewRegion<2> c2 = c1.expand();
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

        NewRegion<2> actual = c.expand(20);
        NewRegion<2> expected;

        for(int x = -20; x < 30; ++x) {
            for(int y = -20; y < 40; ++y) {
                expected << Coord<2>(x, y);
            }
        }

        TS_ASSERT_EQUALS(actual, expected);
    }

    void testDelete()
    {
        NewRegion<2>::VecType expected;
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
        NewRegion<2> minuend, expected;
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
        NewRegion<2> expanded, expected;
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
        TS_ASSERT_EQUALS(expected, expanded -c);
    }

    void testAndAssignmentOperator()
    {
        NewRegion<2> original, mask, expected;
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
        NewRegion<2> original, mask, expected;
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
        NewRegion<2> original, addent, expected;
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
        NewRegion<2> original, addent, expected;
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
        NewRegion<2> a, b;
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
        NewRegion<2> a;
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
        NewRegion<2> a;
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
        TS_ASSERT_EQUALS(a, NewRegion<2>(vec.begin(), vec.end()));
        a << Streak<2>(Coord<2>(12, 10), 30);        
        TS_ASSERT_EQUALS(2, a.toVector().size());
        vec = a.toVector();
        TS_ASSERT_EQUALS(a, NewRegion<2>(vec.begin(), vec.end()));
    }

    void testIteratorInsertConstructor()
    {
        Coord<2> origin(123, 456);
        Coord<2> dimensions(200, 100);
        StripingPartition<2> s(origin, dimensions);
        NewRegion<2> actual(s.begin(), s.end());
        NewRegion<2> expected;
        for (int y = origin.y(); y != (origin.y() + dimensions.y()); ++y) {
            expected << Streak<2>(Coord<2>(origin.x(), y), 
                                  origin.x() + dimensions.x());
        }
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testClear()
    {
        NewRegion<2> a;
        a << Streak<2>(Coord<2>(10, 10), 20);
        TS_ASSERT(!a.empty());

        a.clear();
        TS_ASSERT(a.empty());
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(0, 0), Coord<2>(0, 0)), 
                         a.boundingBox());
    }

    void test3DSimple1()
    {
        NewRegion<3> a;
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
        
        for (NewRegion<3>::StreakIterator i = a.beginStreak(); i != a.endStreak(); ++i) {
            retrievedStreaks << *i;
        }

        TS_ASSERT_EQUALS(retrievedStreaks.size(), streaks.size());
        TS_ASSERT_EQUALS(retrievedStreaks, streaks);
    }

    void test3DSimple2()
    {
        NewRegion<3> a;
        SuperVector<Streak<3> > streaks;
        SuperVector<Streak<3> > retrievedStreaks;
        streaks << Streak<3>(Coord<3>(10, 10, 10), 20)
                << Streak<3>(Coord<3>(11, 10, 10), 20)
                << Streak<3>(Coord<3>(12, 10, 10), 20);
        a << streaks[0]
          << streaks[1]
          << streaks[2];
        
        for (NewRegion<3>::StreakIterator i = a.beginStreak(); i != a.endStreak(); ++i) {
            retrievedStreaks << *i;
        }

        TS_ASSERT_EQUALS(retrievedStreaks.size(), 1);
        TS_ASSERT_EQUALS(retrievedStreaks[0], Streak<3>(Coord<3>(10, 10, 10), 20));
    }

    void test3DSimple3()
    {
        NewRegion<3> a;
        SuperVector<Streak<3> > retrievedStreaks;
        a << Streak<3>(Coord<3>(10, 10, 10), 20);
        a >> Streak<3>(Coord<3>(15, 10, 10), 16);
        
        for (NewRegion<3>::StreakIterator i = a.beginStreak(); i != a.endStreak(); ++i) {
            retrievedStreaks << *i;
        }

        TS_ASSERT_EQUALS(retrievedStreaks.size(), 2);
        TS_ASSERT_EQUALS(retrievedStreaks[0], Streak<3>(Coord<3>(10, 10, 10), 15));
        TS_ASSERT_EQUALS(retrievedStreaks[1], Streak<3>(Coord<3>(16, 10, 10), 20));
    }

    void test3DSimple4()
    {
        NewRegion<3> a;
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

        for (NewRegion<3>::Iterator i = a.begin(); i != a.end(); ++i) {
            actual.insert(*i);
        }

        TS_ASSERT_EQUALS(expected, actual);
        TS_ASSERT_EQUALS(a.boundingBox(), CoordBox<3>(Coord<3>(), Coord<3>(10, 10, 10)));
    }

    void testExpand3D()
    {
        NewRegion<3> actual, expected, base;

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
        NewRegion<2> region;
        region << Streak<2>(Coord<2>(0, 0), 15)
               << Streak<2>(Coord<2>(0, 1), 20)
               << Streak<2>(Coord<2>(0, 2), 20);

        NewRegion<2> actual = region.expandWithTopology(
            2, 
            Coord<2>(20, 20),
            Topologies::Torus<2>::Topology());

        NewRegion<2> expected;
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
        NewRegion<2> region;
        region << Streak<2>(Coord<2>(0, 0), 15)
               << Streak<2>(Coord<2>(0, 1), 20)
               << Streak<2>(Coord<2>(0, 2), 20);

        NewRegion<2> actual = region.expandWithTopology(
            2, 
            Coord<2>(20, 20),
            Topologies::Cube<2>::Topology());

        NewRegion<2> expected;
        expected << Streak<2>(Coord<2>(0,   0), 20)
                 << Streak<2>(Coord<2>(0,   1), 20)
                 << Streak<2>(Coord<2>(0,   2), 20)
                 << Streak<2>(Coord<2>(0,   3), 20)
                 << Streak<2>(Coord<2>(0,   4), 20);

        TS_ASSERT_EQUALS(actual, expected);
    }

    void testBoolean3D()
    {
        NewRegion<3> leftCube, rightCube, frontCube, mergerLR, mergerFL;

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

    // void testBig3D()
    // {
    //     std::cout << "MARK1------------------------------\n"
    //               << "sizeof(SuperMap<int, Streak<3> >) = " << sizeof(SuperMap<int, Streak<3> >) << "\n"
    //               << "sizeof(Streak<3>) = " << sizeof(Streak<3>) << "\n";
    //     NewRegion<3> region;
    //     Coord<3> dim(2000, 2000, 1427);
    //     for (int z = 0; z < dim.z(); ++z) {
    //         for (int y = 0; y < dim.y(); ++y) {
    //             region << Streak<3>(Coord<3>(0, y, z), dim.x());
    //         }
    //     }
    //     std::cout << region.boundingBox() << "\n";

    //     std::cout << "MARK2------------------------------\n";
    //     NewRegion<3> expanded = region.expandWithTopology(
    //         1, 
    //         dim,
    //         Topologies::Cube<3>::Topology());
    //     std::cout << expanded.boundingBox() << "\n";
    // }

    // fixme: move to performance test
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
    //     NewRegion<2> r;
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
    //     NewRegion<2> r;
    //     for (int y = 0; y < benchDim().y(); ++y)
    //         r << Streak<2>(Coord<2>(0, y), benchDim().x());

    //     long long tStart = Chronometer::timeUSec();
    //     NewRegion<2>::Iterator end = r.end();
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
    //     NewRegion<2> r;
    //     for (int y = 0; y < benchDim().y(); ++y)
    //         r << Streak<2>(Coord<2>(0, y), benchDim().x());

    //     long long tStart = Chronometer::timeUSec();
    //     NewRegion<2>::Iterator end = r.end();
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
    //     NewRegion<2> r;
    //     for (int y = 0; y < benchDim().y(); ++y)
    //         r << Streak<2>(Coord<2>(0, y), benchDim().x());

    //     long long tStart = Chronometer::timeUSec();
    //     NewRegion<2>::Iterator end = r.end();
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

    void testRemove3D()
    {
        NewRegion<3> r;
        TS_ASSERT_EQUALS(r.size(), 0);
        TS_ASSERT_EQUALS(r.boundingBox(), CoordBox<3>(Coord<3>(), Coord<3>()));

        r << Streak<3>(Coord<3>(10, 20, 30), 40);
        TS_ASSERT_EQUALS(r.indices[0].size(), 1);
        TS_ASSERT_EQUALS(r.indices[1].size(), 1);
        TS_ASSERT_EQUALS(r.indices[2].size(), 1);
        TS_ASSERT_EQUALS(r.size(), 30);
        TS_ASSERT_EQUALS(r.boundingBox(), 
                         CoordBox<3>(Coord<3>(10, 20, 30), Coord<3>(30, 1, 1)));

        r >> Streak<3>(Coord<3>(15, 20, 30), 35);
        TS_ASSERT_EQUALS(r.indices[0].size(), 2);
        TS_ASSERT_EQUALS(r.indices[1].size(), 1);
        TS_ASSERT_EQUALS(r.indices[2].size(), 1);

        r >> Streak<3>(Coord<3>(36, 20, 30), 37);
        TS_ASSERT_EQUALS(r.indices[0].size(), 3);
        TS_ASSERT_EQUALS(r.indices[1].size(), 1);
        TS_ASSERT_EQUALS(r.indices[2].size(), 1);

        r >> Streak<3>(Coord<3>(30, 20, 30), 50);
        TS_ASSERT_EQUALS(r.indices[0].size(), 1);
        TS_ASSERT_EQUALS(r.indices[1].size(), 1);
        TS_ASSERT_EQUALS(r.indices[2].size(), 1);

        r << Streak<3>(Coord<3>(40, 21, 29), 60);
        TS_ASSERT_EQUALS(r.indices[0].size(), 2);
        TS_ASSERT_EQUALS(r.indices[1].size(), 2);
        TS_ASSERT_EQUALS(r.indices[2].size(), 2);

        r >> Streak<3>(Coord<3>(50, 21, 29), 55);
        TS_ASSERT_EQUALS(r.indices[0].size(), 3);
        TS_ASSERT_EQUALS(r.indices[1].size(), 2);
        TS_ASSERT_EQUALS(r.indices[2].size(), 2);

        r >> Streak<3>(Coord<3>(35, 21, 29), 60);
        TS_ASSERT_EQUALS(r.indices[0].size(), 1);
        TS_ASSERT_EQUALS(r.indices[1].size(), 1);
        TS_ASSERT_EQUALS(r.indices[2].size(), 1);

        TS_ASSERT_EQUALS(r.indices[1][0].second, 0);
        TS_ASSERT_EQUALS(r.indices[2][0].second, 0);
        
        r >> Streak<3>(Coord<3>(10, 20, 30), 15);

        TS_ASSERT_EQUALS(r.indices[0].size(), 0);
        TS_ASSERT_EQUALS(r.indices[1].size(), 0);
        TS_ASSERT_EQUALS(r.indices[2].size(), 0);
    }

    void testStreakIterator()
    {
        SuperVector<Streak<3> > expected;
        SuperVector<Streak<3> > actual;

        NewRegion<3> r;
        TS_ASSERT_EQUALS(r.beginStreak(), r.endStreak());


        Streak<3> newStreak(Coord<3>(10, 10, 10), 20);
        r << newStreak;
        TS_ASSERT_EQUALS(newStreak, *r.beginStreak());
        TS_ASSERT_EQUALS(1, r.numStreaks());
        TS_ASSERT_EQUALS(10, r.size());
        for (NewRegion<3>::StreakIterator i = r.beginStreak(); i != r.endStreak(); ++i) {
            actual << *i;
        }
        expected << newStreak;
        TS_ASSERT_EQUALS(actual, expected);

        newStreak = Streak<3>(Coord<3>(10, 20, 10), 20);
        expected << newStreak;
        r << newStreak;
        TS_ASSERT_EQUALS(2, r.numStreaks());
        TS_ASSERT_EQUALS(20, r.size());
        actual.clear();
        for (NewRegion<3>::StreakIterator i = r.beginStreak(); i != r.endStreak(); ++i) {
            actual << *i;
        }
        TS_ASSERT_EQUALS(actual, expected);

        newStreak = Streak<3>(Coord<3>(30, 20, 10), 40);
        expected << newStreak;
        r << newStreak;
        TS_ASSERT_EQUALS(3, r.numStreaks());
        TS_ASSERT_EQUALS(30, r.size());
        actual.clear();
        for (NewRegion<3>::StreakIterator i = r.beginStreak(); i != r.endStreak(); ++i) {
            actual << *i;
        }
        TS_ASSERT_EQUALS(actual, expected);

        newStreak = Streak<3>(Coord<3>(10, 20, 11), 20);
        expected << newStreak;
        r << newStreak;
        TS_ASSERT_EQUALS(4, r.numStreaks());
        TS_ASSERT_EQUALS(40, r.size());
        actual.clear();
        for (NewRegion<3>::StreakIterator i = r.beginStreak(); i != r.endStreak(); ++i) {
            actual << *i;
        }
        TS_ASSERT_EQUALS(actual, expected);
    }

    void testNormalIterator()
    {
        SuperVector<Coord<3> > expected;
        SuperVector<Coord<3> > actual;

        NewRegion<3> r;
        r << Streak<3>(Coord<3>(10, 20, 30), 40);
        r << Streak<3>(Coord<3>(50, 60, 70), 80);
        for (NewRegion<3>::Iterator i = r.begin(); i != r.end(); ++i) {
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

private:
    NewRegion<2> c;
    CoordVector bigInsertOrdered;
    CoordVector bigInsertShuffled;        

    CoordVector transform(const SuperVector<std::string>& shape)
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
