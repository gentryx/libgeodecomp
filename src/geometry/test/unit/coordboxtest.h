#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CoordBoxTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        origin = Coord<2>(3,4);
        width = 4;
        height = 3;
        rect = CoordBox<2>(origin, Coord<2>(width, height));
    }

    void testConstructor1()
    {
        TS_ASSERT_EQUALS(origin, rect.origin);
        TS_ASSERT_EQUALS(width,  rect.dimensions.x());
        TS_ASSERT_EQUALS(height, rect.dimensions.y());
    }

    void testConstructor2()
    {
        CoordBox<2> rect(Coord<2>(1, 2), Coord<2>(3, 4));
        TS_ASSERT_EQUALS(Coord<2>(1, 2), rect.origin);
        TS_ASSERT_EQUALS(3, rect.dimensions.x());
        TS_ASSERT_EQUALS(4, rect.dimensions.y());
    }

    void testInBounds()
    {
        TS_ASSERT(rect.inBounds(Coord<2>(3, 4)));
        TS_ASSERT(rect.inBounds(Coord<2>(6, 4)));
        TS_ASSERT(rect.inBounds(Coord<2>(3, 6)));
        TS_ASSERT(rect.inBounds(Coord<2>(6, 6)));
        TS_ASSERT(rect.inBounds(Coord<2>(4, 4)));

        TS_ASSERT(!rect.inBounds(Coord<2>(0, 0)));
        TS_ASSERT(!rect.inBounds(Coord<2>(100, 0)));
        TS_ASSERT(!rect.inBounds(Coord<2>(0, 100)));
        TS_ASSERT(!rect.inBounds(Coord<2>(5, 8)));
        TS_ASSERT(!rect.inBounds(Coord<2>(8, 5)));

        CoordBox<3> rect(Coord<3>(3, 4, 5), Coord<3>(30, 20, 10));
        TS_ASSERT(rect.inBounds(Coord<3>( 3,  4,  5)));
        TS_ASSERT(rect.inBounds(Coord<3>(10, 10,  10)));
        TS_ASSERT(rect.inBounds(Coord<3>(32,  4,  5)));
        TS_ASSERT(rect.inBounds(Coord<3>( 3, 23,  5)));
        TS_ASSERT(rect.inBounds(Coord<3>( 3,  4, 14)));

        TS_ASSERT(!rect.inBounds(Coord<3>(10,  3,  9)));
        TS_ASSERT(!rect.inBounds(Coord<3>(33,  4,  5)));
        TS_ASSERT(!rect.inBounds(Coord<3>( 3, 24,  5)));
        TS_ASSERT(!rect.inBounds(Coord<3>( 3,  4, 15)));
    }

    void testSize()
    {
        CoordBox<2> rect1(Coord<2>(-2, 3), Coord<2>(5, 7));
        TS_ASSERT_EQUALS(rect1.size(), (unsigned)35);

        CoordBox<3> rect2(Coord<3>(-2, 3, -4), Coord<3>(5, 7, 8));
        TS_ASSERT_EQUALS(rect2.size(), (unsigned)280);
    }

    void testIntersects()
    {
        CoordBox<2> rect(Coord<2>(10, 20), Coord<2>(30, 40));
        TS_ASSERT_EQUALS(
            true, rect.intersects(CoordBox<2>(Coord<2>(20, 10),
                                              Coord<2>(10, 60))));
        TS_ASSERT_EQUALS(
            true, rect.intersects(CoordBox<2>(Coord<2>(20, 30),
                                              Coord<2>(10, 10))));
        TS_ASSERT_EQUALS(
            true, rect.intersects(CoordBox<2>(Coord<2>(20, 30),
                                              Coord<2>(90, 90))));
        TS_ASSERT_EQUALS(
            false, rect.intersects(CoordBox<2>(Coord<2>(10, 10),
                                               Coord<2>(10, 10))));
        TS_ASSERT_EQUALS(
            false, rect.intersects(CoordBox<2>(Coord<2>(40, 20),
                                               Coord<2>(10, 10))));
        TS_ASSERT_EQUALS(
            false, rect.intersects(CoordBox<2>(Coord<2>(40, 60),
                                               Coord<2>(10, 10))));

        CoordBox<3> box1(Coord<3>(0, 0, 0), Coord<3>(55, 47, 31));
        CoordBox<3> box2(Coord<3>(0, 0, 3), Coord<3>(55, 47,  7));
        TS_ASSERT_EQUALS(true, box1.intersects(box2));
    }

    void testIterator1D()
    {
        CoordBox<1> box(Coord<1>(10), Coord<1>(5));
        std::vector<Coord<1> > expected;
        expected << Coord<1>(10)
                 << Coord<1>(11)
                 << Coord<1>(12)
                 << Coord<1>(13)
                 << Coord<1>(14);

        std::vector<Coord<1> > actual;
        for (CoordBox<1>::Iterator i = box.begin();
             i != box.end();
             ++i) {
            actual << *i;
        }

        TS_ASSERT_EQUALS(expected, actual);
    }

    void testIterator2D()
    {
        CoordBox<2> box(Coord<2>(10, 15), Coord<2>(8, 9));
        std::vector<Coord<2> > expected;
        for (int y = 15; y < 24; ++y) {
            for (int x = 10; x < 18; ++x) {
                expected << Coord<2>(x, y);
            }
        }

        std::vector<Coord<2> > actual;
        for (CoordBox<2>::Iterator i = box.begin();
             i != box.end();
             ++i) {
            actual << *i;
        }

        TS_ASSERT_EQUALS(expected, actual);
    }

    void testIterator3D()
    {
        CoordBox<3> box(Coord<3>(10, 15, 20), Coord<3>(8, 9, 5));
        std::vector<Coord<3> > expected;
        for (int z = 20; z < 25; ++z) {
            for (int y = 15; y < 24; ++y) {
                for (int x = 10; x < 18; ++x) {
                    expected << Coord<3>(x, y, z);
                }
            }
        }

        std::vector<Coord<3> > actual;
        for (CoordBox<3>::Iterator i = box.begin();
             i != box.end();
             ++i) {
            actual << *i;
        }

        TS_ASSERT_EQUALS(expected, actual);
    }

    void testStreakIterator1D()
    {
        CoordBox<1> box(Coord<1>(123), Coord<1>(456));

        std::vector<Streak<1> > expected;
        std::vector<Streak<1> > actual;

        expected << Streak<1>(Coord<1>(123), 579);

        for (CoordBox<1>::StreakIterator i = box.beginStreak(); i != box.endStreak(); ++i) {
            actual << *i;
        }

        TS_ASSERT_EQUALS(expected, actual);
    }

    void testStreakIterator2D()
    {
        CoordBox<2> box(Coord<2>(73, 14), Coord<2>(61, 71));

        std::vector<Streak<2> > expected;
        std::vector<Streak<2> > actual;

        for (int y = 14; y < 85; ++y) {
            expected << Streak<2>(Coord<2>(73, y), 134);
        }

        for (CoordBox<2>::StreakIterator i = box.beginStreak(); i != box.endStreak(); ++i) {
            actual << *i;
        }

        TS_ASSERT_EQUALS(expected, actual);
    }

    void testStreakIterator3DFirst()
    {
        CoordBox<3> box(Coord<3>(10, 15, 20), Coord<3>(13, 11, 12));

        std::vector<Streak<3> > expected;
        std::vector<Streak<3> > actual;

        for (int z = 20; z < 32; ++z) {
            for (int y = 15; y < 26; ++y) {
                expected << Streak<3>(Coord<3>(10, y, z), 23);
            }
        }

        for (CoordBox<3>::StreakIterator i = box.beginStreak(); i != box.endStreak(); ++i) {
            actual << *i;
        }

        TS_ASSERT_EQUALS(expected, actual);
    }

    void testStreakIterator3DSecond()
    {
        CoordBox<3> box(Coord<3>(110, 115, 120), Coord<3>(13, 11, 12));

        std::vector<Streak<3> > expected;
        std::vector<Streak<3> > actual;

        for (int z = 120; z < 32; ++z) {
            for (int y = 115; y < 26; ++y) {
                expected << Streak<3>(Coord<3>(110, y, z), 123);
            }
        }

        for (CoordBox<3>::StreakIterator i = box.beginStreak(); i != box.endStreak(); ++i) {
            Coord<3> origin = i->origin;
            int endX = i->endX;
            actual << Streak<3>(origin, endX);
        }

        TS_ASSERT_EQUALS(expected, actual);
    }

    void testEmptyIteration()
    {
        CoordBox<1> box1(Coord<1>(4), Coord<1>());
        TS_ASSERT_EQUALS(box1.beginStreak(), box1.endStreak());

        CoordBox<2> box2(Coord<2>(1, 4), Coord<2>());
        TS_ASSERT_EQUALS(box2.beginStreak(), box2.endStreak());

        CoordBox<3> box3(Coord<3>(1, 2, 3), Coord<3>());
        TS_ASSERT_EQUALS(box3.beginStreak(), box3.endStreak());
    }

private:
    Coord<2> origin;
    int width;
    int height;
    CoordBox<2> rect;
};

}
