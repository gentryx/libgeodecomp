#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>
#include "../../coordbox.h"

using namespace LibGeoDecomp; 
using namespace boost::assign;

namespace LibGeoDecomp {

class CoordBoxTest : public CxxTest::TestSuite
{
private:
    Coord<2> origin;
    unsigned width;
    unsigned height;
    CoordBox<2> _rect;

public:

    void setUp()
    {
        origin = Coord<2>(3,4);
        width = 4;
        height = 3;
        _rect = CoordBox<2>(origin, Coord<2>(width, height));
    }

    void testConstructor1()
    {
        TS_ASSERT_EQUALS(origin, _rect.origin);
        TS_ASSERT_EQUALS(width,  _rect.dimensions.x());
        TS_ASSERT_EQUALS(height, _rect.dimensions.y());
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
        TS_ASSERT(_rect.inBounds(Coord<2>(3, 4)));
        TS_ASSERT(_rect.inBounds(Coord<2>(6, 4)));
        TS_ASSERT(_rect.inBounds(Coord<2>(3, 6)));
        TS_ASSERT(_rect.inBounds(Coord<2>(6, 6)));
        TS_ASSERT(_rect.inBounds(Coord<2>(4, 4)));

        TS_ASSERT(!_rect.inBounds(Coord<2>(0, 0)));
        TS_ASSERT(!_rect.inBounds(Coord<2>(100, 0)));
        TS_ASSERT(!_rect.inBounds(Coord<2>(0, 100)));
        TS_ASSERT(!_rect.inBounds(Coord<2>(5, 8)));
        TS_ASSERT(!_rect.inBounds(Coord<2>(8, 5)));

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

    void testSequence2D()
    {
        // row major order is guaranteed
        CoordBox<2> rect(Coord<2>(-2, 3), Coord<2>(5, 7)); 
        CoordBoxSequence<2> seq = rect.sequence();
        for (int y = rect.origin.y();
                y < rect.origin.y() + (int)rect.dimensions.y();
                y ++) {
            for (int x = rect.origin.x();
                    x < rect.origin.x() + (int)rect.dimensions.x();
                    x ++) {
                TS_ASSERT_EQUALS(true, seq.hasNext());
                Coord<2> res = seq.next();
                if (res != Coord<2>(x, y))
                    std::cout << "expected: " << Coord<2>(x, y) << "\n"
                              << "actual: " << res << "\n\n";
                TS_ASSERT_EQUALS(Coord<2>(x, y), res);
            }
        }
        TS_ASSERT_EQUALS(false, seq.hasNext());
        TS_ASSERT_THROWS(seq.next(), std::out_of_range);

        CoordBox<2> rect2(Coord<2>(1, 0), Coord<2>(0, 1)); 
        CoordBoxSequence<2> seq2 = rect2.sequence();
        TS_ASSERT_EQUALS(false, seq2.hasNext());
    }

    void testSequence3D()
    {
        // plane/row major order is guaranteed
        CoordBox<3> rect(Coord<3>(-2, 3, 4), Coord<3>(5, 7, 8)); 
        CoordBoxSequence<3> seq = rect.sequence();

        for (int z = rect.origin.z();
                z < rect.origin.z() + (int)rect.dimensions.z();
                z ++) {
            for (int y = rect.origin.y();
                 y < rect.origin.y() + (int)rect.dimensions.y();
                 y ++) {
                for (int x = rect.origin.x();
                     x < rect.origin.x() + (int)rect.dimensions.x();
                     x ++) {
                    TS_ASSERT_EQUALS(true, seq.hasNext());
                    Coord<3> next = seq.next();
                    TS_ASSERT_EQUALS(Coord<3>(x, y, z), next);
                }
            }
        }

        TS_ASSERT_EQUALS(false, seq.hasNext());
        TS_ASSERT_THROWS(seq.next(), std::out_of_range);

        CoordBox<3> rect2(Coord<3>(1, 1, 1), Coord<3>(0, 1, 1)); 
        CoordBoxSequence<3> seq2 = rect2.sequence();
        TS_ASSERT_EQUALS(false, seq2.hasNext());
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
        SuperVector<Coord<1> > expected;
        expected += Coord<1>(10), Coord<1>(11), Coord<1>(12), Coord<1>(13), Coord<1>(14);

        SuperVector<Coord<1> > actual;
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
        SuperVector<Coord<2> > expected;
        for (int y = 15; y < 24; ++y) {
            for (int x = 10; x < 18; ++x) {
                expected << Coord<2>(x, y);
            }
        }

        SuperVector<Coord<2> > actual;
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
        SuperVector<Coord<3> > expected;
        for (int z = 20; z < 25; ++z) {
            for (int y = 15; y < 24; ++y) {
                for (int x = 10; x < 18; ++x) {
                    expected << Coord<3>(x, y, z);
                }
            }
        }

        SuperVector<Coord<3> > actual;
        for (CoordBox<3>::Iterator i = box.begin();
             i != box.end();
             ++i) {
            actual << *i;
        }
        
        TS_ASSERT_EQUALS(expected, actual);
    }
};

};
