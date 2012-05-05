#include <cxxtest/TestSuite.h>
#include <boost/math/tools/precision.hpp>
#include <libgeodecomp/parallelization/chronometer.h>
#include <libgeodecomp/misc/coord.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class CoordTest : public CxxTest::TestSuite 
{
    Coord<2> *_c1;

public:
    bool includesCoord(SuperVector<Coord<2> > vec, int x, int y) {
        Coord<2> compare(x, y);
        for(SuperVector<Coord<2> >::iterator i = vec.begin(); i != vec.end(); i++) {
            if (*i == compare) {
                return true;
            }
        }
        return false;
    }
    
    void setUp()
    {
        _c1 = new Coord<2>(2, 3);
    }
    
    void tearDown()
    {
        delete _c1;
    }

    void testDefaultConstructor()
    {
        Coord<2> a;
        Coord<2> b(0, 0);
        TS_ASSERT_EQUALS(a, b);
    }
    
    void testEqual()
    {
        Coord<1> b1(47);
        Coord<1> b2(47);
        Coord<1> b3(11);
        TS_ASSERT_EQUALS(b1, b2);
        TS_ASSERT(!(b1 == b3));
        TS_ASSERT(!(b2 == b3));

        Coord<2> c2(2, 3);
        Coord<2> c3(2, 4);
        TS_ASSERT_EQUALS(*_c1, c2);
        TS_ASSERT(!(*_c1 == c3));


        Coord<3> d1(1, 2, 3);
        Coord<3> d2(1, 2, 0);
        Coord<3> d3(1, 2, 3);
        TS_ASSERT(!(d1 == d2));
        TS_ASSERT(!(d3 == d2));
        TS_ASSERT_EQUALS(d1, d3);
    }
    
    void testNotEqual()
    {
        Coord<2> c2(2, 3);
        TS_ASSERT(!(*_c1 != c2));

        Coord<2> c3(2, 4);
        TS_ASSERT(*_c1 != c3);

    }

    void testAdd()
    {
        Coord<2> base(5, 7);
        Coord<2> addend(13, 17);
        TS_ASSERT_EQUALS(Coord<2>(18, 24), base + addend);
    }

    void testScale()
    {
        Coord<2> base(5, 7);
        TS_ASSERT_EQUALS(Coord<2>(15, 21), base * 3);
    }

    void testBinaryMinus()
    {
        Coord<2> minuend(10, 2), result(-8, 1);
        TS_ASSERT_EQUALS(result, *_c1 - minuend);
    }


    void testUnaryMinus()
    {
        Coord<2> result(-2, -3);
        TS_ASSERT_EQUALS(result, -(*_c1));
    }

    void test3D()
    {
        Coord<3> c1(1, 2, 3);
        Coord<3> c2(5, 1, 9);
        Coord<3> c3(1, 2, 2);

        TS_ASSERT(c1 < c2);
        TS_ASSERT(c3 < c1);
        
        Coord<3> c4 = c1 + c3;
        TS_ASSERT_EQUALS(2, c4.x());
        TS_ASSERT_EQUALS(4, c4.y());
        TS_ASSERT_EQUALS(5, c4.z());
        
        Coord<3> c5 = c1;
        c5 += c3;
        TS_ASSERT_EQUALS(c4, c5);
        
        TS_ASSERT(c1 != c3);

        Coord<3> c6 = c5;
        c6 -= c1;
        TS_ASSERT_EQUALS(c6, c3);
        TS_ASSERT_EQUALS(c3, c5 - c1);

        TS_ASSERT_EQUALS(Coord<3>(-1, -2, -3), -c1);
        TS_ASSERT_EQUALS(Coord<3>(2, 4, 6), c1 * 2);
        TS_ASSERT_EQUALS(Coord<3>(0, 1, 1), c1 / 2);
    }

    void testLess()
    {
        TS_ASSERT(*_c1 < Coord<2>(3, 4));
        TS_ASSERT(*_c1 < Coord<2>(3, 1));
        TS_ASSERT(*_c1 < Coord<2>(2, 4));
        TS_ASSERT(!(*_c1 < Coord<2>(2, 3)));
        TS_ASSERT(!(*_c1 < Coord<2>(1, 6)));
        TS_ASSERT(!(*_c1 < Coord<2>(1, 1)));
    }

    void testToString()
    {
        TS_ASSERT_EQUALS(Coord<2>(1, 0).toString(), "(1, 0)");

    }       

    void testOperatorLessLess()
    {
        Coord<2> c;
        std::ostringstream temp;
        temp << c;
        TS_ASSERT_EQUALS(c.toString(), temp.str());
    }

    void testElement()
    {
        Coord<2> c1(5);
        TS_ASSERT_EQUALS(5, c1[0]);
        
        Coord<2> c2(4, 6);
        TS_ASSERT_EQUALS(4, c2[0]);
        TS_ASSERT_EQUALS(6, c2[1]);
        
        Coord<3> c3(1, 7, 9);
        TS_ASSERT_EQUALS(1, c3[0]);
        TS_ASSERT_EQUALS(7, c3[1]);
        TS_ASSERT_EQUALS(9, c3[2]);
    }

    void testToExtents()
    {
        boost::multi_array<int, 1> a1;
        boost::multi_array<int, 1> b1;
        boost::multi_array<int, 2> a2;
        boost::multi_array<int, 2> b2;
        boost::multi_array<int, 3> a3;
        boost::multi_array<int, 3> b3;

        a1.resize(boost::extents[4]);
        b1.resize(Coord<1>(4).toExtents());
        TS_ASSERT_EQUALS(a1.size(), b1.size());

        a2.resize(boost::extents[4][3]);
        b2.resize(Coord<2>(3, 4).toExtents());
        TS_ASSERT_EQUALS(a2.size(),    b2.size());
        TS_ASSERT_EQUALS(a2[0].size(), b2[0].size());

        a3.resize(boost::extents[6][5][4]);
        b3.resize(Coord<3>(4, 5, 6).toExtents());
        TS_ASSERT_EQUALS(a3.size(),       b3.size());
        TS_ASSERT_EQUALS(a3[0].size(),    b3[0].size());
        TS_ASSERT_EQUALS(a3[0][0].size(), b3[0][0].size());
    }

    void testCoordDiagonal()
    {
        TS_ASSERT_EQUALS(Coord<1>::diagonal(32), Coord<1>(32));
        TS_ASSERT_EQUALS(Coord<2>::diagonal(31), Coord<2>(31, 31));
        TS_ASSERT_EQUALS(Coord<3>::diagonal(30), Coord<3>(30, 30, 30));
    }

    void testIndexToCoord()
    {
        TS_ASSERT_EQUALS(Coord<1>(3),       IndexToCoord<1>()( 3, Coord<1>(10)));
        TS_ASSERT_EQUALS(Coord<2>(3, 2),    IndexToCoord<2>()(23, Coord<2>(10, 5)));
        TS_ASSERT_EQUALS(Coord<3>(3, 2, 1), IndexToCoord<3>()(73, Coord<3>(10, 5, 4)));
    }

    void testCoordToIndex()
    {
        TS_ASSERT_EQUALS(
            10,      
            CoordToIndex<1>()(Coord<1>(10),      Coord<1>(5)));
        TS_ASSERT_EQUALS(
            (5*7+4), 
            CoordToIndex<2>()(Coord<2>(4, 5),    Coord<2>(7, 8)));
        TS_ASSERT_EQUALS(
            (3*7*8+5*7+4), 
            CoordToIndex<3>()(Coord<3>(4, 5, 3), Coord<3>(7, 8, 4)));
    }

    void testProd()
    {
        TS_ASSERT_EQUALS(Coord<1>(2).prod(),       2);
        TS_ASSERT_EQUALS(Coord<2>(2, 3).prod(),    6);
        TS_ASSERT_EQUALS(Coord<2>(0, 3).prod(),    0);
        TS_ASSERT_EQUALS(Coord<3>(2, 3, 4).prod(), 24);
        TS_ASSERT_EQUALS(Coord<3>(2, 0, 4).prod(), 0);
    }

    void testMax()
    {
        TS_ASSERT_EQUALS(Coord<1>(1).max(Coord<1>(3)), Coord<1>(3));
        TS_ASSERT_EQUALS(Coord<1>(1).max(Coord<1>(0)), Coord<1>(1));

        TS_ASSERT_EQUALS(Coord<2>(3, 4).max(Coord<2>(5, 1)), Coord<2>(5, 4));

        TS_ASSERT_EQUALS(Coord<3>(10, 1, 14).max(Coord<3>( 9, 12,  9)), Coord<3>(10, 12, 14));
        TS_ASSERT_EQUALS(Coord<3>(10, 12, 1).max(Coord<3>( 9,  1, 14)), Coord<3>(10, 12, 14));
        TS_ASSERT_EQUALS(Coord<3>( 9, 11, 1).max(Coord<3>(10, 12, 14)), Coord<3>(10, 12, 14));
    }

    void testMin()
    {
        TS_ASSERT_EQUALS(Coord<1>(7).min(Coord<1>(3)), Coord<1>(3));
        TS_ASSERT_EQUALS(Coord<1>(1).min(Coord<1>(7)), Coord<1>(1));

        TS_ASSERT_EQUALS(Coord<2>(3, 4).min(Coord<2>(5, 1)), Coord<2>(3, 1));

        TS_ASSERT_EQUALS(Coord<3>(10, 1, 14).min(Coord<3>( 9, 12,  9)), Coord<3>(9, 1,  9));
        TS_ASSERT_EQUALS(Coord<3>(10, 12, 1).min(Coord<3>( 9,  1, 14)), Coord<3>(9, 1,  1));
        TS_ASSERT_EQUALS(Coord<3>( 9, 11, 1).min(Coord<3>(10, 12, 14)), Coord<3>(9, 11, 1));
    }

    void testSum()
    {
        TS_ASSERT_EQUALS(Coord<1>(6).sum(),        6);
        TS_ASSERT_EQUALS(Coord<2>(6, 4).sum(),    10);
        TS_ASSERT_EQUALS(Coord<3>(6, 4, 1).sum(), 11);
    }

    void testMult()
    {
        TS_ASSERT_EQUALS(0,  Coord<2>(1, 0)    * Coord<2>(0, 1));
        TS_ASSERT_EQUALS(7,  Coord<2>(3, 4)    * Coord<2>(1, 1));
        TS_ASSERT_EQUALS(26, Coord<3>(3, 4, 1) * Coord<3>(4, 3, 2));
    }

    // fixme: move this to performance tests
    // void testSpeed()
    // {
    //     long long tStart = Chronometer::timeUSec();

    //     const int SIZE = 100;
    //     Coord<3> coords[SIZE];
    //     for (int i = 0; i < SIZE; ++i) {
    //         coords[i] = Coord<3>(i, i + 1, i + 2);
    //     }

    //     for (int r = 0; r < 10000000; ++r) {
    //         Coord<3> buf;
    //         for (int i = 0; i < (SIZE - 1); ++i) {
    //             buf += coords[i];
    //             coords[i] = coords[i + 1];
    //         }
    //         for (int d = 0; d < 3; ++d) {
    //             if (buf[d] > 0x010000) {
    //                 buf[d] &= (0x010000 - 1);
    //             }
    //         }
    //         coords[SIZE - 1] = buf;
    //     }

    //     long long tEnd = Chronometer::timeUSec();
    //     double delta = (tEnd - tStart) / 1000000.0;
    //     std::cout << "delta: " << delta << "\n";
    // }
};

}
