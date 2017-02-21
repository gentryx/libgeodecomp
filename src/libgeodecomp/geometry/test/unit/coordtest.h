#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/config.hpp>
#endif

#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
#include <libgeodecomp/communication/boostserialization.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#endif

#include <sstream>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CoordTest : public CxxTest::TestSuite
{
public:
    bool includesCoord(std::vector<Coord<2> > vec, int x, int y) {
        Coord<2> compare(x, y);
        for(std::vector<Coord<2> >::iterator i = vec.begin(); i != vec.end(); i++) {
            if (*i == compare) {
                return true;
            }
        }
        return false;
    }

    void setUp()
    {
        c1 = new Coord<2>(2, 3);
    }

    void tearDown()
    {
        delete c1;
    }

    void testDefaultConstructor()
    {
        Coord<2> a;
        Coord<2> b(0, 0);
        TS_ASSERT_EQUALS(a, b);
    }

    void testConstructFromFixedCoord()
    {
        FixedCoord<1, 2, 3> f;
        TS_ASSERT_EQUALS(Coord<1>(1),       Coord<1>(f));
        TS_ASSERT_EQUALS(Coord<2>(1, 2),    Coord<2>(f));
        TS_ASSERT_EQUALS(Coord<3>(1, 2, 3), Coord<3>(f));
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
        TS_ASSERT_EQUALS(*c1, c2);
        TS_ASSERT(!(*c1 == c3));


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
        TS_ASSERT(!(*c1 != c2));

        Coord<2> c3(2, 4);
        TS_ASSERT(*c1 != c3);

    }

    void testAdd()
    {
        {
            Coord<1> base(7);
            Coord<1> addend(13);
            TS_ASSERT_EQUALS(Coord<1>(20), base + addend);
        }
        {
            Coord<2> base(5, 7);
            Coord<2> addend(13, 17);
            TS_ASSERT_EQUALS(Coord<2>(18, 24), base + addend);
        }
        {
            Coord<3> base(11, 13, 17);
            Coord<3> addend(19, 23, 29);
            TS_ASSERT_EQUALS(Coord<3>(30, 36, 46), base + addend);
        }
    }

    void testScale()
    {
        {
            Coord<1> base(5);
            TS_ASSERT_EQUALS(Coord<1>(15), base * 3);
            TS_ASSERT_EQUALS(Coord<1>(19), base * 3.9f);
            TS_ASSERT_EQUALS(Coord<1>(25), base * 5.01);
        }
        {
            Coord<2> base(5, 7);
            TS_ASSERT_EQUALS(Coord<2>(15, 21), base * 3);
            TS_ASSERT_EQUALS(Coord<2>(15, 21), base * 3.1f);
            TS_ASSERT_EQUALS(Coord<2>(22, 31), base * 4.5);
        }
        {
            Coord<3> base(5, 7, 6);
            TS_ASSERT_EQUALS(Coord<3>(15, 21, 18), base * 3);
            TS_ASSERT_EQUALS(Coord<3>(16, 22, 19), base * 3.2f);
            TS_ASSERT_EQUALS(Coord<3>(19, 27, 23), base * 3.9);
        }
    }

    void testScaleWithCoord()
    {
        {
            TS_ASSERT_EQUALS(Coord<1>(15), Coord<1>(5).scale(Coord<1>(3)));
        }
        {
            TS_ASSERT_EQUALS(Coord<2>(15, 8), Coord<2>(5, 4).scale(Coord<2>(3, 2)));
        }
        {
            TS_ASSERT_EQUALS(Coord<3>(15, 8, 42), Coord<3>(5, 4, 7).scale(Coord<3>(3, 2, 6)));
        }
    }

    void testBinaryMinus()
    {
        Coord<2> minuend(10, 2), result(-8, 1);
        TS_ASSERT_EQUALS(result, *c1 - minuend);
    }


    void testUnaryMinus()
    {
        Coord<2> result(-2, -3);
        TS_ASSERT_EQUALS(result, -(*c1));
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
        TS_ASSERT(*c1 < Coord<2>(3, 4));
        TS_ASSERT(*c1 < Coord<2>(3, 1));
        TS_ASSERT(*c1 < Coord<2>(2, 4));
        TS_ASSERT(!(*c1 < Coord<2>(2, 3)));
        TS_ASSERT(!(*c1 < Coord<2>(1, 6)));
        TS_ASSERT(!(*c1 < Coord<2>(1, 1)));
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

    void testCoordDiagonal()
    {
        TS_ASSERT_EQUALS(Coord<1>::diagonal(32), Coord<1>(32));
        TS_ASSERT_EQUALS(Coord<2>::diagonal(31), Coord<2>(31, 31));
        TS_ASSERT_EQUALS(Coord<3>::diagonal(30), Coord<3>(30, 30, 30));
    }

    void testIndexToCoord()
    {
        TS_ASSERT_EQUALS(Coord<1>(3),       Coord<1>(10      ).indexToCoord( 3));
        TS_ASSERT_EQUALS(Coord<2>(3, 2),    Coord<2>(10, 5   ).indexToCoord(23));
        TS_ASSERT_EQUALS(Coord<3>(3, 2, 1), Coord<3>(10, 5, 4).indexToCoord(73));
    }

    void testToIndex()
    {
        TS_ASSERT_EQUALS(
            std::size_t(10),
            Coord<1>(10     ).toIndex(Coord<1>(5)));
        TS_ASSERT_EQUALS(
            std::size_t(5*7+4),
            Coord<2>(4, 5   ).toIndex(Coord<2>(7, 8)));
        TS_ASSERT_EQUALS(
            std::size_t(3 * 7 * 8 + 5 * 7 + 4),
            Coord<3>(4, 5, 3).toIndex(Coord<3>(7, 8, 4)));
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
        TS_ASSERT_EQUALS((Coord<1>(1).max)(Coord<1>(3)), Coord<1>(3));
        TS_ASSERT_EQUALS((Coord<1>(1).max)(Coord<1>(0)), Coord<1>(1));

        TS_ASSERT_EQUALS((Coord<2>(3, 4).max)(Coord<2>(5, 1)), Coord<2>(5, 4));

        TS_ASSERT_EQUALS((Coord<3>(10, 1, 14).max)(Coord<3>( 9, 12,  9)), Coord<3>(10, 12, 14));
        TS_ASSERT_EQUALS((Coord<3>(10, 12, 1).max)(Coord<3>( 9,  1, 14)), Coord<3>(10, 12, 14));
        TS_ASSERT_EQUALS((Coord<3>( 9, 11, 1).max)(Coord<3>(10, 12, 14)), Coord<3>(10, 12, 14));
    }

    void testMin()
    {
        TS_ASSERT_EQUALS((Coord<1>(7).min)(Coord<1>(3)), Coord<1>(3));
        TS_ASSERT_EQUALS((Coord<1>(1).min)(Coord<1>(7)), Coord<1>(1));

        TS_ASSERT_EQUALS((Coord<2>(3, 4).min)(Coord<2>(5, 1)), Coord<2>(3, 1));

        TS_ASSERT_EQUALS((Coord<3>(10, 1, 14).min)(Coord<3>( 9, 12,  9)), Coord<3>(9, 1,  9));
        TS_ASSERT_EQUALS((Coord<3>(10, 12, 1).min)(Coord<3>( 9,  1, 14)), Coord<3>(9, 1,  1));
        TS_ASSERT_EQUALS((Coord<3>( 9, 11, 1).min)(Coord<3>(10, 12, 14)), Coord<3>(9, 11, 1));
    }

    void testMaxElement()
    {
        TS_ASSERT_EQUALS(Coord<1>(5).maxElement(), 5);

        TS_ASSERT_EQUALS(Coord<2>(6, 1).maxElement(), 6);
        TS_ASSERT_EQUALS(Coord<2>(5, 7).maxElement(), 7);

        TS_ASSERT_EQUALS(Coord<3>( 8,  1, 0).maxElement(), 8);
        TS_ASSERT_EQUALS(Coord<3>( 5,  9, 0).maxElement(), 9);
        TS_ASSERT_EQUALS(Coord<3>(-5, -7, 0).maxElement(), 0);
        TS_ASSERT_EQUALS(Coord<3>(-7, -5, 0).maxElement(), 0);
    }

    void testMinElement()
    {
        TS_ASSERT_EQUALS(Coord<1>(5).minElement(), 5);

        TS_ASSERT_EQUALS(Coord<2>(6, 1).minElement(), 1);
        TS_ASSERT_EQUALS(Coord<2>(5, 7).minElement(), 5);

        TS_ASSERT_EQUALS(Coord<3>( 8, 10, 100).minElement(), 8);
        TS_ASSERT_EQUALS(Coord<3>( 5, 90, 100).minElement(), 5);
        TS_ASSERT_EQUALS(Coord<3>( 5,  7,   1).minElement(), 1);
        TS_ASSERT_EQUALS(Coord<3>( 7,  5,   0).minElement(), 0);
    }

    void testAbs()
    {
        TS_ASSERT_EQUALS(Coord<1>(10), Coord<1>( 10).abs());
        TS_ASSERT_EQUALS(Coord<1>(11), Coord<1>(-11).abs());

        TS_ASSERT_EQUALS(Coord<2>(20, 21), Coord<2>( 20,  21).abs());
        TS_ASSERT_EQUALS(Coord<2>(22, 23), Coord<2>( 22, -23).abs());
        TS_ASSERT_EQUALS(Coord<2>(24, 25), Coord<2>(-24,  25).abs());
        TS_ASSERT_EQUALS(Coord<2>(26, 27), Coord<2>(-26, -27).abs());

        TS_ASSERT_EQUALS(Coord<3>(30, 31, 32), Coord<3>( 30,  31,  32).abs());
        TS_ASSERT_EQUALS(Coord<3>(33, 34, 35), Coord<3>( 33,  34, -35).abs());
        TS_ASSERT_EQUALS(Coord<3>(36, 37, 38), Coord<3>( 36, -37,  38).abs());
        TS_ASSERT_EQUALS(Coord<3>(39, 40, 41), Coord<3>( 39, -40, -41).abs());
        TS_ASSERT_EQUALS(Coord<3>(42, 43, 44), Coord<3>(-42,  43,  44).abs());
        TS_ASSERT_EQUALS(Coord<3>(45, 46, 47), Coord<3>(-45,  46, -47).abs());
        TS_ASSERT_EQUALS(Coord<3>(48, 49, 50), Coord<3>(-48, -49,  50).abs());
        TS_ASSERT_EQUALS(Coord<3>(51, 52, 53), Coord<3>(-51, -52, -53).abs());
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

    void testSerializationWithHPX()
    {
#ifdef LIBGEODECOMP_WITH_HPX
        Coord<2> c(47,11);
        Coord<2> d(1, 2);

        std::vector<char> buf;
        {
            hpx::serialization::output_archive archive(buf);
            archive << c;
        }
        {
            hpx::serialization::input_archive archive(buf);
            archive >> d;
        }
        TS_ASSERT_EQUALS(c, d);
#endif
    }

    void testSerializationWithBoostSerialization()
    {

#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
        Coord<2> c(47,11);
        Coord<2> d(1, 2);

        std::stringstream buf;
        {
            boost::archive::text_oarchive archive(buf);
            archive << c;
        }
        {
            boost::archive::text_iarchive archive(buf);
            archive >> d;
        }
        TS_ASSERT_EQUALS(c, d);
#endif
    }

    void testDim()
    {
        TS_ASSERT_EQUALS(1, Coord<1>::DIM);
        TS_ASSERT_EQUALS(2, Coord<2>::DIM);
        TS_ASSERT_EQUALS(3, Coord<3>::DIM);
    }

    void testConversion()
    {
        TS_ASSERT_EQUALS(Coord<1>(2), Coord<1>(FloatCoord<1>(2.0)));
        TS_ASSERT_EQUALS(Coord<1>(2), Coord<1>(FloatCoord<1>(2.1)));
        TS_ASSERT_EQUALS(Coord<1>(3), Coord<1>(FloatCoord<1>(3.9)));

        TS_ASSERT_EQUALS(Coord<2>(2, 5), Coord<2>(FloatCoord<2>(2.0, 5.0)));
        TS_ASSERT_EQUALS(Coord<2>(2, 1), Coord<2>(FloatCoord<2>(2.1, 1.1)));
        TS_ASSERT_EQUALS(Coord<2>(3, 4), Coord<2>(FloatCoord<2>(3.9, 4.4)));

        TS_ASSERT_EQUALS(Coord<3>(2, 5, 9), Coord<3>(FloatCoord<3>(2.0, 5.0, 9.0)));
        TS_ASSERT_EQUALS(Coord<3>(2, 1, 8), Coord<3>(FloatCoord<3>(2.1, 1.1, 8.1)));
        TS_ASSERT_EQUALS(Coord<3>(3, 4, 7), Coord<3>(FloatCoord<3>(3.9, 4.4, 7.9)));
    }

private:
    Coord<2> *c1;
};

}
