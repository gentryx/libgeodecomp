#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/misc/testhelper.h>

#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class FloatCoordTest : public CxxTest::TestSuite
{
public:
    void testDefaultConstructors()
    {
        FloatCoord<2> a;
        FloatCoord<2> b(0, 0);
        TS_ASSERT_EQUALS(a, b);

        Coord<3> c(1, 2, 3);
        FloatCoord<3> d(c);
        TS_ASSERT_EQUALS(d, FloatCoord<3>(1, 2, 3));
    }

    void testLength()
    {
        TS_ASSERT_EQUALS_DOUBLE(1.2, FloatCoord<1>(-1.2).length());
        TS_ASSERT_EQUALS_DOUBLE(5.0, FloatCoord<2>(3, -4).length());
        TS_ASSERT_EQUALS_DOUBLE(7.0, FloatCoord<3>(2, 3, 6).length());
    }

    void testAbs()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(1.1), FloatCoord<1>( 1.1).abs());
        TS_ASSERT_EQUALS(FloatCoord<1>(2.1), FloatCoord<1>(-2.1).abs());

        TS_ASSERT_EQUALS(FloatCoord<2>(3.1, 4.1), FloatCoord<2>( 3.1,  4.1).abs());
        TS_ASSERT_EQUALS(FloatCoord<2>(5.1, 6.1), FloatCoord<2>(-5.1, -6.1).abs());

        TS_ASSERT_EQUALS(FloatCoord<3>(7.1, 8.1, 9.1), FloatCoord<3>( 7.1,  8.1,  9.1).abs());
        TS_ASSERT_EQUALS(FloatCoord<3>(7.2, 8.2, 9.2), FloatCoord<3>(-7.2, -8.2, -9.2).abs());
    }

    void testSum()
    {
        TS_ASSERT_EQUALS_DOUBLE(2.0, FloatCoord<1>(2.0).sum());
        TS_ASSERT_EQUALS_DOUBLE(3.0, FloatCoord<2>(1.5, 1.5).sum());
        TS_ASSERT_EQUALS_DOUBLE(6.0, FloatCoord<3>(1, 3.3, 1.7).sum());
    }

    void testScale()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(7.5            ),
                         FloatCoord<1>(2.5            ).scale(FloatCoord<1>(3.0          )));
        TS_ASSERT_EQUALS(FloatCoord<2>(7.5, 20.0      ),
                         FloatCoord<2>(2.5,  4.0      ).scale(FloatCoord<2>(3.0, 5.0     )));
        TS_ASSERT_EQUALS(FloatCoord<3>(7.5, 20.0, 43.5),
                         FloatCoord<3>(2.5,  4.0,  6.0).scale(FloatCoord<3>(3.0, 5.0, 7.25)));

        TS_ASSERT_EQUALS(FloatCoord<1>(7.5             ),
                         FloatCoord<1>(2.5             ).scale(Coord<1>(3      )));
        TS_ASSERT_EQUALS(FloatCoord<2>(7.5, 20.0       ),
                         FloatCoord<2>(2.5,  4.0       ).scale(Coord<2>(3, 5   )));
        TS_ASSERT_EQUALS(FloatCoord<3>(7.5, 20.0, 43.75),
                         FloatCoord<3>(2.5,  4.0,  6.25).scale(Coord<3>(3, 5, 7)));
    }

    void testProd()
    {
        TS_ASSERT_EQUALS_DOUBLE( 2.0, FloatCoord<1>(2.0).prod());
        TS_ASSERT_EQUALS_DOUBLE( 6.0, FloatCoord<2>(2.0, 3.0).prod());
        TS_ASSERT_EQUALS_DOUBLE(30.0, FloatCoord<3>(2.0, 3.0, 5.0).prod());
    }

    void testMax()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(2.0),
                         (FloatCoord<1>(2.0).max)(FloatCoord<1>(0.0)));
        TS_ASSERT_EQUALS(FloatCoord<1>(2.0),
                         (FloatCoord<1>(0.0).max)(FloatCoord<1>(2.0)));

        TS_ASSERT_EQUALS(FloatCoord<2>(2.0, 3.0),
                         (FloatCoord<2>(2.0, 0.0).max)(FloatCoord<2>(0.0, 3.0)));
        TS_ASSERT_EQUALS(FloatCoord<2>(2.0, 3.0),
                         (FloatCoord<2>(0.0, 3.0).max)(FloatCoord<2>(2.0, 0.0)));

        TS_ASSERT_EQUALS(FloatCoord<3>(2.0, 3.0, 4.0),
                         (FloatCoord<3>(0.0, 3.0, 0.0).max)(FloatCoord<3>(2.0, 0.0, 4.0)));
        TS_ASSERT_EQUALS(FloatCoord<3>(2.0, 3.0, 4.0),
                         (FloatCoord<3>(2.0, 0.0, 4.0).max)(FloatCoord<3>(0.0, 3.0, 0.0)));
    }

    void testMin()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(-2.0),
                         (FloatCoord<1>(-2.0).min)(FloatCoord<1>( 0.0)));
        TS_ASSERT_EQUALS(FloatCoord<1>(-2.0),
                         (FloatCoord<1>( 0.0).min)(FloatCoord<1>(-2.0)));

        TS_ASSERT_EQUALS(FloatCoord<2>(-2.0, -3.0),
                         (FloatCoord<2>(-2.0,  0.0).min)(FloatCoord<2>( 0.0, -3.0)));
        TS_ASSERT_EQUALS(FloatCoord<2>(-2.0, -3.0),
                         (FloatCoord<2>( 0.0, -3.0).min)(FloatCoord<2>(-2.0,  0.0)));

        TS_ASSERT_EQUALS(FloatCoord<3>(-2.0, -3.0, -4.0),
                         (FloatCoord<3>( 0.0, -3.0,  0.0).min)(FloatCoord<3>(-2.0,  0.0, -4.0)));
        TS_ASSERT_EQUALS(FloatCoord<3>(-2.0, -3.0, -4.0),
                         (FloatCoord<3>(-2.0,  0.0, -4.0).min)(FloatCoord<3>( 0.0, -3.0,  0.0)));
    }

    void testMaxElement()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(5.5).maxElement(), 5.5);

        TS_ASSERT_EQUALS(FloatCoord<2>(6.1, 1.1).maxElement(), 6.1);
        TS_ASSERT_EQUALS(FloatCoord<2>(5.1, 7.1).maxElement(), 7.1);

        TS_ASSERT_EQUALS(FloatCoord<3>( 8.3,  1.4, 0.2).maxElement(), 8.3);
        TS_ASSERT_EQUALS(FloatCoord<3>( 5.3,  9.4, 0.2).maxElement(), 9.4);
        TS_ASSERT_EQUALS(FloatCoord<3>(-5.3, -7.4, 0.2).maxElement(), 0.2);
        TS_ASSERT_EQUALS(FloatCoord<3>(-7.3, -5.4, 0.2).maxElement(), 0.2);
    }

    void testMinElement()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(5.5).minElement(), 5.5);

        TS_ASSERT_EQUALS(FloatCoord<2>(6.5, 1.5).minElement(), 1.5);
        TS_ASSERT_EQUALS(FloatCoord<2>(5.5, 7.5).minElement(), 5.5);

        TS_ASSERT_EQUALS(FloatCoord<3>( 8.5, 10.5, 100.1).minElement(), 8.5);
        TS_ASSERT_EQUALS(FloatCoord<3>( 5.5, 90.5, 100.9).minElement(), 5.5);
        TS_ASSERT_EQUALS(FloatCoord<3>( 5.5,  7.5,   1.4).minElement(), 1.4);
        TS_ASSERT_EQUALS(FloatCoord<3>( 9.5,  7.4,   9.4).minElement(), 7.4);
        TS_ASSERT_EQUALS(FloatCoord<3>( 7.5,  5.5,   0.4).minElement(), 0.4);
    }

    void testOperatorPlus()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(3),
                         FloatCoord<1>(1) + FloatCoord<1>(2));
        TS_ASSERT_EQUALS(FloatCoord<2>(4, 6),
                         FloatCoord<2>(1, 2) + FloatCoord<2>(3, 4));
        TS_ASSERT_EQUALS(FloatCoord<3>(5, 7, 9),
                         FloatCoord<3>(1, 2, 3) + FloatCoord<3>(4, 5, 6));

        {
            FloatCoord<1> a(1);
            a += FloatCoord<1>(4);
            TS_ASSERT_EQUALS(FloatCoord<1>(5), a);
        }
        {
            FloatCoord<2> a(1, 2);
            a += FloatCoord<2>(6, 3);
            TS_ASSERT_EQUALS(FloatCoord<2>(7, 5), a);
        }
        {
            FloatCoord<3> a(1, 7, 2);
            a += FloatCoord<3>(6, 1, 7);
            TS_ASSERT_EQUALS(FloatCoord<3>(7, 8, 9), a);
        }
    }

    void testOperatorPlusWithOtherCoord()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(3),
                         FloatCoord<1>(1) + Coord<1>(2));
        TS_ASSERT_EQUALS(FloatCoord<2>(4, 6),
                         FloatCoord<2>(1, 2) + Coord<2>(3, 4));
        TS_ASSERT_EQUALS(FloatCoord<3>(5, 7, 9),
                         FloatCoord<3>(1, 2, 3) + Coord<3>(4, 5, 6));

        {
            FloatCoord<1> a(1);
            a += Coord<1>(4);
            TS_ASSERT_EQUALS(FloatCoord<1>(5), a);
        }
        {
            FloatCoord<2> a(1, 2);
            a += Coord<2>(6, 3);
            TS_ASSERT_EQUALS(FloatCoord<2>(7, 5), a);
        }
        {
            FloatCoord<3> a(1, 7, 2);
            a += Coord<3>(6, 1, 7);
            TS_ASSERT_EQUALS(FloatCoord<3>(7, 8, 9), a);
        }
    }

    void testOperatorMinus()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(-1),
                         FloatCoord<1>(1) - FloatCoord<1>(2));
        TS_ASSERT_EQUALS(FloatCoord<2>(-2, -3),
                         FloatCoord<2>(1, 2) - FloatCoord<2>(3, 5));
        TS_ASSERT_EQUALS(FloatCoord<3>(-7, -3, -4),
                         FloatCoord<3>(1, 2, 3) - FloatCoord<3>(8, 5, 7));


        {
            FloatCoord<1> a(5);
            a -= FloatCoord<1>(4);
            TS_ASSERT_EQUALS(FloatCoord<1>(1), a);
        }
        {
            FloatCoord<2> a(7, 5);
            a -= FloatCoord<2>(6, 3);
            TS_ASSERT_EQUALS(FloatCoord<2>(1, 2), a);
        }
        {
            FloatCoord<3> a(7, 8, 9);
            a -= FloatCoord<3>(6, 1, 7);
            TS_ASSERT_EQUALS(FloatCoord<3>(1, 7, 2), a);
        }
    }

    void testOperatorMinusWithOtherCoord()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(-1),
                         FloatCoord<1>(1) - Coord<1>(2));
        TS_ASSERT_EQUALS(FloatCoord<2>(-2, -3),
                         FloatCoord<2>(1, 2) - Coord<2>(3, 5));
        TS_ASSERT_EQUALS(FloatCoord<3>(-7, -3, -4),
                         FloatCoord<3>(1, 2, 3) - Coord<3>(8, 5, 7));


        {
            FloatCoord<1> a(5);
            a -= Coord<1>(4);
            TS_ASSERT_EQUALS(FloatCoord<1>(1), a);
        }
        {
            FloatCoord<2> a(7, 5);
            a -= Coord<2>(6, 3);
            TS_ASSERT_EQUALS(FloatCoord<2>(1, 2), a);
        }
        {
            FloatCoord<3> a(7, 8, 9);
            a -= Coord<3>(6, 1, 7);
            TS_ASSERT_EQUALS(FloatCoord<3>(1, 7, 2), a);
        }
    }

    void testOperatorUnaryMinus()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(-1), -FloatCoord<1>( 1));
        TS_ASSERT_EQUALS(FloatCoord<1>( 6), -FloatCoord<1>(-6));

        TS_ASSERT_EQUALS(FloatCoord<2>(-1,  5), -FloatCoord<2>( 1, -5));
        TS_ASSERT_EQUALS(FloatCoord<2>( 6, -8), -FloatCoord<2>(-6,  8));

        TS_ASSERT_EQUALS(FloatCoord<3>(-1,  5, 6), -FloatCoord<3>( 1, -5, -6));
        TS_ASSERT_EQUALS(FloatCoord<3>( 6, -8, 9), -FloatCoord<3>(-6,  8, -9));
    }

    void testOperatorMultiply()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(4),
                         FloatCoord<1>(1) * 4);
        TS_ASSERT_EQUALS(FloatCoord<2>(4, 8),
                         FloatCoord<2>(1, 2) * 4);
        TS_ASSERT_EQUALS(FloatCoord<3>(4, 8, 12),
                         FloatCoord<3>(1, 2, 3) * 4);

        TS_ASSERT_EQUALS(FloatCoord<1>(4),
                         FloatCoord<1>(1) * 4);
        TS_ASSERT_EQUALS(FloatCoord<2>(4, 8),
                         FloatCoord<2>(1, 2) * 4);
        TS_ASSERT_EQUALS(FloatCoord<3>(4, 8, 12),
                         FloatCoord<3>(1, 2, 3) * 4);
        {
            FloatCoord<1> f(3);
            f *= 5;
            TS_ASSERT_EQUALS(FloatCoord<1>(15), f);
        }
        {
            FloatCoord<2> f(3, 5);
            f *= 5;
            TS_ASSERT_EQUALS(FloatCoord<2>(15, 25), f);
        }
        {
            FloatCoord<3> f(3, 5, 2);
            f *= 5;
            TS_ASSERT_EQUALS(FloatCoord<3>(15, 25, 10), f);
        }
    }

    void testDotProduct()
    {
        TS_ASSERT_EQUALS(
            6.0,
            FloatCoord<1>(2.0) *
            FloatCoord<1>(3.0));
        TS_ASSERT_EQUALS(
            -27.5,
            FloatCoord<2>(2.5, -7.0) *
            FloatCoord<2>(3.0,  5.0));
        TS_ASSERT_EQUALS(
            -7.5,
            FloatCoord<3>(2.5, -7.0, 10.0) *
            FloatCoord<3>(3.0,  5.0,  2.0));
    }

    void testDotProductWithOtherCoord()
    {
        TS_ASSERT_EQUALS(
            6.0,
            FloatCoord<1>(2.0) *
            Coord<1>(3));
        TS_ASSERT_EQUALS(
            -27.5,
            FloatCoord<2>(2.5, -7.0) *
            Coord<2>(3, 5));
        TS_ASSERT_EQUALS(
            -7.5,
            FloatCoord<3>(2.5, -7.0, 10.0) *
            Coord<3>(3, 5, 2));
    }

    void testOperatorEquals()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(4), FloatCoord<1>(4));
        TS_ASSERT(!(FloatCoord<1>(4) == FloatCoord<1>(5)));
        TS_ASSERT( (FloatCoord<1>(4) != FloatCoord<1>(5)));

        TS_ASSERT_EQUALS(FloatCoord<2>(4, 1), FloatCoord<2>(4, 1));
        TS_ASSERT(!(FloatCoord<2>(4, 4) == FloatCoord<2>(5, 4)));
        TS_ASSERT( (FloatCoord<2>(4, 4) != FloatCoord<2>(5, 4)));

        TS_ASSERT_EQUALS(FloatCoord<3>(4, 1, 5), FloatCoord<3>(4, 1, 5));
        TS_ASSERT(!(FloatCoord<3>(4, 4, 4) == FloatCoord<3>(4, 4, 3)));
        TS_ASSERT( (FloatCoord<3>(4, 4, 4) != FloatCoord<3>(4, 4, 3)));

        TS_ASSERT(FloatCoord<1>(1) != FloatCoord<1>(2));
        TS_ASSERT(!(FloatCoord<1>(1) != FloatCoord<1>(1)));
        TS_ASSERT( (FloatCoord<1>(1) == FloatCoord<1>(1)));

        TS_ASSERT(FloatCoord<2>(1, 2) != FloatCoord<2>(1, 9));
        TS_ASSERT(!(FloatCoord<2>(1, 2) != FloatCoord<2>(1, 2)));
        TS_ASSERT( (FloatCoord<2>(1, 2) == FloatCoord<2>(1, 2)));

        TS_ASSERT(FloatCoord<3>(1, 2, 3) != FloatCoord<3>(1, 2, 4));
        TS_ASSERT(!(FloatCoord<3>(1, 2, 3) != FloatCoord<3>(1, 2, 3)));
        TS_ASSERT( (FloatCoord<3>(1, 2, 3) == FloatCoord<3>(1, 2, 3)));
    }

    void testOperatorLess()
    {
        TS_ASSERT(  FloatCoord<1>(4.0) < FloatCoord<1>(4.1));
        TS_ASSERT(!(FloatCoord<1>(4.2) < FloatCoord<1>(4.1)));

        TS_ASSERT(  FloatCoord<2>(3.9, 4.0) < FloatCoord<2>(4.0, 4.0));
        TS_ASSERT(  FloatCoord<2>(4.0, 3.9) < FloatCoord<2>(4.0, 4.0));
        TS_ASSERT(!(FloatCoord<2>(4.0, 4.0) < FloatCoord<2>(4.0, 4.0)));

        TS_ASSERT(  FloatCoord<3>(3.9, 4.0, 4.0) < FloatCoord<3>(4.0, 4.0, 4.0));
        TS_ASSERT(  FloatCoord<3>(4.0, 3.9, 4.0) < FloatCoord<3>(4.0, 4.0, 4.0));
        TS_ASSERT(  FloatCoord<3>(4.0, 4.0, 3.9) < FloatCoord<3>(4.0, 4.0, 4.0));
        TS_ASSERT(!(FloatCoord<3>(4.0, 4.0, 4.0) < FloatCoord<3>(4.0, 4.0, 4.0)));

        TS_ASSERT(!(FloatCoord<3>(4.1, 4.0, 4.0) < FloatCoord<3>(4.0, 4.0, 4.0)));
        TS_ASSERT(!(FloatCoord<3>(4.0, 4.1, 4.0) < FloatCoord<3>(4.0, 4.0, 4.0)));
        TS_ASSERT(!(FloatCoord<3>(4.0, 4.0, 4.1) < FloatCoord<3>(4.0, 4.0, 4.0)));
    }

    void testOperatorEqualsWithOtherCoordType()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>(4), Coord<1>(4));
        TS_ASSERT(!(FloatCoord<1>(5.1) == Coord<1>(5)));

        TS_ASSERT_EQUALS(FloatCoord<2>(4, 1), Coord<2>(4, 1));
        TS_ASSERT(!(FloatCoord<2>(5.1, 4) == Coord<2>(5, 4)));

        TS_ASSERT_EQUALS(FloatCoord<3>(4, 1, 5), Coord<3>(4, 1, 5));
        TS_ASSERT(!(FloatCoord<3>(4, 4, 3.1) == Coord<3>(4, 4, 3)));

        TS_ASSERT(FloatCoord<1>(2.1) != Coord<1>(2));
        TS_ASSERT(!(FloatCoord<1>(1) != Coord<1>(1)));

        TS_ASSERT(FloatCoord<2>(1, 9.1) != Coord<2>(1, 9));
        TS_ASSERT(!(FloatCoord<2>(1, 2) != Coord<2>(1, 2)));

        TS_ASSERT(FloatCoord<3>(1, 2, 4.1) != Coord<3>(1, 2, 4));
        TS_ASSERT(!(FloatCoord<3>(1, 2, 3) != Coord<3>(1, 2, 3)));
    }

    void testOperatorDivide()
    {
        FloatCoord<1> c1(3.5);
        FloatCoord<2> c2(5.5, 7.0);
        FloatCoord<3> c3(3.5, 1.0, 1.25);

        FloatCoord<1> d1(14.0);
        FloatCoord<2> d2(16.5, 21.0);
        FloatCoord<3> d3(14.0, 4.0, 5.0);

        TS_ASSERT_EQUALS(c1, d1 / 4);
        TS_ASSERT_EQUALS(c2, d2 / 3);
        TS_ASSERT_EQUALS(c3, d3 / 4);

        d1 /= 4;
        d2 /= 3;
        d3 /= 4;

        TS_ASSERT_EQUALS(c1, d1);
        TS_ASSERT_EQUALS(c2, d2);
        TS_ASSERT_EQUALS(c3, d3);

        TS_ASSERT_EQUALS(c1 / FloatCoord<1>(0.5),            FloatCoord<1>(7.0));
        TS_ASSERT_EQUALS(c2 / FloatCoord<2>(5.5, 2.0),       FloatCoord<2>(1.0, 3.5));
        TS_ASSERT_EQUALS(c3 / FloatCoord<3>(0.5, 0.5, 0.25), FloatCoord<3>(7.0, 2.0, 5.0));

        TS_ASSERT_EQUALS(c1 / Coord<1>(2),       FloatCoord<1>(1.75));
        TS_ASSERT_EQUALS(c2 / Coord<2>(5, 2),    FloatCoord<2>(1.1,  3.5));
        TS_ASSERT_EQUALS(c3 / Coord<3>(2, 8, 5), FloatCoord<3>(1.75, 0.125, 0.25));
    }

    void testDiagonal()
    {
        TS_ASSERT_EQUALS(FloatCoord<1>::diagonal(12.34), FloatCoord<1>(12.34));
        TS_ASSERT_EQUALS(FloatCoord<2>::diagonal(23.45), FloatCoord<2>(23.45, 23.45));
        TS_ASSERT_EQUALS(FloatCoord<3>::diagonal(34.56), FloatCoord<3>(34.56, 34.56, 34.56));
    }

    void testDominates1D()
    {
        FloatCoord<1> f1(10.1);
        FloatCoord<1> f2(11.1);
        Coord<1> c1(10);
        Coord<1> c2(11);

        TS_ASSERT( f1.dominates(f1));
        TS_ASSERT( f1.dominates(f2));
        TS_ASSERT(!f1.dominates(c1));
        TS_ASSERT( f1.dominates(c2));

        TS_ASSERT(!f2.dominates(f1));
        TS_ASSERT( f2.dominates(f2));
        TS_ASSERT(!f2.dominates(c1));
        TS_ASSERT(!f2.dominates(c2));
    }

    void testDominates2D()
    {
        FloatCoord<2> fC(10.1, 10.5);

        FloatCoord<2> fR(11.1, 10.5);
        FloatCoord<2> fT(10.1, 10.1);
        FloatCoord<2> fL(10.0, 10.5);
        FloatCoord<2> fB(10.1, 11.5);

        Coord<2> c1(10, 10);
        Coord<2> c2(11, 11);

        TS_ASSERT( fC.dominates(fC));

        TS_ASSERT( fC.dominates(fR));
        TS_ASSERT( fC.dominates(fB));
        TS_ASSERT(!fC.dominates(fL));
        TS_ASSERT(!fC.dominates(fT));

        TS_ASSERT(!fC.dominates(c1));
        TS_ASSERT( fC.dominates(c2));
    }

    void testDominates3D()
    {
        FloatCoord<3> fC(10.1, 20.1, 30.1);

        FloatCoord<3> fL(10.0, 20.1, 30.1);
        FloatCoord<3> fR(10.9, 20.1, 30.1);

        FloatCoord<3> fT(10.1, 20.0, 30.1);
        FloatCoord<3> fB(10.1, 20.9, 30.1);

        FloatCoord<3> fN(10.1, 20.1, 30.9);
        FloatCoord<3> fS(10.1, 20.1, 30.0);

        Coord<3> c1(10, 20, 30);
        Coord<3> c2(11, 21, 31);
        Coord<3> c3(10, 21, 30);
        Coord<3> c4(11, 20, 30);

        TS_ASSERT( fC.dominates(fC));

        TS_ASSERT(!fC.dominates(fL));
        TS_ASSERT( fC.dominates(fR));
        TS_ASSERT(!fC.dominates(fT));
        TS_ASSERT( fC.dominates(fB));
        TS_ASSERT( fC.dominates(fN));
        TS_ASSERT(!fC.dominates(fS));

        TS_ASSERT(!fC.dominates(c1));
        TS_ASSERT( fC.dominates(c2));
        TS_ASSERT(!fC.dominates(c3));
        TS_ASSERT(!fC.dominates(c4));
    }

    void testStrictlyDominates1D()
    {
        FloatCoord<1> f1(10.1);
        FloatCoord<1> f2(11.1);
        Coord<1> c1(10);
        Coord<1> c2(11);

        TS_ASSERT(!f1.strictlyDominates(f1));
        TS_ASSERT( f1.strictlyDominates(f2));
        TS_ASSERT(!f1.strictlyDominates(c1));
        TS_ASSERT( f1.strictlyDominates(c2));

        TS_ASSERT(!f2.strictlyDominates(f1));
        TS_ASSERT(!f2.strictlyDominates(f2));
        TS_ASSERT(!f2.strictlyDominates(c1));
        TS_ASSERT(!f2.strictlyDominates(c2));
    }

    void testStrictlyDominates2D()
    {
        FloatCoord<2> fC(10.1, 10.5);

        FloatCoord<2> fR(11.1, 10.5);
        FloatCoord<2> fT(10.1, 10.1);
        FloatCoord<2> fL(10.0, 10.5);
        FloatCoord<2> fB(10.1, 11.5);

        FloatCoord<2> fDiag(10.2, 10.6);

        Coord<2> c1(10, 10);
        Coord<2> c2(11, 11);

        TS_ASSERT(!fC.strictlyDominates(fC));

        TS_ASSERT(!fC.strictlyDominates(fR));
        TS_ASSERT(!fC.strictlyDominates(fB));
        TS_ASSERT(!fC.strictlyDominates(fL));
        TS_ASSERT(!fC.strictlyDominates(fT));

        TS_ASSERT( fC.strictlyDominates(fDiag));

        TS_ASSERT(!fC.strictlyDominates(c1));
        TS_ASSERT( fC.strictlyDominates(c2));

    }

    void testStrictlyDominates3D()
    {
        FloatCoord<3> fC(10.1, 20.1, 30.1);

        FloatCoord<3> fL(10.0, 20.1, 30.1);
        FloatCoord<3> fR(10.9, 20.1, 30.1);

        FloatCoord<3> fT(10.1, 20.0, 30.1);
        FloatCoord<3> fB(10.1, 20.9, 30.1);

        FloatCoord<3> fN(10.1, 20.1, 30.9);
        FloatCoord<3> fS(10.1, 20.1, 30.0);

        FloatCoord<3> fDiag(10.2, 20.2, 30.2);

        Coord<3> c1(10, 20, 30);
        Coord<3> c2(11, 21, 31);
        Coord<3> c3(10, 21, 30);
        Coord<3> c4(11, 20, 30);

        TS_ASSERT(!fC.strictlyDominates(fC));

        TS_ASSERT(!fC.strictlyDominates(fL));
        TS_ASSERT(!fC.strictlyDominates(fR));
        TS_ASSERT(!fC.strictlyDominates(fT));
        TS_ASSERT(!fC.strictlyDominates(fB));
        TS_ASSERT(!fC.strictlyDominates(fN));
        TS_ASSERT(!fC.strictlyDominates(fS));

        TS_ASSERT( fC.strictlyDominates(fDiag));

        TS_ASSERT(!fC.strictlyDominates(c1));
        TS_ASSERT( fC.strictlyDominates(c2));
        TS_ASSERT(!fC.strictlyDominates(c3));
        TS_ASSERT(!fC.strictlyDominates(c4));
    }

    void testDim()
    {
        TS_ASSERT_EQUALS(1, FloatCoord<1>::DIM);
        TS_ASSERT_EQUALS(2, FloatCoord<2>::DIM);
        TS_ASSERT_EQUALS(3, FloatCoord<3>::DIM);
    }

    void testCrossProduct()
    {
        TS_ASSERT_EQUALS(FloatCoord<3>( 1, 0, 0), FloatCoord<3>(0, 1, 0).crossProduct(FloatCoord<3>(0, 0, 1)));
        TS_ASSERT_EQUALS(FloatCoord<3>(-1, 0, 0), FloatCoord<3>(0, 0, 1).crossProduct(FloatCoord<3>(0, 1, 0)));

        TS_ASSERT_EQUALS(FloatCoord<3>(0, -1, 0), FloatCoord<3>(1, 0, 0).crossProduct(FloatCoord<3>(0, 0, 1)));
        TS_ASSERT_EQUALS(FloatCoord<3>(0,  1, 0), FloatCoord<3>(0, 0, 1).crossProduct(FloatCoord<3>(1, 0, 0)));

        TS_ASSERT_EQUALS(FloatCoord<3>(0, 0, -1), FloatCoord<3>(0, 1, 0).crossProduct(FloatCoord<3>(1, 0, 0)));
        TS_ASSERT_EQUALS(FloatCoord<3>(0, 0,  1), FloatCoord<3>(1, 0, 0).crossProduct(FloatCoord<3>(0, 1, 0)));
    }
};

}
