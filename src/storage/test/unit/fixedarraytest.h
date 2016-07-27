#include <cxxtest/TestSuite.h>
#include <libgeodecomp/storage/fixedarray.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class FixedArrayTest : public CxxTest::TestSuite
{
public:

    void testInsertDelete()
    {
        FixedArray<int, 20> a;
        TS_ASSERT_EQUALS(std::size_t(0), a.size());

        a << 0
          << 1
          << 2
          << 3;
        TS_ASSERT_EQUALS(std::size_t(4), a.size());
        TS_ASSERT_EQUALS(0, a[0]);
        TS_ASSERT_EQUALS(1, a[1]);
        TS_ASSERT_EQUALS(2, a[2]);
        TS_ASSERT_EQUALS(3, a[3]);

        a.erase(&a[1]);
        TS_ASSERT_EQUALS(std::size_t(3), a.size());
        TS_ASSERT_EQUALS(0, a[0]);
        TS_ASSERT_EQUALS(2, a[1]);
        TS_ASSERT_EQUALS(3, a[2]);

        FixedArray<int, 10> b;
        TS_ASSERT_THROWS(b.reserve(11), std::out_of_range);

        b.reserve(3);
        TS_ASSERT_EQUALS(std::size_t(3), b.size());
        std::copy(a.begin(), a.end(), b.begin());
        TS_ASSERT_EQUALS(0, b[0]);
        TS_ASSERT_EQUALS(2, b[1]);
        TS_ASSERT_EQUALS(3, b[2]);

        a.clear();
        TS_ASSERT_EQUALS(std::size_t(0), a.size());
    }

    void testRemove()
    {
        FixedArray<int, 30> array;
        array << 10 << 11 << 12 << 13;

        FixedArray<int, 10> expectedA;
        expectedA << 10 << 12 << 13;

        FixedArray<int, 15> expectedB;
        expectedB << 12 << 13;

        FixedArray<int, 20> expectedC;
        expectedC << 12;

        FixedArray<int, 25> expectedD;

        array.remove(1);
        TS_ASSERT_EQUALS(array, expectedA);

        array.remove(0);
        TS_ASSERT_EQUALS(array, expectedB);

        array.remove(1);
        TS_ASSERT_EQUALS(array, expectedC);

        array.remove(0);
        TS_ASSERT_EQUALS(array, expectedD);
    }

    void testEraseRemoveOnEnd()
    {
        FixedArray<double, 1024> a(1024, 5);
        a.erase(a.end() - 1);

        a << 123;
        a.remove(509);
    }

    void testConstructors()
    {
        FixedArray<int, 4> a(1, 13);
        FixedArray<int, 4> b(4, 47);
        FixedArray<int, 4> c(3, 11);

        TS_ASSERT_EQUALS(a.size(), std::size_t(1));
        TS_ASSERT_EQUALS(a[0], 13);

        TS_ASSERT_EQUALS(b.size(), std::size_t(4));
        TS_ASSERT_EQUALS(b[0], 47);
        TS_ASSERT_EQUALS(b[1], 47);
        TS_ASSERT_EQUALS(b[2], 47);
        TS_ASSERT_EQUALS(b[3], 47);

        TS_ASSERT_EQUALS(c.size(), std::size_t(3));
        TS_ASSERT_EQUALS(c[0], 11);
        TS_ASSERT_EQUALS(c[1], 11);
        TS_ASSERT_EQUALS(c[2], 11);

        a = b;
        TS_ASSERT_EQUALS(a.size(), std::size_t(4));
        TS_ASSERT_EQUALS(a[0], 47);
        TS_ASSERT_EQUALS(a[1], 47);
        TS_ASSERT_EQUALS(a[2], 47);
        TS_ASSERT_EQUALS(a[3], 47);

        a = c;
        TS_ASSERT_EQUALS(a.size(), std::size_t(3));
        TS_ASSERT_EQUALS(a[0], 11);
        TS_ASSERT_EQUALS(a[1], 11);
        TS_ASSERT_EQUALS(a[2], 11);
    }

    void testAddition()
    {
        FixedArray<int, 6> sourceA;
        FixedArray<int, 6> sourceB;
        FixedArray<int, 6> expected;

        sourceA  << 1 << 3 << 5;
        sourceB  << 2 << 1 << 0 << 4;
        expected << 3 << 4 << 5 << 4;

        TS_ASSERT_EQUALS(sourceA + sourceB, expected);
        TS_ASSERT_EQUALS(sourceB + sourceA, expected);
        TS_ASSERT_EQUALS(sourceA += sourceB, expected);
        TS_ASSERT_EQUALS(sourceA, expected);
    }

    void testSubstraction()
    {
        FixedArray<int, 6> sourceA;
        FixedArray<int, 6> sourceB;
        FixedArray<int, 6> expected1;
        FixedArray<int, 6> expected2;

        sourceA   <<  1 <<  3 <<  5;
        sourceB   <<  2 <<  1 <<  0 <<  4;
        expected1 << -1 <<  2 <<  5 << -4;
        expected2 <<  1 << -2 << -5 <<  4;

        TS_ASSERT_EQUALS(sourceA - sourceB, expected1);
        TS_ASSERT_EQUALS(sourceB - sourceA, expected2);
        TS_ASSERT_EQUALS(sourceA -= sourceB, expected1);
        TS_ASSERT_EQUALS(sourceA, expected1);
    }

    void testDivision()
    {
        FixedArray<int, 6> source;
        FixedArray<int, 6> expected;

        source   << 3 << 9 << 15;
        expected << 1 << 3 << 5;

        TS_ASSERT_EQUALS(source / 3, expected);
        TS_ASSERT_EQUALS(source /= 3, expected);
        TS_ASSERT_EQUALS(source, expected);
    }

    void testMultiplication()
    {
        FixedArray<int, 6> source;
        FixedArray<int, 6> expected;

        source   << 1 << 3 << 5;
        expected << 3 << 9 << 15;

        TS_ASSERT_EQUALS(source * 3, expected);
        TS_ASSERT_EQUALS(source *= 3, expected);
        TS_ASSERT_EQUALS(source, expected);
    }

    void testComparison()
    {
        FixedArray<int, 6> a;
        FixedArray<int, 6> b;
        FixedArray<int, 6> c;

        a << 1 << 3 << 5;
        b << 1 << 9 << 15;
        c << 1 << 9 << 15 << 0;

        TS_ASSERT( (a == a));
        TS_ASSERT(!(a <  a));
        TS_ASSERT(!(a >  a));

        TS_ASSERT(!(a == b));
        TS_ASSERT( (a <  b));
        TS_ASSERT(!(a >  b));

        TS_ASSERT(!(a == c));
        TS_ASSERT( (a <  c));
        TS_ASSERT(!(a >  c));

        TS_ASSERT(!(b == a));
        TS_ASSERT(!(b <  a));
        TS_ASSERT( (b >  a));

        TS_ASSERT( (b == b));
        TS_ASSERT(!(b <  b));
        TS_ASSERT(!(b >  b));

        TS_ASSERT(!(b == c));
        TS_ASSERT( (b <  c));
        TS_ASSERT(!(b >  c));

        TS_ASSERT(!(c == a));
        TS_ASSERT(!(c <  a));
        TS_ASSERT( (c >  a));

        TS_ASSERT(!(c == b));
        TS_ASSERT(!(c <  b));
        TS_ASSERT( (c >  b));

        TS_ASSERT( (c == c));
        TS_ASSERT(!(c <  c));
        TS_ASSERT(!(c >  c));
    }

    void testToString()
    {
        std::stringstream s;
        FixedArray<int, 6> a;
        a << 1 << 2 << 3 << 5 << 6;
        s << a;

        TS_ASSERT_EQUALS(s.str(), "(1, 2, 3, 5, 6)");
    }
};

}
