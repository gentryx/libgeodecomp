#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/floatcoord.h>

#include <boost/assign/std/deque.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/shared_ptr.hpp>
#include <cxxtest/TestSuite.h>

using namespace boost::assign;
using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class StdContainerOverloadsTest_StdSet : public CxxTest::TestSuite
{
public:
    void testOperatorLessLess()
    {
        std::set<int> s;
        {
            std::ostringstream temp;
            temp << s;
            TS_ASSERT_EQUALS("{}", temp.str());
        }

        s << 4;
        {
            std::ostringstream temp;
            temp << s;
            TS_ASSERT_EQUALS("{4}", temp.str());
        }

        s << 1;
        {
            std::ostringstream temp;
            temp << s;
            TS_ASSERT_EQUALS("{1, 4}", temp.str());
        }

        s << 9;
        {
            std::ostringstream temp;
            temp << s;
            TS_ASSERT_EQUALS("{1, 4, 9}", temp.str());
        }
    }

    void testInsertMinMaxAndEraseMin()
    {
        std::set<int> set;
        set.insert(2);
        set.insert(1);
        set.insert(3);
        set << 0
            << -1;
        TS_ASSERT_EQUALS((max)(set),  3);
        TS_ASSERT_EQUALS((min)(set), -1);

        erase_min(set);
        TS_ASSERT_EQUALS((min)(set), 0);
    }

    void testOperatorAndAnd()
    {
        std::set<int> set0;
        set0.insert(2);
        set0.insert(1);
        set0.insert(3);
        set0.insert(0);

        std::set<int> set1;
        set1.insert(2);
        set1.insert(4);
        set1.insert(3);
        set1.insert(0);

        std::set<int> expected;
        expected.insert(2);
        expected.insert(3);
        expected.insert(0);

        TS_ASSERT_EQUALS(set0 && set1, expected);
    }

    void testOperatorOrOr()
    {
        std::set<int> set0;
        set0.insert(2);
        set0.insert(1);
        set0.insert(3);

        std::set<int> set1;
        set1.insert(2);
        set1.insert(0);

        std::set<int> expected;
        expected.insert(1);
        expected.insert(2);
        expected.insert(3);
        expected.insert(0);

        TS_ASSERT_EQUALS(set0 || set1, expected);
    }

    void testOperatorOrEquals()
    {
        std::set<int> set0;
        std::set<int> set1;
        set0 << 1;
        set0 << 2;
        set1 << 2;
        set1 << 3;
        set0 |= set1;

        std::set<int> expected;
        expected << 1
                 << 2
                 << 3;
        TS_ASSERT_EQUALS(set0, expected);
    }

    void testOperatorPlus()
    {
        std::set<int> set0;
        std::set<int> set1;
        set0 << 1;
        set0 << 2;
        set1 << 2;
        set1 << 3;

        TS_ASSERT_EQUALS(set0 + set1, set0 || set1);
    }
};

class StdContainerOverloadsTest_StdMap : public CxxTest::TestSuite
{
public:
    void testOperatorLessLess()
    {
        std::map<int, int> a;
        {
            std::ostringstream temp;
            temp << a;
            TS_ASSERT_EQUALS("{}", temp.str());
        }

        a[0] = 1;
        {
            std::ostringstream temp;
            temp << a;
            TS_ASSERT_EQUALS("{0 => 1}", temp.str());
        }

        a[1] = 2;
        {
            std::ostringstream temp;
            temp << a;
            TS_ASSERT_EQUALS("{0 => 1, 1 => 2}", temp.str());
        }

        a[2] = 3;
        {
            std::ostringstream temp;
            temp << a;
            TS_ASSERT_EQUALS("{0 => 1, 1 => 2, 2 => 3}", temp.str());
        }
    }
};

class StdContainerOverloadsTest_StdVector : public CxxTest::TestSuite
{
public:

    void deleteChecker(int excludeObj)
    {
        // create test object
        int size = 7;
        std::vector<int> original(size);
        for (int i = 0; i < size - 2; i++) {
            original[i] = 1 << i;
        }

        original[size - 2] = 4;
        original[size - 1] = 4;
        del(original, excludeObj);

        // create reference
        std::vector<int> cropped;
        for (int i = 0; i < size - 2; i++) {
            int val = 1 << i;
            if (val != excludeObj) {
                cropped.push_back(val);
            }
        }

        if (4 != excludeObj) {
            cropped.push_back(4);
            cropped.push_back(4);
        }

        TS_ASSERT_EQUALS(original, cropped);
    }

    void testConstructor()
    {
        std::vector<int> expected;
        expected.push_back(4);
        expected.push_back(7);
        expected.push_back(11);
        std::vector<int> actual(expected.begin(), expected.end());
        TS_ASSERT_EQUALS(actual.size(), expected.size());
        TS_ASSERT_EQUALS(actual[0], expected[0]);
        TS_ASSERT_EQUALS(actual[1], expected[1]);
        TS_ASSERT_EQUALS(actual[2], expected[2]);
    }

    void testDelete()
    {
        deleteChecker(-1);
        deleteChecker(1);
        deleteChecker(4);
        deleteChecker(16);
    }

    void testPop()
    {
        std::vector<int> stack;
        stack << 1
              << 2
              << 3;
        TS_ASSERT_EQUALS(3, pop(stack));
        TS_ASSERT_EQUALS(2, pop(stack));
        TS_ASSERT_EQUALS(1, pop(stack));
    }

    void testPopFront()
    {
        std::vector<int> stack;
        stack << 1
              << 2
              << 3;
        TS_ASSERT_EQUALS(1, pop_front(stack));
        TS_ASSERT_EQUALS(2, pop_front(stack));
        TS_ASSERT_EQUALS(3, pop_front(stack));
    }

    void testMove()
    {
        boost::shared_ptr<double> ptr(new double);
        *ptr = 555.666;
        TS_ASSERT(0 != get_pointer(ptr));
        std::vector<boost::shared_ptr<double> > vec;

        vec << ptr;

        TS_ASSERT(0 != get_pointer(ptr));
        TS_ASSERT_EQUALS(555.666, *ptr);
    }

    void testPushFront()
    {
        std::vector<int> a;
        a += 47, 11, 2000;

        std::vector<int> b;
        b += 11, 2000;
        push_front(b, 47);

        TS_ASSERT_EQUALS(a, b);
    }

    void testSum()
    {
        std::vector<int> s;
        s += 12, 43, -9, -8, 15;
        TS_ASSERT_EQUALS(53, sum(s));
    }

    void testAppend()
    {
        std::vector<int> a;
        a += 1, 2, 3;
        std::vector<int> b;
        b += 4, 5;
        std::vector<int> c;
        c += 1, 2, 3, 4, 5;

        TS_ASSERT_EQUALS(a + b, c);
        append(a, b);
        TS_ASSERT_EQUALS(a, c);
    }

    void testCoordToVector()
    {
        std::vector<int> expected1;
        expected1 += 1, 2, 4;
        TS_ASSERT_EQUALS(expected1, toVector(Coord<3>(1, 2, 4)));

        std::vector<int> expected2;
        expected2 += 6, 8;
        TS_ASSERT_EQUALS(expected2, toVector(Coord<2>(6, 8)));

        std::vector<int> expected3;
        expected3 += 9;
        TS_ASSERT_EQUALS(expected3, toVector(Coord<1>(9)));

        std::vector<double> expected4;
        expected4 += 1.2, 3.4;
        TS_ASSERT_EQUALS(expected4, toVector(FloatCoord<2>(1.2, 3.4)));
    }

    void testOperatorLessLess()
    {
        std::vector<int> a;
        {
            std::ostringstream temp;
            temp << a;
            TS_ASSERT_EQUALS("[]", temp.str());
        }

        a += 1;
        {
            std::ostringstream temp;
            temp << a;
            TS_ASSERT_EQUALS("[1]", temp.str());
        }

        a += 2, 3;
        {
            std::ostringstream temp;
            temp << a;
            TS_ASSERT_EQUALS("[1, 2, 3]", temp.str());
        }
    }

    void testContains()
    {
        std::vector<int> a;
        a += 0, 1;
        TS_ASSERT_EQUALS(contains(a, 2), false);
        TS_ASSERT_EQUALS(contains(a, 1), true);
    }

    void testSort()
    {
        std::vector<unsigned> v;
        std::vector<unsigned> w;
        v += 0, 3, 1, 2;
        w += 0, 1, 2, 3;
        sort(v);
        TS_ASSERT_EQUALS(v, w);
    }

    void testMinMaxVector()
    {
        std::vector<int> a;
        a += 0, 3, 1 ,2;
        TS_ASSERT_EQUALS((min)(a), 0);
        TS_ASSERT_EQUALS((max)(a), 3);

        (min)(a) = 47;
        TS_ASSERT_EQUALS((min)(a), 1);
        TS_ASSERT_EQUALS((max)(a), 47);

        (max)(a) = -1;
        TS_ASSERT_EQUALS((min)(a), -1);
        TS_ASSERT_EQUALS((max)(a), 3);

        del(a, -1);
        del(a, 3);
        TS_ASSERT_EQUALS((min)(a), 1);
        TS_ASSERT_EQUALS((max)(a), 2);
    }
};


class StdContainerOverloadsTest_StdDeque : public CxxTest::TestSuite
{
public:
    void testOperatorLessLess()
    {
        std::deque<int> s;
        {
            std::ostringstream temp;
            temp << s;
            TS_ASSERT_EQUALS("()", temp.str());
        }

        s << 1;

        {
            std::ostringstream temp;
            temp << s;
            TS_ASSERT_EQUALS("(1)", temp.str());
        }

        s << 3 << 5;

        {
            std::ostringstream temp;
            temp << s;
            TS_ASSERT_EQUALS("(1, 3, 5)", temp.str());
        }
    }

    void testOperatorPlus()
    {
        std::deque<int> a;
        std::deque<int> b;
        std::deque<int> c;
        std::deque<int> d;

        a << 1 << 2 << 3 << 4;
        b << 5 << 6 << 7 << 0;

        c << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 0;

        d = a + b;

        TS_ASSERT_EQUALS(d, c);
    }

    void testMinMax()
    {
        std::deque<int> a;
        a << 4 << 1 << 5 << 2 << 7 << 9;

        TS_ASSERT_EQUALS((min)(a), 1);
        TS_ASSERT_EQUALS((max)(a), 9);
    }

    void testSort()
    {
        std::deque<int> a;
        std::deque<int> b;
        a << 4 << 1 << 5 << 2 << 7 << 9 << 0;
        b << 0 << 1 << 2 << 4 << 5 << 7 << 9;

        sort(a);
        TS_ASSERT_EQUALS(a, b);
    }

    void testContains()
    {
        std::deque<int> a;
        a += 0, 1, 4, -1;
        TS_ASSERT_EQUALS(contains(a,  2), false);
        TS_ASSERT_EQUALS(contains(a,  3), false);
        TS_ASSERT_EQUALS(contains(a,  1), true);
        TS_ASSERT_EQUALS(contains(a, -1), true);
        TS_ASSERT_EQUALS(contains(a,  4), true);
    }

    void testSum()
    {
        std::deque<int> a;
        a += 5, 2, 6, 1, 3, 100;

        TS_ASSERT_EQUALS(sum(a), 117);
    }
};

}
