#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/supervector.h>

using namespace boost::assign;
using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class SuperVectorTest : public CxxTest::TestSuite
{
public:

    void deleteChecker(int excludeObj)
    {
        // create test object
        int size = 7;
        SuperVector<int> original(size);
        for (int i = 0; i < size - 2; i++) {
            original[i] = 1 << i;
        }

        original[size - 2] = 4;
        original[size - 1] = 4;
        original.del(excludeObj);

        // create reference
        SuperVector<int> cropped;
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
        SuperVector<int> actual(expected.begin(), expected.end());
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
        SuperVector<int> stack;
        stack << 1
              << 2
              << 3;
        TS_ASSERT_EQUALS(3, stack.pop());
        TS_ASSERT_EQUALS(2, stack.pop());
        TS_ASSERT_EQUALS(1, stack.pop());
    }

    void testPopFront()
    {
        SuperVector<int> stack;
        stack << 1
              << 2
              << 3;
        TS_ASSERT_EQUALS(1, stack.pop_front());
        TS_ASSERT_EQUALS(2, stack.pop_front());
        TS_ASSERT_EQUALS(3, stack.pop_front());
    }

    void testPushFront()
    {
        SuperVector<int> a;
        a += 47, 11, 2000;

        SuperVector<int> b;
        b += 11, 2000;
        b.push_front(47);

        TS_ASSERT_EQUALS(a, b);
    }

    void testSum()
    {
        SuperVector<int> s;
        s += 12, 43, -9, -8, 15;
        TS_ASSERT_EQUALS(53, s.sum());
    }

    void testAppend()
    {
        SuperVector<int> a;
        a += 1, 2, 3;
        SuperVector<int> b;
        b += 4, 5;
        SuperVector<int> c;
        c += 1, 2, 3, 4, 5;

        TS_ASSERT_EQUALS(a + b, c);
        a.append(b);
        TS_ASSERT_EQUALS(a, c);
    }

    void testOperatorLessLess()
    {
        SuperVector<int> a;
        a += 1, 2, 3;
        std::ostringstream temp;
        temp << a;
        TS_ASSERT_EQUALS("[1, 2, 3]", temp.str());
    }

    void testContains()
    {
        SuperVector<int> a;
        a += 0, 1;
        TS_ASSERT_EQUALS(a.contains(2), false);
        TS_ASSERT_EQUALS(a.contains(1), true);
    }

    void testSort()
    {
        SuperVector<unsigned> v;
        SuperVector<unsigned> w;
        v += 0, 3, 1, 2;
        w += 0, 1, 2, 3;
        v.sort();
        TS_ASSERT_EQUALS(v, w);
    }

    void testMax()
    {
        SuperVector<unsigned> a;
        a += 0, 3, 1 ,2;
        TS_ASSERT_EQUALS(a.max(), (unsigned)3);
    }
};

};
