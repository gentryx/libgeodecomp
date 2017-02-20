#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/misc/random.h>
#include <cxxtest/TestSuite.h>
#include <limits>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class RandomTest : public CxxTest::TestSuite
{
public:

    void testDouble1()
    {
        double sum = 0;
        for (int i = 0; i < 1000; ++i) {
            sum += Random::genDouble();
        }

        TS_ASSERT(450 < sum);
        TS_ASSERT(550 > sum);
    }

    void testDouble2()
    {
        double sum = 0;
        for (int i = 0; i < 1000; ++i) {
            sum += Random::genDouble(1.0);
        }

        TS_ASSERT(450 < sum);
        TS_ASSERT(550 > sum);
    }

    void testUnsigned()
    {
        int repeats = 1000;
        long long max = (long long)(std::numeric_limits<unsigned>::max()) * repeats;
        long long sum = 0;

        for (int i = 0; i < repeats; ++i) {
            sum += Random::genUnsigned();
        }

        TS_ASSERT((0.45 * max) < sum);
        TS_ASSERT((0.55 * max) > sum);
    }

    void testSeed()
    {
        std::vector<double> vec1;
        std::vector<double> vec2;

        Random::seed(47);
        for (int i = 0; i < 10; ++i) {
            vec1 << Random::genDouble();
        }

        Random::seed(47);
        for (int i = 0; i < 10; ++i) {
            vec2 << Random::genDouble();
        }

        TS_ASSERT_EQUALS(vec1, vec2);

        vec1.clear();
        vec2.clear();

        Random::seed(11);
        for (int i = 0; i < 20; ++i) {
            vec1 << Random::genDouble();
        }

        Random::seed(11);
        for (int i = 0; i < 20; ++i) {
            vec2 << Random::genDouble();
        }

    }
};

}
