#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/misc/random.h>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class RandomTest : public CxxTest::TestSuite
{
public:

    void testDouble1()
    {
        double sum = 0;
        for (int i = 0; i < 1000; ++i)
            sum += Random::gen_d();
        TS_ASSERT(450 < sum);
        TS_ASSERT(550 > sum);
    }

    void testDouble2()
    {
        double sum = 0;
        for (int i = 0; i < 1000; ++i)
            sum += Random::gen_d(1.0);
        TS_ASSERT(450 < sum);
        TS_ASSERT(550 > sum);
    }

    void testUnsigned()
    {
        int repeats = 1000;
        long long max = (long long)boost::integer_traits<unsigned>::const_max * repeats;
        long long sum = 0;
        for (int i = 0; i < repeats; ++i)
            sum += Random::gen_u();
        TS_ASSERT((0.45 * max) < sum);
        TS_ASSERT((0.55 * max) > sum);
    }

    void testSeed()
    {
        std::vector<double> vec1;
        std::vector<double> vec2;

        Random::seed(47);
        for (int i = 0; i < 10; ++i) {
            vec1 << Random::gen_d();
        }

        Random::seed(47);
        for (int i = 0; i < 10; ++i) {
            vec2 << Random::gen_d();
        }

        TS_ASSERT_EQUALS(vec1, vec2);

        vec1.clear();
        vec2.clear();

        Random::seed(11);
        for (int i = 0; i < 20; ++i) {
            vec1 << Random::gen_d();
        }

        Random::seed(11);
        for (int i = 0; i < 20; ++i) {
            vec2 << Random::gen_d();
        }

    }
};

}
