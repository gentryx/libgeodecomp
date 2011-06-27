#include <cxxtest/TestSuite.h>
#include "../../random.h"

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class RandomTest : public CxxTest::TestSuite 
{

public:

    void testDouble1cd ()
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
};

}
