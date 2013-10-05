#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/scopedtimer.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class ScopedTimerTest : public CxxTest::TestSuite
{
public:

    void testBasic()
    {
        int max = 100;
        long microSeconds = 250;
        double seconds = microSeconds * 1e-6;
        std::vector<double> times(max, 0);

        for (int i = 0; i < max; ++i) {
            ScopedTimer t(&times[i]);

            // busy wait because usleep isn't accurate enough
            long t0 = ScopedTimer::timeUSec();
            long elapsed = 0;
            while (elapsed < microSeconds) {
                elapsed = ScopedTimer::timeUSec() - t0;
            }
        }

        sort(times);

        // drop largest 10 percent to ignore OS jitter
        for (int i = 0; i < (0.9 * max); ++i) {
            // accept at most 1% deviation
            TS_ASSERT_LESS_THAN(seconds * 0.99, times[i]);
            TS_ASSERT_LESS_THAN(times[i], seconds * 1.01);
        }
    }
};

}
