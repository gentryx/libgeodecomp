#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/scopedtimer.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class ScopedTimerTest : public CxxTest::TestSuite
{
public:

    void testTimeConsecutive()
    {
        for (int i = 0; i < 1024; i++) {
            TS_ASSERT(ScopedTimer::time() <= ScopedTimer::time());
        }
    }

    void testTimeOverflow()
    {
        long tOld = ScopedTimer::timeUSec();
        long tNew;

        for (int i = 0; i < 4; i++) {
            usleep(250000);
            tNew = ScopedTimer::timeUSec();
            TS_ASSERT(tNew > tOld);
            tOld = tNew;
        }
    }

    void testBasicUsageAndBusyWait()
    {
        int max = 100;
        long microSeconds = 250;
        double seconds = microSeconds * 1e-6;
        std::vector<double> times(max, 0);
        double timeOdd = 0;
        double timeEven = 0;

        for (int i = 0; i < max; ++i) {
            ScopedTimer timerA(&times[i]);
            ScopedTimer timerB(i % 2 ? &timeOdd : &timeEven);

            // busy wait because usleep isn't accurate enough
            ScopedTimer::busyWait(microSeconds);
        }

        sort(times);

        // drop largest 10 percent to ignore OS jitter
        for (int i = 0; i < (0.9 * max); ++i) {
            // accept at most 5% deviation
            TS_ASSERT_LESS_THAN(seconds * 0.95, times[i]);
            TS_ASSERT_LESS_THAN(times[i], seconds * 1.05);
        }
    }
};

}
