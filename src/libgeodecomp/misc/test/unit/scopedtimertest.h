#include <libgeodecomp/misc/scopedtimer.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <unistd.h>
#include <cxxtest/TestSuite.h>

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
        double  tOld = ScopedTimer::time();
        double  tNew;

        for (int i = 0; i < 4; i++) {
            usleep(250000);
            tNew = ScopedTimer::time();
            TS_ASSERT(tNew > tOld);
            tOld = tNew;
        }
    }

    void testBasicUsageAndBusyWait()
    {
        int max = 1000;
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

        // drop largest 40 percent to ignore OS jitter
        for (int i = 0; i < (0.6 * max); ++i) {
            // accept at most 10% deviation
            TS_ASSERT_LESS_THAN(seconds * 0.9, times[i]);
            TS_ASSERT_LESS_THAN(times[i], seconds * 1.1);
        }
    }
};

}
