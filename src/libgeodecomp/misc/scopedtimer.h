#ifndef LIBGEODECOMP_MISC_SCOPEDTIMER_H
#define LIBGEODECOMP_MISC_SCOPEDTIMER_H

#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_HPX
#include <libgeodecomp/misc/cudaboostworkaround.h>
#include <hpx/config.hpp>
#include <hpx/timing/high_resolution_timer.hpp>
#else
#include <libflatarray/testbed/benchmark.hpp>
#endif

namespace LibGeoDecomp {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4820 )
#endif

/**
 * Records the time between creation and destruction.
 */
class ScopedTimer
{
public:
    inline explicit ScopedTimer(double *totalElapsedTime) :
        startTime(time()),
        totalElapsedTime(totalElapsedTime)
    {}

    ~ScopedTimer()
    {
        *totalElapsedTime += time() - startTime;
    }

    /**
     * returns the time in secods, with microsecond accuracy.
     */
    static double time()
    {
#ifdef LIBGEODECOMP_WITH_HPX
        return hpx::chrono::high_resolution_timer::now();
#else
        return LibFlatArray::benchmark::time();
#endif
    }

    /**
     * sleeps for the given period. This is more accurate than usleep() and friends.
     */
    static void busyWait(long microSeconds)
    {
        double t0 = ScopedTimer::time();
        double elapsed = 0;
        while (elapsed < (microSeconds * 1e-6)) {
            elapsed = ScopedTimer::time() - t0;
        }
    }

private:
    double startTime;
    double *totalElapsedTime;
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
