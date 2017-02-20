#ifndef LIBGEODECOMP_MISC_SCOPEDTIMER_H
#define LIBGEODECOMP_MISC_SCOPEDTIMER_H

#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/config.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#else
#include <sys/time.h>
#endif

namespace LibGeoDecomp {

/**
 * Records the time between creation and destruction.
 */
class ScopedTimer
{
public:
    inline explicit ScopedTimer(double *totalElapsedTime) :
        totalElapsedTime(totalElapsedTime),
        startTime(time())
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
        return hpx::util::high_resolution_timer::now();
#else
        timeval t;
        gettimeofday(&t, 0);

        return t.tv_sec + t.tv_usec * 1.0e-6;
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
    double *totalElapsedTime;
    double startTime;
};

}

#endif
