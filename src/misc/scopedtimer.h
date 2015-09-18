#ifndef LIBGEODECOMP_MISC_SCOPEDTIMER_H
#define LIBGEODECOMP_MISC_SCOPEDTIMER_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/config.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#else
#include <boost/date_time/posix_time/posix_time.hpp>
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
        boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
        return now.time_of_day().total_microseconds() * 1e-6;
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
