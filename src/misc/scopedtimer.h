#ifndef LIBGEODECOMP_MISC_SCOPEDTIMER_H
#define LIBGEODECOMP_MISC_SCOPEDTIMER_H

#include <boost/date_time/posix_time/posix_time.hpp>

namespace LibGeoDecomp {

class ScopedTimer
{
public:
    inline ScopedTimer(double *totalElapsedTime) :
        totalElapsedTime(totalElapsedTime),
        startTime(time())
    {}

    ~ScopedTimer()
    {
        *totalElapsedTime += time() - startTime;
    }

    /**
     * returns the current time, measured in microseconds (and with according accuracy).
     */
    static long timeUSec()
    {
        boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
        return now.time_of_day().total_microseconds();
    }

    /**
     * returns the time in secods, with microsecond accuracy.
     */
    static double time()
    {
        return timeUSec() * 1e-6;
    }

private:
    double *totalElapsedTime;
    double startTime;
};

}

#endif
