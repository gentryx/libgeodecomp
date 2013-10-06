#include <boost/date_time/posix_time/posix_time.hpp>
#include <libgeodecomp/misc/chronometer.h>

namespace LibGeoDecomp {

void Chronometer::tic()
{
   workIntervalStart = ScopedTimer::timeUSec();
}

void Chronometer::toc()
{
    if (workIntervalStart == 0) {
        throw std::logic_error("Call Chronometer::tic() before calling toc()");
    }
   workLength += ScopedTimer::timeUSec() - workIntervalStart;
}

double Chronometer::nextCycle()
{
    long long work, cycle;
    nextCycle(&cycle, &work);
    return ((double)work) / cycle;
}

void Chronometer::nextCycle(long long *cycleLength, long long *curWorkLength)
{
    *cycleLength = ScopedTimer::timeUSec() - cycleStart;
    *curWorkLength = workLength;
    startCycle();
}

void Chronometer::startCycle()
{
   workLength = 0;
   workIntervalStart = 0;
   cycleStart = ScopedTimer::timeUSec();
}

}
