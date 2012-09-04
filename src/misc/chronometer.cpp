#include <boost/date_time/posix_time/posix_time.hpp>
#include <libgeodecomp/misc/chronometer.h>

namespace LibGeoDecomp {

Chronometer::Chronometer()
{
    startCycle();
}

void Chronometer::tic()
{
    _workIntervalStart = time();
}

void Chronometer::toc()
{
    if (_workIntervalStart == 0) {
        throw std::logic_error( "Call Chronometer::tic() before calling toc()");
    }
    _workLength += time() - _workIntervalStart;
}

double Chronometer::nextCycle()
{
    long long work, cycle;
    nextCycle(&cycle, &work);
    return ((double)work) / cycle;
}

void Chronometer::nextCycle(long long *cycleLength, long long *workLength)
{
    *cycleLength = time() - _cycleStart;    
    *workLength = _workLength;
    startCycle();
}

long long Chronometer::time() const
{
    return timeUSec();
}

long long Chronometer::timeUSec()
{
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    return now.time_of_day().total_microseconds();
}

void Chronometer::startCycle()
{
    _workLength = 0;
    _workIntervalStart = 0;
    _cycleStart = time();    
}

};
