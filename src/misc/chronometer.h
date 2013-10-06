#ifndef LIBGEODECOMP_MISC_CHRONOMETER_H
#define LIBGEODECOMP_MISC_CHRONOMETER_H

#include <libgeodecomp/misc/fixedarray.h>
#include <libgeodecomp/misc/scopedtimer.h>

namespace LibGeoDecomp {

/**
 * This class can be used to measure execution time of different parts
 * of our code. This is useful to determine the relative load of a
 * node or to find out which part of the algorithm the most time.
 */
class Chronometer
{
public:
    friend class ChronometerTest;
    friend class Typemaps;

    enum TimeInterval {
        TOTAL_TIME,
        COMPUTE_TIME,
        COMPUTE_TIME_INNER,
        COMPUTE_TIME_GHOST,
        PATCH_ACCEPTERS_TIME,
        PATCH_PROVIDERS_TIME
    };

    static const int NUM_INTERVALS = 6;

    Chronometer() :
        totalTimes(NUM_INTERVALS, 0)
    {
        reset();
    }

    /**
     * Aggregates two chronometers. This is good for accumulating
     * measurements from multiple sources.
     */
    Chronometer& operator+=(const Chronometer& other)
    {
        for (int i = 0; i < NUM_INTERVALS; ++i) {
            totalTimes[i] += other.totalTimes[i];
        }

        return *this;
    }

    Chronometer operator+(const Chronometer& other) const
    {
        Chronometer ret(*this);
        ret += other;
        return ret;
    }
    /**
     * Starts measurement for the given interval. Timing will be
     * stopped when the ScopedTimer dies.
     */
    ScopedTimer tic(TimeInterval interval)
    {
        return ScopedTimer(&totalTimes[interval]);
    }

    /**
     * Flushes all time totals to 0.
     */
    void reset()
    {
        std::fill(totalTimes.begin(), totalTimes.end(), 0);
    }

    double interval(TimeInterval i) const
    {
        return totalTimes[i];
    }

    /**
     * Returns the ratio of the accumulated times i1 and i2, returns
     * 0.5 if the second interval is empty. (ratio() is typically used
     * for load balancing. For that purpose 0.5 is easier to digest
     * than NaN.)
     */
    double ratio(TimeInterval i1 = COMPUTE_TIME, TimeInterval i2 = TOTAL_TIME)
    {
        if (totalTimes[i2] == 0) {
            return 0.5;
        }
        return totalTimes[i1] / totalTimes[i2];
    }

private:
    FixedArray<double, Chronometer::NUM_INTERVALS> totalTimes;
};

}

#endif
