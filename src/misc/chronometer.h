#ifndef LIBGEODECOMP_MISC_CHRONOMETER_H
#define LIBGEODECOMP_MISC_CHRONOMETER_H

#include <libgeodecomp/misc/scopedtimer.h>

namespace LibGeoDecomp {

/**
 * This class takes over all the time-keeping needed to find out how
 * much time a Simulator spends on working and how long it has to wait
 * for communication with his mates.
 *
 * The idea behind the Chronometer's interface is that for an
 * arbitrary slice of time -- a cycle -- all the times \f$w_i\f$ spent
 * on working is recorded. Along with the total length \f$t\f$ of the
 * complete cycle, it is possible to compute the fraction \f$f\f$ of
 * time spent on working (in contrast to the time spent on blocking
 * for communication etc.):
 *
 * \f[
 * f = \frac{1}{t} \sum_{i=0}^{i<n} w_i
 * \f]
 */
class Chronometer
{
public:
    // fixme:
    // 1. rework so that StripingSimulator will operate on it
    // 2. fix comments
    // 3. make time optinally use HPX timer
    // 4. get rid of Statistics
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

    static const int NUM_INTERVALS = 5;

    Chronometer()
    {
        startCycle();
        reset();
    }

    /**
     * The time passing between tic() and toc() will be
     * counted as one \f$w_i\f$.
     */
    // fixme: kill this
    void tic();

    /**
     * see tic().
     */
    // fixme: kill this
    void toc();

    /**
     * starts measurement for the given interval. Timing will be stopped when the ScopedTimer dies.
     */
    ScopedTimer tic(TimeInterval interval)
    {
        return ScopedTimer(&totalTimes[interval]);
    }

    /**
     * flushes all time totals to 0.
     */
    void reset()
    {
        for (int i = 0; i < NUM_INTERVALS; ++i) {
            totalTimes[i] = 0;
        }
    }

    /**
     * Reset the start time of the current cycle and flush all the
     * working times \f$w_i\f$, returns \f$f\f$.
     */
    // fixme: kill this
    double nextCycle();

    /**
     * Reset the start time of the current cycle and flush all the
     * woking times \f$w_i\f$, returns (\f$f\f$,\f$t\f$).
     */
    void nextCycle(long long *cycleLength, long long *workLength);

    double ratio(TimeInterval i1, TimeInterval i2)
    {
        return totalTimes[i1] / totalTimes[i2];
    }

private:
    double totalTimes[NUM_INTERVALS];
    long long cycleStart;
    long long workIntervalStart;
    long long workLength;

    /**
     * returns a timestamp from a high resolution timer. There are no
     * warranties for this method except that consecutive calls will
     * return strictly monotonic increasing values.
     */
    long long time() const;

    void startCycle();
};

}

#endif
