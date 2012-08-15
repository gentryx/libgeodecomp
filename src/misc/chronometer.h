#ifndef _libgeodecomp_parallelization_chronometer_h_
#define _libgeodecomp_parallelization_chronometer_h_

#include <stdexcept>

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
    friend class ChronometerTest;
public:
    Chronometer();

    /**
     * The time passing between tic() and toc() will be
     * counted as one \f$w_i\f$.
     */
    void tic();

    /**
     * see tic().
     */
    void toc();

    /**
     * Reset the start time of the current cycle and flush all the
     * working times \f$w_i\f$, @returns \f$f\f$.
     */
    double nextCycle();

    /**
     * Reset the start time of the current cycle and flush all the
     * woking times \f$w_i\f$, @returns (\f$f\f$,\f$t\f$).
     */
    void nextCycle(long long *cycleLength, long long *workLength);

    static long long timeUSec();

private:
    long long _cycleStart;
    long long _workIntervalStart;
    long long _workLength;

    /**
     * @returns a timestamp from a high resolution timer. There are no
     * warranties for this method except that consecutive calls will
     * return strictly monotonic increasing values.
     */
    long long time() const;

    void startCycle();
};

};

#endif
