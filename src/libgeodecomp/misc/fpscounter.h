#ifndef LIBGEODECOMP_MISC_FPSCOUNTER_H
#define LIBGEODECOMP_MISC_FPSCOUNTER_H

#include <libgeodecomp/misc/chronometer.h>

namespace LibGeoDecomp {

/**
 * Small utility class which keeps track of the averagy frames per
 * second (FPS). Useful for interactive Writers, e.g. QtWidgetWriter.
 */
class FPSCounter
{
public:
    FPSCounter() :
        startTime(Chronometer::timeUSec()),
        frames(0)
    {}

    void incFrames()
    {
        ++frames;
    }

    long long getFrames()
    {
        return frames;
    }

    double fps()
    {
        long long time = Chronometer::timeUSec();
        long long delta = time - startTime;
        // keep two digits
        long buf = 100 * frames * 1000000.0 / delta;
        return buf * 0.01;
    }

private:
    volatile long long startTime;
    volatile long long frames;
};

}

#endif
