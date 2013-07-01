#ifndef LIBGEODECOMP_IO_REMOTESTEERER_WAITACTION
#define LIBGEODECOMP_IO_REMOTESTEERER_WAITACTION

#include <libgeodecomp/io/remotesteerer/action.h>

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

/**
 * Suspends the command server until feedback is available.
 */
template<typename CELL_TYPE>
class WaitAction : public Action<CELL_TYPE>
{
public:
    WaitAction() :
        Action<CELL_TYPE>(
            "wait",
            "usage: \"wait [n]\", will wait until n lines of feedback from the simulation have been received. If n is omitted, it will wait for 1 line.")
    {}

    void operator()(const StringVec& parameters, Pipe& pipe)
    {
        int lines = 1;
        if (parameters.size() > 0) {
            lines = StringOps::atoi(parameters[0]);
        }

        pipe.waitForFeedback(lines);
    }
};

}

}

#endif
