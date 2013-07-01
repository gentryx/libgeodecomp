#ifndef LIBGEODECOMP_IO_REMOTESTEERER_PASSTHROUGHACTION
#define LIBGEODECOMP_IO_REMOTESTEERER_PASSTHROUGHACTION

#include <libgeodecomp/io/remotesteerer/action.h>

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

/**
 * This Action is helpful if a given user command has to be
 * executed by a Handler on the simulation node (i.e. all commands
 * which work on grid data).
 */
template<typename CELL_TYPE>
class PassThroughAction : public Action<CELL_TYPE>
{
public:
    using Action<CELL_TYPE>::key;

    PassThroughAction(const std::string& key, const std::string& helpMessage) :
        Action<CELL_TYPE>(key, helpMessage)
    {}

    void operator()(const StringVec& parameters, Pipe& pipe)
    {
        pipe.addSteeringRequest(key() + " " + StringOps::join(parameters, " "));
    }
};

}

}

#endif
