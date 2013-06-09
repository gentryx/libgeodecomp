#ifndef LIBGEODECOMP_IO_REMOTESTEERER_HANDLER_H
#define LIBGEODECOMP_IO_REMOTESTEERER_HANDLER_H

#include <libgeodecomp/io/remotesteerer/pipe.h>

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

/**
 * A Handler is similar to an Action in the sense it will act upon
 * commands invoked by the user. Unlike Actions, a Handler has direct
 * access to the grid, but will be triggered by the RemoteSteerer, not
 * via the CommandServer.
 */
template<typename CELL_TYPE>
class Handler
{
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef GridBase<CELL_TYPE, Topology::DIM> GridType;

    Handler(const std::string& myKey) :
        myKey(myKey)
    {}

    /**
     * Hanlde command specified by parameters; returns true if command
     * was processed, false if command should be requeued. This is
     * useful if for instance not all relevant cells were available in
     * validRegion.
     */
    virtual bool operator()(const StringVec& parameters, Pipe& pipe, GridType *grid, const Region<Topology::DIM>& validRegion, unsigned step) = 0;

    virtual std::string key()
    {
        return myKey;
    }

private:
    std::string myKey;
};

}

}

#endif
