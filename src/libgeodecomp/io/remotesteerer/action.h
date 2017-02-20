#ifndef LIBGEODECOMP_IO_REMOTESTEERER_ACTION_H
#define LIBGEODECOMP_IO_REMOTESTEERER_ACTION_H

#include <libgeodecomp/io/remotesteerer/pipe.h>

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

template<typename CELL_TYPE>
class CommandServer;

/**
 * This class can be used as a base for functors which extend the
 * functionality of the CommandServer within the RemoteSteerer. After
 * an Action has been registered, it can be invoked by the user.
 */
template<typename CELL_TYPE>
class Action
{
public:
    Action(const std::string& myKey, const std::string& myHelpMessage) :
        myKey(myKey),
        myHelpMessage(myHelpMessage)
    {}

    virtual ~Action()
    {}

    virtual void operator()(const StringVec& parameters, Pipe& pipe) = 0;

    virtual std::string helpMessage()
    {
        return myHelpMessage;
    }

    /**
     * The key is used to determine whether the Action is responsible
     * for a given command (i.e. the command name equals the Action's
     * key).
     */
    virtual std::string key()
    {
        return myKey;
    }

private:
    std::string myKey;
    std::string myHelpMessage;
};

}

}

#endif
