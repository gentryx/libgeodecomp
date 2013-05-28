#ifndef LIBGEODECOMP_IO_REMOTESTEERER_ACTION_H
#define LIBGEODECOMP_IO_REMOTESTEERER_ACTION_H

#include <libgeodecomp/io/remotesteerer/commandserver.h>
#include <libgeodecomp/io/remotesteerer/steererdata.h>

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
    typedef StringOps::StringVec StringVec;

    Action(const std::string& myHelpMessage, const std::string& myKey) :
        myHelpMessage(myHelpMessage),
        myKey(myKey)
    {}

    virtual void operator()(const StringOps::StringVec& parameters, CommandServer<CELL_TYPE> *server) = 0;

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
    std::string myHelpMessage;
    std::string myKey;
};

}

}

#endif
