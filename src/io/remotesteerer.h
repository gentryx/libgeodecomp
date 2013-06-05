#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef LIBGEODECOMP_IO_REMOTESTEERER_H
#define LIBGEODECOMP_IO_REMOTESTEERER_H

#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/io/remotesteerer/commandserver.h>
#include <libgeodecomp/io/remotesteerer/handler.h>
#include <libgeodecomp/io/remotesteerer/pipe.h>
#include <libgeodecomp/mpilayer/typemaps.h>
#include <libgeodecomp/mpilayer/mpilayer.h>

namespace LibGeoDecomp {

using namespace RemoteSteererHelpers;

/**
 * The RemoteSteerer allows the user to control a parallel simulation
 * from a single network connection (on the "connection node"). It
 * employs a two-way callback scheme to enable convenient extension of
 * its functionality:
 *
 * - On the connection node Action objects can be invoked via the
 *   CommandServer to provide the user with a richer terminal
 *   interface.
 *
 * - On the compute nodes Handler objects will then execute the user
 *   requests on simulation data. Handlers and Actions may communicate
 *   via a asynchronous message buffer.
 *
 * Keep in mind that the connection node will generally double as an
 * execution node.
 */
template<typename CELL_TYPE>
class RemoteSteerer : public Steerer<CELL_TYPE>
{
public:
    typedef StringOps::StringVec StringVec;
    typedef typename Steerer<CELL_TYPE>::Topology Topology;
    typedef typename Steerer<CELL_TYPE>::GridType GridType;
    typedef SuperMap<std::string, boost::shared_ptr<Handler<CELL_TYPE> > > HandlerMap;

    RemoteSteerer(
        unsigned period,
        int port,
        int root = 0,
        MPI::Comm *communicator = &MPI::COMM_WORLD) :
        Steerer<CELL_TYPE>(period),
        port(port),
        pipe(new Pipe(root, communicator))
    {
        if (MPILayer(communicator).rank() == root) {
            commandServer.reset(new CommandServer<CELL_TYPE>(port, pipe));
        }
    }

    virtual void nextStep(
        GridType *grid,
        const Region<Topology::DIM>& validRegion,
        unsigned step)
    {
        pipe->sync();
        StringVec steeringRequests = pipe->retrieveSteeringRequests();

        for (StringVec::iterator i = steeringRequests.begin();
             i != steeringRequests.end();
             ++i) {
            StringVec parameters = StringOps::tokenize(*i, " ");
            std::string command = parameters.pop_front();

            if (handlers.count(command) > 0) {
                int handled = (*handlers[command])(parameters, *pipe, grid, validRegion, step);
                if (!handled) {
                    pipe->addSteeringRequest(*i);
                } else {
                    std::string message = "handler not found: " + command;
                    LOG(Logger::WARN, message);
                    pipe->addSteeringFeedback(message);
                }
            }
        }
    }

    void addAction(Action<CELL_TYPE> *action)
    {
        commandServer->addAction(action);
    }

    void addHandler(Handler<CELL_TYPE> *handler)
    {
        handlers[handler->key()] = boost::shared_ptr<Handler<CELL_TYPE> >(handler);
    }

    void sendCommand(const std::string& command)
    {
        CommandServer<CELL_TYPE>::sendCommand(command, port);
    }

private:
    HandlerMap handlers;
    int port;
    boost::shared_ptr<Pipe> pipe;
    boost::shared_ptr<CommandServer<CELL_TYPE> > commandServer;
};

}

#endif
#endif
