#ifndef LIBGEODECOMP_IO_REMOTESTEERER_H
#define LIBGEODECOMP_IO_REMOTESTEERER_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_BOOST_ASIO
#ifdef LIBGEODECOMP_WITH_THREADS
#ifdef LIBGEODECOMP_WITH_MPI

#include <libgeodecomp/communication/typemaps.h>
#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/io/remotesteerer/commandserver.h>
#include <libgeodecomp/io/remotesteerer/handler.h>
#include <libgeodecomp/io/remotesteerer/gethandler.h>
#include <libgeodecomp/io/remotesteerer/pipe.h>
#include <libgeodecomp/misc/sharedptr.h>

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
    friend class RemoteSteererTest;
    typedef typename Steerer<CELL_TYPE>::SteererFeedback SteererFeedback;
    typedef typename Steerer<CELL_TYPE>::Topology Topology;
    typedef typename Steerer<CELL_TYPE>::GridType GridType;
    typedef std::map<std::string, typename SharedPtr<Handler<CELL_TYPE> >::Type> HandlerMap;
    static const int DIM = Topology::DIM;

    RemoteSteerer(
        unsigned period,
        int port,
        int root = 0,
        MPI_Comm communicator = MPI_COMM_WORLD) :
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
        const Region<DIM>& validRegion,
        const Coord<DIM>& gridDim,
        unsigned step,
        SteererEvent event,
        std::size_t rank,
        bool lastCall,
        SteererFeedback *feedback)
    {
        LOG(DBG, "RemoteSteerer::nextStep(step = " << step << ")");
        pipe->sync();
        StringVec steeringRequests = pipe->retrieveSteeringRequests();

        for (StringVec::iterator i = steeringRequests.begin();
             i != steeringRequests.end();
             ++i) {
            LOG(DBG, "RemoteSteerer::nextStep got" << *i);
            StringVec parameters = StringOps::tokenize(*i, " ");
            std::string command = pop_front(parameters);

            if (handlers.count(command) == 0) {
                std::string message = "handler not found: " + command;
                LOG(Logger::WARN, message);
                pipe->addSteeringFeedback(message);
                continue;
            }

            int handled = (*handlers[command])(parameters, *pipe, grid, validRegion, step);
            // we may have to requeue requrests which could not yet be handled
            if (!handled) {
                pipe->addSteeringRequest(*i);
            }
        }

        pipe->sync();
    }

    void addAction(Action<CELL_TYPE> *action)
    {
        commandServer->addAction(action);
    }

    void addHandler(Handler<CELL_TYPE> *handler)
    {
        handlers[handler->key()] = typename SharedPtr<Handler<CELL_TYPE> >::Type(handler);
    }

    template<typename MEMBER_TYPE>
    void addDataAccessor(DataAccessor<CELL_TYPE, MEMBER_TYPE> *accessor)
    {
        if (commandServer) {
            GetAction<CELL_TYPE> *action = new GetAction<CELL_TYPE>(accessor->name());
            addAction(action);
        }

        typename SharedPtr<DataAccessor<CELL_TYPE, MEMBER_TYPE> >::Type accessorPtr(accessor);
        handlers["get_" + accessor->name()].reset(new GetHandler<CELL_TYPE, MEMBER_TYPE>(accessorPtr));
    }

    void sendCommand(const std::string& command)
    {
        CommandServer<CELL_TYPE>::sendCommand(command, port);
    }

    StringVec sendCommandWithFeedback(const std::string& command, int feedbackLines)
    {
        return CommandServer<CELL_TYPE>::sendCommandWithFeedback(command, feedbackLines, port);
    }

private:
    HandlerMap handlers;
    int port;
    SharedPtr<Pipe>::Type pipe;
    typename SharedPtr<CommandServer<CELL_TYPE> >::Type commandServer;
};

}

#endif
#endif
#endif

#endif
