#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef LIBGEODECOMP_IO_REMOTESTEERER_H
#define LIBGEODECOMP_IO_REMOTESTEERER_H

#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/misc/commandserver.h>
#include <libgeodecomp/misc/remotesteererhelper.h>
#include <libgeodecomp/mpilayer/typemaps.h>
#include <mpi.h>

namespace LibGeoDecomp {

using namespace RemoteSteererHelper;

/**
 * The RemoteSteerer allows the user to control a parallel simulation
 * from a single network connection.
 */
template<typename CELL_TYPE,
         typename STEERER_DATA_TYPE=SteererData<CELL_TYPE>,
         typename CONTROL=DefaultSteererControl<CELL_TYPE> >
class RemoteSteerer : public Steerer<CELL_TYPE>
{
public:
    typedef typename Steerer<CELL_TYPE>::Topology Topology;
    typedef typename Steerer<CELL_TYPE>::GridType GridType;

    RemoteSteerer(
        const unsigned& period,
        int port,
        CommandServer::FunctionMap commandMap = getDefaultMap(),
        STEERER_DATA_TYPE *steererData = 0,
        const MPI::Intracomm& comm = MPI::COMM_WORLD) :
        Steerer<CELL_TYPE>(period),
        steererData(steererData),
        comm(comm)
    {
        if (comm.Get_rank() != 0) {
            return;
        }

        std::cout << "RemoteSteerer(), steererData->dataAccessors.size() = " << steererData->dataAccessors.size() << "\n";
        server = new CommandServer::Server(port, commandMap, steererData);
        server->startServer();
    }

    virtual ~RemoteSteerer()
    {
        // fixme: avoid delete. do we really need defaultData/Map?
        delete server;
    }

    static CommandServer::FunctionMap getDefaultMap()
    {
        CommandServer::FunctionMap defaultMap;
        defaultMap["help"] = helpFunction;
        defaultMap["get"] = getFunction;
        defaultMap["set"] = setFunction;
        defaultMap["finish"] = finishFunction;

        return defaultMap;
    }

    virtual void nextStep(
            GridType *grid,
            const Region<Topology::DIM>& validRegion,
            unsigned step) {
        RemoteSteererHelper::MessageBuffer *msgBuffer;
        if (comm.Get_rank() == 0) {
            msgBuffer = new RemoteSteererHelper::MessageBuffer(comm, server->session);
        } else {
            msgBuffer = new RemoteSteererHelper::MessageBuffer(comm, NULL);
        }
        std::cout << "RemoteSteerer::nextStep(), steererData->dataAccessors().size() = " << steererData->dataAccessors.size() << "\n";
        CONTROL()(grid, validRegion, step, msgBuffer, steererData, comm);
        if (comm.Get_size() > 1) {
            msgBuffer->collectMessages();
        }
        delete msgBuffer;
    }

    static void helpFunction(std::vector<std::string> stringVec,
                            CommandServer::Session *session,
                            void *data) {
        CommandServer::FunctionMap commandMap = session->getMap();
        for (CommandServer::FunctionMap::iterator it = commandMap.begin();
             it != commandMap.end(); ++ it) {
            std::string command = (*it).first;
            if (command.compare("help") != 0) {
                session->sendMessage(command + ":\n");
                std::vector<std::string> parameter;
                parameter.push_back((*it).first);
                parameter.push_back("help");
                (it->second)(parameter, session, data);
            }
        }
    }

    /**
     *
     */
    static void getFunction(std::vector<std::string> stringVec,
                            CommandServer::Session *session,
                            void *data) 
    {
        STEERER_DATA_TYPE *sdata = (STEERER_DATA_TYPE*) data;
        int x = 0;
        int y = 0;
        int z = -1;
        std::string helpMsg = "    Usage: get <x> <y> [<z>]\n";
        helpMsg += "          adds a get action to the list\n";
        helpMsg += "          <x>, <y> and <z> must be integers\n";
        helpMsg += "          for execution use finish\n";
        try {
            if (stringVec.at(1).compare("help") == 0) {
                session->sendMessage(helpMsg);
                return;
            }
            x = mystrtoi(stringVec.at(1).c_str());
            y = mystrtoi(stringVec.at(2).c_str());
            if (stringVec.size() > 3) {
                z = mystrtoi(stringVec.at(3).c_str());
            }
        }
        catch (std::exception& e) {
            session->sendMessage(helpMsg);
            return;
        }
        sdata->getX.push_back(x);
        sdata->getY.push_back(y);
        sdata->getZ.push_back(z);
    }

    static void setFunction(std::vector<std::string> stringVec,
                            CommandServer::Session *session,
                            void *data) {
        STEERER_DATA_TYPE *sdata = (STEERER_DATA_TYPE*) data;
        int x = 0;
        int y = 0;
        int z = -1;
        std::string value;
        std::string var;
        std::string helpMsg = "    Usage: set <variable> <value> <x> <y> [<z>]\n";
        helpMsg += "          add a set action to the list\n";
        helpMsg += "          <x>, <y> and <z> must be integers\n";
        helpMsg += "          for execution use finish\n";
        try {
            if (stringVec.at(1).compare("help") == 0) {
                session->sendMessage(helpMsg);
                return;
            }
            var = stringVec.at(1);
            value = stringVec.at(2);
            x = mystrtoi(stringVec.at(3).c_str());
            y = mystrtoi(stringVec.at(4).c_str());
            if (stringVec.size() > 5) {
                z = mystrtoi(stringVec.at(5).c_str());
            }
        }
        catch (std::exception& e) {
            session->sendMessage(helpMsg);
            return;
        }
        sdata->setX.push_back(x);
        sdata->setY.push_back(y);
        sdata->setZ.push_back(z);
        sdata->val.push_back(value);
        sdata->var.push_back(var);
    }

    static void finishFunction(std::vector<std::string> stringVec,
                            CommandServer::Session *session,
                            void *data) {
        STEERER_DATA_TYPE *sdata = (STEERER_DATA_TYPE*) data;
        std::string helpMsg = "    Usage: finish\n";
        helpMsg += "          sets lock and waits until a step is finished\n";
        helpMsg += "          has to be used to start default set and get operations\n";
        if (stringVec.size() > 1) {
            session->sendMessage(helpMsg);
            return;
        }
        session->sendMessage("waiting until next step finished ...\n");
        sdata->finishMutex.unlock();
        sdata->waitMutex.lock();
    }

private:
    CommandServer::Server *server;
    // fixme: use shared_ptr here and in command server
    STEERER_DATA_TYPE *steererData;
    MPI::Intracomm comm;

};

}

#endif
#endif
