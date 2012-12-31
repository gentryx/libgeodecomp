#ifndef _libgeodecomp_io_remotesteerer_h_
#define _libgeodecomp_io_remotesteerer_h_

#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/misc/commandserver.h>
#include <libgeodecomp/misc/remotesteererhelper.h>

namespace LibGeoDecomp {

using namespace RemoteSteererHelper;

/*
 *
 */
template<typename CELL_TYPE, int DIM,
         typename CONTROL = DefaultSteererControl<CELL_TYPE, DIM> >
class RemoteSteerer : public Steerer<CELL_TYPE>
{
public:
    /**
     *
     */
    typedef typename Steerer<CELL_TYPE>::Topology Topology;
    typedef typename Steerer<CELL_TYPE>::GridType GridType;

    /**
     * if you use this constructo, you should also use your own SteererControl
     */
    RemoteSteerer(const unsigned& _period, int _port,
                  CommandServer::functionMap* _commandMap,
                  void* _userData) :
        Steerer<CELL_TYPE>(_period),
        userData(_userData)
    {
        server = new CommandServer::Server(_port, _commandMap, _userData);
        server->startServer();
    }

    /**
     *
     */
    RemoteSteerer(const unsigned& _period, int _port,
                  DataAccessor<CELL_TYPE>** _dataAccessors,
                  int _numVars,
                  CommandServer::functionMap* _commandMap = NULL,
                  void* _userData = NULL) :
        Steerer<CELL_TYPE>(_period),
        dataAccessors(_dataAccessors),
        numVars(_numVars),
        userData(_userData),
        defaultData(NULL),
        defaultMap(NULL)
    {
        if (_userData == NULL)
        {
            defaultData = new SteererData<CELL_TYPE>(_dataAccessors, _numVars);
            userData = reinterpret_cast<void*>(defaultData);
        }
        if (_commandMap == NULL)
        {
            defaultMap = getDefaultMap();
            _commandMap = defaultMap;
        }
        server = new CommandServer::Server(_port, _commandMap, userData);
        server->startServer();
    }

    /**
     *
     */
    virtual ~RemoteSteerer()
    {
        delete server;
        if (defaultData != NULL)
        {
            delete defaultData;
        }
        if (defaultMap != NULL)
        {
            delete defaultMap;
        }
    }

    static CommandServer::functionMap* getDefaultMap()
    {
        CommandServer::functionMap* defaultMap = new CommandServer::functionMap();
        (*defaultMap)["help"] = helpFunction;
        (*defaultMap)["get"] = getFunction;
        (*defaultMap)["set"] = setFunction;
        (*defaultMap)["finish"] = finishFunction;

        return defaultMap;
    }

    /**
     *
     */
    virtual void nextStep(
        GridType *grid,
        const Region<Topology::DIMENSIONS>& validRegion,
        const unsigned& step)
    {
        if (server->session != NULL)
        {
            CONTROL()(grid, validRegion, step, server->session, userData);
        }
    }

private:
    /**
     *
     */
    DataAccessor<CELL_TYPE> **dataAccessors;
    int numVars;
    CommandServer::Server* server;
    void *userData;
    SteererData<CELL_TYPE>* defaultData;
    CommandServer::functionMap* defaultMap;

    /**
     *
     */
    static void helpFunction(std::vector<std::string> stringVec,
                            CommandServer::Session *session,
                            void *data)
    {
        CommandServer::functionMap commandMap = session->getMap();
        for (CommandServer::functionMap::iterator it = commandMap.begin();
             it != commandMap.end(); ++ it)
        {
            std::string command = (*it).first;
            if (command.compare("help") != 0)
            {
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
        SteererData<CELL_TYPE> *sdata = (SteererData<CELL_TYPE>*) data;
        int x = 0;
        int y = 0;
        int z = -1;
        std::string help_msg = "    Usage: get <x> <y> [<z>]\n";
        help_msg += "          adds a get action to the list\n";
        help_msg += "          <x>, <y> and <z> must be integers\n";
        try
        {
            if (stringVec.at(1).compare("help") == 0)
            {
                session->sendMessage(help_msg);
                return;
            }
            x = mystrtoi(stringVec.at(1).c_str());
            y = mystrtoi(stringVec.at(2).c_str());
            if (stringVec.size() > 3)
            {
                z = mystrtoi(stringVec.at(3).c_str());
            }
        }
        catch (std::exception& e)
        {
            session->sendMessage(help_msg);
            return;
        }
        sdata->get_x.push_back(x);
        sdata->get_y.push_back(y);
        sdata->get_z.push_back(z);
        sdata->getValue_mutex.unlock();
    }

    /**
     *
     */
    static void setFunction(std::vector<std::string> stringVec,
                            CommandServer::Session *session,
                            void *data)
    {
        SteererData<CELL_TYPE> *sdata = (SteererData<CELL_TYPE>*) data;
        int x = 0;
        int y = 0;
        int z = -1;
        std::string value;
        std::string var;
        std::string help_msg = "    Usage: set <variable> <value> <x> <y> [<z>]\n";
        help_msg += "          add a set action to the list\n";
        help_msg += "          <x>, <y> and <z> must be integers\n";
        try
        {
            if (stringVec.at(1).compare("help") == 0)
            {
                session->sendMessage(help_msg);
                return;
            }
            var = stringVec.at(1);
            value = stringVec.at(2);
            x = mystrtoi(stringVec.at(3).c_str());
            y = mystrtoi(stringVec.at(4).c_str());
            if (stringVec.size() > 5)
            {
                z = mystrtoi(stringVec.at(5).c_str());
            }
        }
        catch (std::exception& e)
        {
            session->sendMessage(help_msg);
            return;
        }
        sdata->set_x.push_back(x);
        sdata->set_y.push_back(y);
        sdata->set_z.push_back(z);
        sdata->val.push_back(value);
        sdata->var.push_back(var);
        sdata->setValue_mutex.unlock();
    }

    /**
     *
     */
    static void finishFunction(std::vector<std::string> stringVec,
                            CommandServer::Session *session,
                            void *data)
    {
        SteererData<CELL_TYPE> *sdata = (SteererData<CELL_TYPE>*) data;
        std::string help_msg = "    Usage: finish\n";
        help_msg += "          sets lock and waits until a step is finished\n";
        if (stringVec.size() > 1)
        {
            session->sendMessage(help_msg);
            return;
        }
        session->sendMessage("waiting until next step finished ...\n");
        sdata->finish_mutex.unlock();
        sdata->wait_mutex.lock();
    }
};

}

#endif
