#ifndef LIBGEODECOMP_IO_REMOTESTEERER_COMMANDSERVER_H
#define LIBGEODECOMP_IO_REMOTESTEERER_COMMANDSERVER_H

#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <cerrno>
#include <iostream>
#include <string>
#include <stdexcept>
#include <map>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/io/remotesteerer/action.h>
#include <libgeodecomp/misc/stringops.h>

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

using boost::asio::ip::tcp;

/**
 * A server which can be reached by TCP (nc, telnet, ...). Its purpose
 * is to do connection handling and parsing of incoming user commands.
 * Action objects can be bound to certain commands and will be
 * invoked. This allows a flexible extension of the CommandServer's
 * functionality by composition, without having to resort to inheritance.
 */
template<typename CELL_TYPE>
class CommandServer
{
public:
    typedef StringOps::StringVec StringVec;
    typedef SuperVector<boost::shared_ptr<Action<CELL_TYPE> > > ActionVec;

    /**
     * This helper class lets us and the user safely close the
     * CommandServer's network service, which is nice as it is using
     * blocking IO and it's a major PITA to cancel that.
     */
    class QuitAction : public Action<CELL_TYPE>
    {
    public:
        QuitAction(bool *continueFlag) :
            Action<CELL_TYPE>("Terminates the CommandServer and closes its socket.", "quit"),
            continueFlag(continueFlag)
        {}

        void operator()(const StringOps::StringVec& parameters, Pipe& pipe)
        {
            LOG(Logger::INFO, "QuitAction called");
            *continueFlag = false;
        }

    private:
        bool *continueFlag;
    };

    /**
     * This Action is helpful if a given user command has to be
     * executed by a Handler on the simulation node (i.e. all commands
     * which work on grid data).
     */
    class PassThroughAction : public Action<CELL_TYPE>
    {
    public:
        using Action<CELL_TYPE>::key;

        PassThroughAction(const std::string& helpMessage, const std::string& key) :
            Action<CELL_TYPE>(helpMessage, key)
        {}

        void operator()(const StringOps::StringVec& parameters, Pipe& pipe)
        {
            pipe.addSteeringRequest(key() + " " + StringOps::join(parameters, " "));
        }
    };

    class GetAction : public PassThroughAction
    {
    public:
        GetAction() :
            PassThroughAction("usage: \"get X Y [Z] MEMBER\", will return member MEMBER of cell at grid coordinate (X, Y, Z) if the model is 3D, or (X, Y) in the 2D case", "get")
        {}
    };

    class SetAction : public PassThroughAction
    {
    public:
        SetAction() :
            PassThroughAction("usage: \"get X Y [Z] MEMBER VALUE\", will set member MEMBER of cell at grid coordinate (X, Y, Z) (if the model is 3D, or (X, Y) in the 2D case) to value VALUE", "set")
        {}
    };

    CommandServer(int port, boost::shared_ptr<Pipe> pipe) :
        port(port),
        pipe(pipe),
        serverThread(&CommandServer::runServer, this)
    {
        addAction(new QuitAction(&continueFlag));
        addAction(new SetAction);
        addAction(new GetAction);

        // The thread may take a while to start up. We need to wait
        // here so we don't try to clean up in the d-tor before the
        // thread has created anything.
        boost::unique_lock<boost::mutex> lock(mutex);
        while(!acceptor) {
            threadCreationVar.wait(lock);
        }
    }

    ~CommandServer()
    {
        signalClose();
        LOG(Logger::DEBUG, "CommandServer waiting for network thread");
        serverThread.join();
    }

    /**
     * Sends a message back to the end user. This is the primary way
     * for (user-defined) Actions to give feedback.
     */
    void sendMessage(const std::string& message)
    {
        boost::system::error_code errorCode;
        boost::asio::write(
            *socket,
            boost::asio::buffer(message),
            boost::asio::transfer_all(),
            errorCode);

        if (errorCode) {
            LOG(Logger::WARN, "CommandServer::sendMessage encountered " << errorCode.message());
        }
    }

    /**
     * A convenience method to send a string to a CommandServer
     * listeting on the given host/port combination.
     */
    static void sendCommand(const std::string& command, const std::string& host, int port)
    {
        boost::asio::io_service ioService;
        tcp::resolver resolver(ioService);
        tcp::resolver::query query(host, StringOps::itoa(port));
        tcp::resolver::iterator endpointIterator = resolver.resolve(query);
        tcp::socket socket(ioService);
        boost::asio::connect(socket, endpointIterator);
        boost::system::error_code errorCode;

        boost::asio::write(
            socket,
            boost::asio::buffer(command),
            boost::asio::transfer_all(),
            errorCode);

        if (errorCode) {
            LOG(Logger::WARN, "error while writing to socket: " << errorCode.message());
        }
    }

    /**
     * Register a server-side callback for handling user input. The
     * CommandServer will assume ownership of the action and free its
     * memory upon destruction.
     */
    void addAction(Action<CELL_TYPE> *action)
    {
        actions << boost::shared_ptr<Action<CELL_TYPE> >(action);
    }

private:
    int port;
    boost::shared_ptr<Pipe> pipe;
    boost::asio::io_service ioService;
    boost::shared_ptr<tcp::acceptor> acceptor;
    boost::shared_ptr<tcp::socket> socket;
    boost::thread serverThread;
    boost::condition_variable threadCreationVar;
    boost::mutex mutex;
    ActionVec actions;
    bool continueFlag;

    void runSession()
    {
        for (;;) {
            boost::array<char, 1024> buf;
            boost::system::error_code errorCode;
            size_t length = socket->read_some(boost::asio::buffer(buf), errorCode);

            if (length > 0) {
                std::string input(buf.data(), length);
                handleInput(input);
            }

            if (errorCode == boost::asio::error::eof) {
                LOG(Logger::INFO, "CommandServer::runSession(): client closed connection");
                return;
            }

            if (errorCode) {
                LOG(Logger::WARN, "CommandServer::runSession encountered " << errorCode.message());
            }

            if (!socket->is_open()) {
                return;
            }
        }
    }

    void handleInput(const std::string& input)
    {
        LOG(Logger::DEBUG, "Logger::handleInput(" << input << ")");

        StringOps::StringVec lines = StringOps::tokenize(input, "\n");
        for (StringOps::StringVec::iterator iter = lines.begin();
             iter != lines.end();
             ++iter) {
            StringOps::StringVec parameters = StringOps::tokenize(*iter, " ");

            if (parameters.size() == 0) {
                sendMessage("no command given\n");
                continue;
            }

            std::string command = parameters.pop_front();

            bool commandFound = false;
            for (typename ActionVec::iterator i = actions.begin(); i != actions.end(); ++i) {
                if ((*i)->key() == command) {
                    (*(*i))(parameters, *pipe);
                    commandFound = true;
                }
            }

            if (!commandFound) {
                std::string message = "command not found: " + parameters[0];
                LOG(Logger::WARN, message);

                message += "\ntry \"help\"\n";
                sendMessage(message);
            }
        }
    }

    int runServer()
    {
        try {
            // Thread-aware initialization, allows c-tor to exit safely.
            {
                boost::unique_lock<boost::mutex> lock(mutex);
                acceptor.reset(new tcp::acceptor(ioService, tcp::endpoint(tcp::v4(), port)));
            }
            threadCreationVar.notify_one();

            boost::system::error_code errorCode;
            continueFlag = true;

            while (continueFlag) {
                LOG(Logger::DEBUG, "CommandServer waiting for new connection");
                socket.reset(new tcp::socket(ioService));
                acceptor->accept(*socket, errorCode);

                if (errorCode) {
                    LOG(Logger::WARN, "CommandServer::runServer() encountered " << errorCode.message());
                } else {
                    LOG(Logger::INFO, "CommandServer: client connected");
                    runSession();
                }
            }
        }
        catch (std::exception& e) {
            LOG(Logger::FATAL, "CommandServer::runServer() caught exception " << e.what() << ", exiting");
            return 1;
        }

        return 0;
    }

    void signalClose()
    {
        sendCommand("quit", "127.0.0.1", port);
    }
};


}

}

#endif
