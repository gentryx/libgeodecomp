#ifndef LIBGEODECOMP_IO_REMOTESTEERER_COMMANDSERVER_H
#define LIBGEODECOMP_IO_REMOTESTEERER_COMMANDSERVER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/io/remotesteerer/action.h>
#include <libgeodecomp/io/remotesteerer/getaction.h>
#include <libgeodecomp/io/remotesteerer/interactor.h>
#include <libgeodecomp/io/remotesteerer/waitaction.h>
#include <libgeodecomp/misc/stringops.h>

// Kill some warnings in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4711 )
#endif

#include <cerrno>
#include <iostream>
#include <string>
#include <stdexcept>
#include <map>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

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
    typedef std::map<std::string, typename SharedPtr<Action<CELL_TYPE> >::Type > ActionMap;

    /**
     * This helper class lets us and the user safely close the
     * CommandServer's network service, which is nice as it is using
     * blocking IO and it's a major PITA to cancel that.
     */
    // fixme: move to dedicated file
    class QuitAction : public Action<CELL_TYPE>
    {
    public:
        explicit QuitAction(bool *continueFlag) :
            Action<CELL_TYPE>("quit", "Terminates the CommandServer and closes its socket."),
            continueFlag(continueFlag)
        {}

        void operator()(const StringVec& parameters, Pipe& pipe)
        {
            LOG(INFO, "QuitAction called");
            *continueFlag = false;
        }

    private:
        bool *continueFlag;
    };

    /**
     * This class is just a NOP, which may be used by the client to
     * retrieve new steering feedback. This can't happen automatically
     * as the CommandServer's listener thread blocks for input from
     * the client.
     */
    // fixme: move to dedicated file
    class PingAction : public Action<CELL_TYPE>
    {
    public:
        using Action<CELL_TYPE>::key;

        PingAction() :
            Action<CELL_TYPE>("ping", "wake the CommandServer, useful to retrieve a new batch of feedback"),
            c(0)
        {}

        void operator()(const StringVec& parameters, Pipe& pipe)
        {
            // // Do only reply if there is no feedback already waiting.
            // // This is useful if the client is using ping to keep us
            // // alive, but can only savely read back one line in
            // // return. In that case this stragety avoids a memory leak
            // // in our write buffer.
            // if (pipe.copySteeringFeedback().size() == 0) {
                pipe.addSteeringFeedback("pong " + StringOps::itoa(++c));
            // }
        }

    private:
        int c;
    };

    CommandServer(
        int port,
        SharedPtr<Pipe>::Type pipe) :
        port(port),
        pipe(pipe)
        // serverThread(&CommandServer::runServer, this)
    {
        addAction(new QuitAction(&continueFlag));
        addAction(new PingAction);
        addAction(new WaitAction<CELL_TYPE>);

        // The thread may take a while to start up. We need to wait
        // here so we don't try to clean up in the d-tor before the
        // thread has created anything.

        // fixme
        // boost::unique_lock<boost::mutex> lock(mutex);
        // while(!acceptor) {
        //     threadCreationVar.wait(lock);
        // }
    }

    ~CommandServer()
    {
        // fixme
        // signalClose();
        // LOG(DBG, "CommandServer waiting for network thread");
        // serverThread.join();
    }

    /**
     * Sends a message back to the end user. This is the primary way
     * for (user-defined) Actions to give feedback.
     */
    void sendMessage(const std::string& message)
    {
        LOG(DBG, "CommandServer::sendMessage(" << message << ")");
        // fixme
        // boost::system::error_code errorCode;
        // boost::asio::write(
        //     *socket,
        //     boost::asio::buffer(message),
        //     boost::asio::transfer_all(),
        //     errorCode);

        // if (errorCode) {
        //     LOG(WARN, "CommandServer::sendMessage encountered " << errorCode.message());
        // }
    }

    /**
     * A convenience method to send a string to a CommandServer
     * listeting on the given host/port combination.
     */
    static void sendCommand(const std::string& command, int port, const std::string& host = "127.0.0.1")
    {
        sendCommandWithFeedback(command, 0, port, host);
    }

    static StringVec sendCommandWithFeedback(const std::string& command, int feedbackLines, int port, const std::string& host = "127.0.0.1")
    {
        LOG(DBG, "CommandServer::sendCommandWithFeedback(" << command << ", port = " << port << ", host = " << host << ")");
        Interactor interactor(command, feedbackLines, false, port, host);
        interactor();
        return interactor.feedback();
        // fixme
        // boost::asio::io_service ioService;
        // tcp::resolver resolver(ioService);
        // tcp::resolver::query query(host, StringOps::itoa(port));
        // tcp::resolver::iterator endpointIterator = resolver.resolve(query);
        // tcp::socket socket(ioService);
        // boost::asio::connect(socket, endpointIterator);
        // boost::system::error_code errorCode;

        // boost::asio::write(
        //     socket,
        //     boost::asio::buffer(command),
        //     boost::asio::transfer_all(),
        //     errorCode);

        // if (errorCode) {
        //     LOG(WARN, "error while writing to socket: " << errorCode.message());
        // }

        StringVec ret;

        for (int i = 0; i < feedbackLines; ++i) {
            // fixme
            // boost::asio::streambuf buf;
            // boost::system::error_code errorCode;

            LOG(DBG, "CommandServer::sendCommandWithFeedback() reading line");

            // fixme
            // std::size_t length = boost::asio::read_until(socket, buf, '\n', errorCode);
            // if (errorCode) {
            //     LOG(WARN, "error while writing to socket: " << errorCode.message());
            // }

            // // purge \n at end of line
            // if (length) {
            //     length -= 1;
            // }

            // std::istream lineBuf(&buf);
            // std::string line(length, 'X');
            // lineBuf.read(&line[0], length);
            // ret << line;
        }

        return ret;
    }

    /**
     * Register a server-side callback for handling user input. The
     * CommandServer will assume ownership of the action and free its
     * memory upon destruction.
     */
    void addAction(Action<CELL_TYPE> *action)
    {
        actions[action->key()] = typename SharedPtr<Action<CELL_TYPE> >::Type(action);
    }

private:
    int port;
    SharedPtr<Pipe>::Type pipe;
    // fixme
    // boost::asio::io_service ioService;
    // SharedPtr<tcp::acceptor>::Type acceptor;
    // SharedPtr<tcp::socket>::Type socket;
    // boost::thread serverThread;
    // boost::condition_variable threadCreationVar;
    // boost::mutex mutex;
    ActionMap actions;
    bool continueFlag;

    void runSession()
    {
        // fixme
        // for (;;) {
        //     boost::array<char, 1024> buf;
        //     boost::system::error_code errorCode;
        //     LOG(DBG, "CommandServer::runSession(): reading");
        //     std::size_t length = socket->read_some(boost::asio::buffer(buf), errorCode);
        //     LOG(DBG, "CommandServer::runSession(): read " << length << " bytes");

        //     if (length > 0) {
        //         std::string input(buf.data(), length);
        //         handleInput(input);
        //     }

        //     if (errorCode == boost::asio::error::eof) {
        //         LOG(INFO, "CommandServer::runSession(): client closed connection");
        //         return;
        //     }

        //     if (errorCode) {
        //         LOG(WARN, "CommandServer::runSession encountered " << errorCode.message());
        //     }

        //     if (!socket->is_open()) {
        //         return;
        //     }

        //     StringVec feedback = pipe->retrieveSteeringFeedback();
        //     for (StringVec::iterator i = feedback.begin();
        //          i != feedback.end();
        //          ++i) {
        //         LOG(DBG, "CommandServer::runSession sending »" << *i << "«");
        //         sendMessage(*i + "\n");
        //     }
        // }
    }

    void handleInput(const std::string& input)
    {
        LOG(DBG, "CommandServer::handleInput(" << input << ")");
        StringVec lines = StringOps::tokenize(input, "\n");
        std::string zeroString("x");
        zeroString[0] = 0;

        for (StringVec::iterator iter = lines.begin();
             iter != lines.end();
             ++iter) {
            StringVec parameters = StringOps::tokenize(*iter, " \n\r");

            if (*iter == zeroString) {
                // silently ignore strings containing a single 0
                continue;
            }

            if (parameters.size() == 0) {
                pipe->addSteeringFeedback("no command given");
                continue;
            }

            std::string command = pop_front(parameters);
            if (actions.count(command) == 0) {
                std::string message = "command not found: " + command;
                LOG(WARN, message);
                pipe->addSteeringFeedback(message);
                pipe->addSteeringFeedback("try \"help\"");
            } else {
                (*actions[command])(parameters, *pipe);
            }
        }
    }

    int runServer()
    {
        // fixme
        // try {
        //     // Thread-aware initialization, allows c-tor to exit safely.
        //     {
        //         boost::unique_lock<boost::mutex> lock(mutex);
        //         acceptor.reset(new tcp::acceptor(ioService, tcp::endpoint(tcp::v4(), port)));
        //     }
        //     threadCreationVar.notify_one();

        //     boost::system::error_code errorCode;
        //     continueFlag = true;

        //     while (continueFlag) {
        //         LOG(DBG, "CommandServer: waiting for new connection");
        //         socket.reset(new tcp::socket(ioService));
        //         acceptor->accept(*socket, errorCode);

        //         if (errorCode) {
        //             LOG(WARN, "CommandServer::runServer() encountered " << errorCode.message());
        //         } else {
        //             LOG(INFO, "CommandServer: client connected");
        //             runSession();
        //             LOG(INFO, "CommandServer: client disconnected");
        //         }
        //     }
        // }
        // catch (std::exception& e) {
        //     LOG(FATAL, "CommandServer::runServer() listening on port " << port
        //         << " caught exception " << e.what() << ", exiting");
        //     return 1;
        // }

        return 0;
    }

    void signalClose()
    {
        sendCommand("quit", port);
    }
};


}

}

#endif
