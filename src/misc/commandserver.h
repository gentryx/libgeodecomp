#ifndef LIBGEODECOMP_MISC_COMMANDSERVER_H
#define LIBGEODECOMP_MISC_COMMANDSERVER_H
#include <iostream>
#include <string>
#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread.hpp>
#include <cerrno>
#include <stdexcept>
#include <map>

using boost::asio::ip::tcp;

namespace LibGeoDecomp {

/*
 * a server, which can be reached by tcp(nc, telnet, ...)
 * executes methods, which are bind to a command
 */
// fixme: needs test
// fixme: move to src/io/remotesteerer/commandserver.h
class CommandServer
{
public:
    class Session;

    typedef boost::shared_ptr<tcp::socket> SocketPtr;
    typedef std::vector<std::string> StringVec;
    typedef std::map<std::string, void(*)(StringVec, Session*, void*)> FunctionMap;

    class Session
    {
    public:
        Session(SocketPtr socket, FunctionMap commandMap, void *userData) :
            socket(socket),
            commandMap(commandMap),
            userData(userData)
        { }

        size_t sendMessage(const std::string& message)
        {
            size_t bytes;
            boost::system::error_code errorCode;
            bytes = boost::asio::write(
                *socket,
                boost::asio::buffer(message),
                boost::asio::transfer_all(),
                errorCode);

            if (errorCode) {
                printError(errorCode);
            }
            return bytes;
        }

        void runSession ()
        {
            for (;;) {
                // use a buffer of 1024 Bytes
                boost::array<char, 1024> buf;
                boost::system::error_code errorCode;
                size_t length = socket->read_some(boost::asio::buffer(buf), errorCode);

                if (length > 0) {
                    std::string input(buf.data(), length);
                    handleInput(input);
                }

                if (errorCode == boost::asio::error::eof) {
                    std::cout << "client closed connection" << std::endl;
                    return; // Connection closed cleanly by peer.
                }

                if (errorCode) {
                    printError(errorCode);
                    return;
                }
            }
        }

        FunctionMap getMap()
        {
            return commandMap;
        }

    private:
        SocketPtr socket;
        FunctionMap commandMap;
        void *userData;

        static void splitString(std::string input, StringVec& output, std::string s)
        {
            boost::split(output, input,
                         boost::is_any_of(s), boost::token_compress_on);
        }


        void handleInput(const std::string& input)
        {
            StringVec lines;
            StringVec parameter;
            std::string message;

            splitString(input, lines, std::string("\n"));
            for (StringVec::iterator iter = lines.begin();
                 iter != lines.end(); ++iter) {
                splitString(*iter, parameter, std::string(" "));
                switch(parameter.size()) {
                case 0:
                    message = "no command\n";
                    sendMessage(message);
                    break;
                default:
                    FunctionMap::iterator it = commandMap.find(parameter[0]);
                    if (it != commandMap.end()) {
                        //(this->*(commandMap[parameter[0]]))(parameter, this,
                        //                                    this->userData);
                        commandMap[parameter[0]](parameter, this, userData);
                    } else {
                        message = "command not found: " + parameter[0] + "\n";
                        message += "try \"help\"\n";
                        sendMessage(message);
                    }
                }
            }
        }
    };

    class Server
    {
    public:
        Server(int port, FunctionMap commandMap, void *userData) :
            port(port),
            commandMap(commandMap),
            userData(userData)
        {}

        // fixme: move to constructor?
        int startServer()
        {
            serverThread = boost::shared_ptr<boost::thread>
                (new boost::thread(&Server::runServer, this));

            return 0;
        }

        Session *session;

  private:
        int port;
        boost::shared_ptr<boost::thread> serverThread;
        SocketPtr socket;
        FunctionMap commandMap;
        void *userData;

        int runServer()
        {
            try {
                boost::asio::io_service ioService;
                tcp::acceptor acc(ioService, tcp::endpoint(tcp::v4(), port));
                boost::system::error_code ec;

                std::cout << "Commandserver started" << std::endl;

                for (;;) {
                    socket = SocketPtr(new tcp::socket(ioService));
                    acc.accept(*socket, ec);
                    if (ec) {
                        printError(ec);
                    } else {
                        std::cout << "client connected" << std::endl;
                        session = new Session(socket, commandMap, userData);
                        session->runSession();
                        delete session;
                        session = NULL;
                    }
                }
            }
            catch (std::exception& e) {
                std::cerr << "Exception: " << e.what() << "\n";
                return 1;
            }
        }
    };

    static void printError(boost::system::error_code& ec)
    {
        std::cerr << "error: " << ec.message() << std::endl;
    }

};

}

#endif
