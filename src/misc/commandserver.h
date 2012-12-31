#ifndef _libgeodecomp_misc_commandserver_h_
#define _libgeodecomp_misc_commandserver_h_
#include <iostream>
#include <string>
#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread.hpp>
#include <cerrno>
#include <stdexcept>
#include <map>

#define MAXLENGTH 1024

using boost::asio::ip::tcp;

namespace LibGeoDecomp {
/*
 *
 */
class CommandServer
{
public:
    class Session;

    typedef boost::shared_ptr<tcp::socket> socket_ptr;
    typedef std::vector<std::string> stringVec;
    //typedef void (*fptr)(stringVec,
    //        RemoteSteerer<CELL_TYPE, CONTROL>*, void*);
    typedef std::map<std::string, void(*)(stringVec, Session*, void*)> functionMap;

    /**
     *
     */
    static void splitString(std::string input, stringVec& output)
    {
        boost::split(output, input,
                     boost::is_any_of(" "), boost::token_compress_on);
    }

    /*
     *
     */
    class Session{
    public:
        /*
         *
         */
        Session (socket_ptr _socket, functionMap _commandMap, void* _userData) :
            socket(_socket),
            commandMap(_commandMap),
            userData(_userData)
        {
        }

        /**
         *
         */
        size_t sendMessage(std::string message)
        {
            size_t bytes;
            boost::system::error_code ec;
            bytes = boost::asio::write(*socket, boost::asio::buffer(message),
                                       boost::asio::transfer_all(), ec);
            if (ec)
            {
                printError(ec);
            }
            return bytes;
        }

        /**
         *
         */
        void runSession ()
        {
            for (;;)
            {
                boost::array<char, MAXLENGTH> buf;
                boost::system::error_code ec;
                std::string input;
                stringVec parameter;
                std::string message;
                typename functionMap::iterator it;

                size_t length = socket->read_some(boost::asio::buffer(buf), ec);
                if (ec == boost::asio::error::eof)
                {
                    std::cout << "client closed connection" << std::endl;
                    return; // Connection closed cleanly by peer.
                }
                else if (ec)
                {
                    printError(ec);
                    return;
                }

                input.assign(buf.data(), length - 1);
                splitString(input, parameter);
                switch(parameter.size())
                {
                case 0:
                    message = "no command\n";
                    sendMessage(message);
                    break;
                default:
                    it = commandMap.find(parameter[0]);
                    if (it != commandMap.end())
                    {
                        //(this->*(commandMap[parameter[0]]))(parameter, this,
                        //                                    this->userData);
                        commandMap[parameter[0]](parameter, this, userData);
                    }
                    else
                    {
                        message = "command not found: " + parameter[0] + "\n";
                        message += "try \"help\"\n";
                        sendMessage(message);
                    }
                }
            }
        }

        /*
         *
         */
        functionMap getMap()
        {
            return commandMap;
        }

    private:
        /**
         *
         */
        socket_ptr socket;
        functionMap commandMap;
        void *userData;
    };

    class Server
    {
    public:
        /*
         *
         */
        Server (int _port, functionMap* _commandMap, void* _userData) :
            port(_port),
            commandMap(*_commandMap),
            userData(_userData)
        {
        }

        /**
         *
         */
        int runServer()
        {
            try
            {
                boost::asio::io_service io_service;
                tcp::acceptor acc(io_service, tcp::endpoint(tcp::v4(), port));
                boost::system::error_code ec;

                std::cout << "server started" << std::endl;

                for (;;)
                {
                    socket = socket_ptr(new tcp::socket(io_service));
                    acc.accept(*socket, ec);
                    if (ec)
                    {
                        printError(ec);
                    }
                    else
                    {
                        std::cout << "client connected" << std::endl;
                        session = new Session(socket, commandMap, userData);
                        session->runSession();
                        delete session;
                        session = NULL;
                    }
                }
            }
            catch (std::exception& e)
            {
                std::cerr << "Exception: " << e.what() << "\n";
                return 1;
            }
        }

        /**
         *
         */
        int startServer()
        {
            server_thread = boost::shared_ptr<boost::thread>
                (new boost::thread(&Server::runServer, this));

            return 0;
        }

        /*
         *
         */
        Session* session;

    private:
        /**
         *
         */
        int port;
        boost::shared_ptr<boost::thread> server_thread;
        socket_ptr socket;
        functionMap commandMap;
        void *userData;
    };

    /**
     *
     */
    static void printError(boost::system::error_code& ec)
    {
        std::cerr << "error: " << ec.message() << std::endl;
    }

};

}

#endif
