#ifndef LIBGEODECOMP_IO_REMOTESTEERER_INTERACTOR_H
#define LIBGEODECOMP_IO_REMOTESTEERER_INTERACTOR_H

#include <boost/asio.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/misc/stringvec.h>
#include <libgeodecomp/misc/stringvec.h>
#include <libgeodecomp/misc/stringops.h>

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

using boost::asio::ip::tcp;

/**
 * This class can be used to encapsulate synchronous/asynchronous
 * communication with a RemoteSteerer's CommandServer. This is useful
 * for instance for writing unit tests where manually creating sockets
 * and syncing a thread is tedious.
 */
class Interactor
{
public:

    Interactor(
        const std::string& command,
        int feedbackLines,
        bool threaded,
        int port,
        const std::string& host = "127.0.0.1") :
        command(command),
        feedbackLines(feedbackLines),
        port(port),
        host(host),
        started(false),
        completed(false)
    {
        // The constructor for the thread needs to be in the body as
        // the initializer list doesn't guarantee us that the flags
        // are set in advance. But this is crucial as otherwise the
        // results of the thread might be overwritten.
        if (threaded) {
            thread = boost::thread(ThreadWrapper<Interactor>(this));
        }
    }

    ~Interactor()
    {
        if (thread.joinable()) {
            thread.join();
        }
    }

    void waitForStartup()
    {
        boost::unique_lock<boost::mutex> lock(mutex);
        while(!started) {
            signal.wait(lock);
        }
    }

    void waitForCompletion()
    {
        boost::unique_lock<boost::mutex> lock(mutex);
        while(!completed) {
            signal.wait(lock);
        }
    }

    StringVec feedback()
    {
        return feedbackBuffer;
    }

    int operator()()
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
            boost::asio::buffer(command + "\n"),
            boost::asio::transfer_all(),
            errorCode);

        if (errorCode) {
            LOG(Logger::WARN, "error while writing to socket: " << errorCode.message());
        }

        notifyStartup();

        for (int i = 0; i < feedbackLines; ) {
            // periodically wake up CommandServer to cycle events
            boost::asio::write(
                socket,
                boost::asio::buffer("ping\n"),
                boost::asio::transfer_all(),
                errorCode);
            if (errorCode) {
                LOG(WARN, "Error while pinging " << errorCode);
            }

            boost::asio::streambuf buf;
            boost::system::error_code errorCode;

            LOG(DEBUG, "Interactor::operator() reading...");
            size_t length = boost::asio::read_until(socket, buf, '\n', errorCode);
            if (errorCode) {
                LOG(WARN, "error while reading from socket: " << errorCode.message());
            }

            std::istream lineBuf(&buf);
            std::string line(length, 'X');
            lineBuf.read(&line[0], length);
            line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());

            LOG(DEBUG, "Interactor::operator() read »" << line << "«");

            StringVec tokens = StringOps::tokenize(line, " ");
            if (tokens[0] != "pong") {
                feedbackBuffer << line;
                ++i;
                // sleeping here just to avoid turning the event loop into a busy wait
                usleep(10000);
            }
        }

        notifyCompletion();

        return 0;
    }

private:
    StringVec feedbackBuffer;
    boost::condition_variable signal;
    boost::mutex mutex;
    std::string command;
    int feedbackLines;
    int port;
    std::string host;
    bool started;
    bool completed;
    boost::thread thread;

    void notifyStartup()
    {
        boost::unique_lock<boost::mutex> lock(mutex);
        started = true;
        signal.notify_one();
    }

    void notifyCompletion()
    {
        boost::unique_lock<boost::mutex> lock(mutex);
        completed = true;
        signal.notify_one();
    }

    template<typename DELEGATE>
    class ThreadWrapper
    {
    public:
        ThreadWrapper(DELEGATE *delegate) :
            delegate(delegate)
        {}

        int operator()()
        {
            return (*delegate)();
        }

    private:
        DELEGATE *delegate;
    };
};

}

}

#endif
