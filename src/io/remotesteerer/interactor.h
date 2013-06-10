#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_THREADS
#ifndef LIBGEODECOMP_IO_REMOTESTEERER_INTERACTOR_H
#define LIBGEODECOMP_IO_REMOTESTEERER_INTERACTOR_H


#include <boost/asio.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <libgeodecomp/io/remotesteerer/commandserver.h>
#include <libgeodecomp/misc/stringvec.h>

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

using boost::asio::ip::tcp;

/**
 * This class can be used to encapsulate synchronous/asynchronous
 * communication with a RemoteSteerer's CommandServer. This is useful
 * for instance for writing unit tests where manually creating sockets
 * and syncing a thread is tedious.
 */
template<typename CELL_TYPE>
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
            thread = boost::thread(ThreadWrapper<Interactor<CELL_TYPE> >(this));
        }
    }

    ~Interactor()
    {
        thread.join();
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
            boost::asio::buffer(command),
            boost::asio::transfer_all(),
            errorCode);

        if (errorCode) {
            LOG(Logger::WARN, "error while writing to socket: " << errorCode.message());
        }

        notifyStartup();

        for (int i = 0; i < feedbackLines; ++i) {
            boost::asio::streambuf buf;
            boost::system::error_code errorCode;

            size_t length = boost::asio::read_until(socket, buf, '\n', errorCode);
            if (errorCode) {
                LOG(Logger::WARN, "error while writing to socket: " << errorCode.message());
            }

            std::istream lineBuf(&buf);
            std::string line(length, 'X');
            lineBuf.read(&line[0], length);
            feedbackBuffer << line;
        }

        notifyCompletion();

        return 0;
    }

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
#endif

