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
        std::size_t feedbackLines,
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
            waitForStartup();
        }
    }

    ~Interactor()
    {
        if (thread.joinable()) {
            thread.join();
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
        LOG(DBG, "Interactor::operator(" << command << ")");
        boost::asio::io_service ioService;
        tcp::resolver resolver(ioService);
        tcp::resolver::query query(host, StringOps::itoa(port));
        tcp::resolver::iterator endpointIterator = resolver.resolve(query);
        tcp::socket socket(ioService);
        boost::asio::connect(socket, endpointIterator);
        boost::system::error_code errorCode;

        std::string commandSuffix = "\nwait " + StringOps::itoa(feedbackLines) + "\n";
        boost::asio::write(
            socket,
            boost::asio::buffer(command + commandSuffix),
            boost::asio::transfer_all(),
            errorCode);

        if (errorCode) {
            LOG(Logger::WARN, "error while writing to socket: " << errorCode.message());
        }

        notifyStartup();

        for (;;) {
            LOG(DBG, "Interactor::operator() reading... [" << feedbackBuffer.size() << "/" << feedbackLines << "]");
            if (feedbackBuffer.size() >= feedbackLines) {
                break;
            }

            boost::array<char, 1024> buf;
            boost::system::error_code errorCode;
            std::size_t length = socket.read_some(boost::asio::buffer(buf), errorCode);
            std::string input(buf.data(), length);
            StringVec lines = StringOps::tokenize(input, "\n");
            handleInput(lines);
        }

        notifyCompletion();

        LOG(DBG, "Interactor::operator() done");
        return 0;
    }

private:
    StringVec feedbackBuffer;
    boost::condition_variable signal;
    boost::mutex mutex;
    std::string command;
    std::size_t feedbackLines;
    int port;
    std::string host;
    bool started;
    bool completed;
    boost::thread thread;

    void waitForStartup()
    {
        boost::unique_lock<boost::mutex> lock(mutex);
        while(!started) {
            signal.wait(lock);
        }
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

    void handleInput(const StringVec& lines)
    {
        LOG(DBG, "Interactor::handleInput(" << lines << ")");
        // only add lines which are not equal to "\0"
        for (std::size_t i = 0; i < lines.size(); ++i) {
            const std::string& line = lines[i];
            if (line == "") {
                LOG(WARN, "Interactor rejects empty line as feedback");
                continue;
            }
            if ((line.size() == 1) || (line[0] == 0)) {
                LOG(WARN, "Interactor rejects null line as feedback");
                continue;
            }

            LOG(DBG, "Interactor accepted line »" << line << "«");
            feedbackBuffer << line;
        }
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
