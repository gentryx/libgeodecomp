#ifndef LIBGEODECOMP_IO_REMOTESTEERER_PIPE_H
#define LIBGEODECOMP_IO_REMOTESTEERER_PIPE_H

#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <libgeodecomp/config.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/mpilayer/mpilayer.h>

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

/**
 * The purpose of the Pipe is to forward steering data between the
 * connection node and the execution nodes. This forwarding is done
 * asynchronously as Actions from the connection node run in a
 * seperate thread while Handlers are run by the RemoteSteerer, which
 * in turn is triggered by the Simulator. The Pipe also manages
 * scattering/gathering of the data via MPI, which is essentially a
 * 1-n operation. The external interface is thread-safe.
 */
class Pipe
{
public:
    friend class PipeTest;

#ifdef LIBGEODECOMP_FEATURE_MPI
    Pipe(
        int root = 0,
        MPI::Comm *communicator = &MPI::COMM_WORLD) :
        mpiLayer(communicator),
        root(root)
    {}
#endif

    void addSteeringRequest(const std::string& request)
    {
        LOG(DEBUG, "Pipe::addSteeringRequest(" << request << ")");
        boost::lock_guard<boost::mutex> lock(mutex);
        steeringRequestsQueue << request;
    }

    void addSteeringFeedback(const std::string& feedback)
    {
        LOG(DEBUG, "Pipe::addSteeringFeedback(" << feedback << ")");
        boost::lock_guard<boost::mutex> lock(mutex);
        steeringFeedback << feedback;
        signal.notify_one();
    }

    StringVec retrieveSteeringRequests()
    {
        LOG(DEBUG, "Pipe::retrieveSteeringRequests()");
        StringVec requests;
        boost::lock_guard<boost::mutex> lock(mutex);
        std::swap(requests, steeringRequests);
        LOG(DEBUG, "  retrieveSteeringRequests yields " << requests.size());
        if (requests.size() > 0) {
            LOG(DEBUG, "  steeringRequests: " << requests);
        }
        return requests;
    }

    StringVec copySteeringRequestsQueue()
    {
        LOG(DEBUG, "Pipe::copySteeringRequestsQueue()");
        boost::lock_guard<boost::mutex> lock(mutex);
        StringVec requests = steeringRequestsQueue;
        return requests;
    }

    StringVec retrieveSteeringFeedback()
    {
        LOG(DEBUG, "Pipe::retrieveSteeringFeedback()");
        StringVec feedback;
        boost::lock_guard<boost::mutex> lock(mutex);
        std::swap(feedback, steeringFeedback);
        LOG(DEBUG, "  retrieveSteeringFeedback yields " << feedback.size());
        LOG(DEBUG, "  retrieveSteeringFeedback is " << feedback);
        return feedback;
    }

    StringVec copySteeringFeedback()
    {
        LOG(DEBUG, "Pipe::copySteeringFeedback()");
        boost::lock_guard<boost::mutex> lock(mutex);
        StringVec feedback = steeringFeedback;
        return feedback;
    }

    void sync()
    {
        LOG(DEBUG, "Pipe::sync()");
        boost::lock_guard<boost::mutex> lock(mutex);
#ifdef LIBGEODECOMP_FEATURE_MPI
        broadcastSteeringRequests();
        moveSteeringFeedbackToRoot();
#endif
    }

    void waitForFeedback(int lines = 1)
    {
        LOG(DEBUG, "Pipe::waitForFeedback(" << lines << ")");
        boost::unique_lock<boost::mutex> lock(mutex);

        while (steeringFeedback.size() < lines) {
            LOG(DEBUG, "  still waiting for feedback (" << steeringFeedback.size() << "/" << lines << ")\n");
            signal.wait(lock);
        }
        LOG(DEBUG, "  feedback acquired");
    }

private:
    boost::mutex mutex;
    boost::condition_variable signal;
    StringVec steeringRequestsQueue;
    StringVec steeringRequests;
    StringVec steeringFeedback;

#ifdef LIBGEODECOMP_FEATURE_MPI
    MPILayer mpiLayer;
    int root;

    void broadcastSteeringRequests()
    {
        LOG(DEBUG, "Pipe::broadcastSteeringRequests()");

        int numRequests = mpiLayer.broadcast(steeringRequestsQueue.size(), root);
        if (mpiLayer.rank() != root) {
            steeringRequestsQueue.resize(numRequests);
        } else {
            LOG(DEBUG, "  steeringRequestsQueue: " << steeringRequestsQueue);
        }

        SuperVector<int> requestSizes(numRequests);

        if (mpiLayer.rank() == root) {
            for (int i = 0; i < numRequests; ++i) {
                requestSizes[i] = steeringRequestsQueue[i].size();
            }
        }

        mpiLayer.broadcastVector(&requestSizes, root);
        if (mpiLayer.rank() != root) {
            for (int i = 0; i < numRequests; ++i) {
                steeringRequestsQueue[i].resize(requestSizes[i]);
            }
        }

        for (int i = 0; i < numRequests; ++i) {
            mpiLayer.broadcast(&steeringRequestsQueue[i][0], requestSizes[i], root);
        }
        steeringRequests.append(steeringRequestsQueue);
        steeringRequestsQueue.clear();
        LOG(DEBUG, "  steeringRequests: " << steeringRequests);
    }

    /**
     * Will move steering requests from the compute nodes to the root
     * where it can then be forwarded to the user.
     */
    void moveSteeringFeedbackToRoot()
    {
        // marshall all feedback in order to use scalable gather afterwards
        SuperVector<char> localBuffer;
        SuperVector<int> localLengths;
        for (StringVec::iterator i = steeringFeedback.begin();
             i != steeringFeedback.end();
             ++i) {
            localBuffer.insert(localBuffer.end(), i->begin(), i->end());
            localLengths << i->size();
        }
        LOG(DEBUG, "moveSteeringFeedbackToRoot: " << localBuffer << " " << localLengths);

        // how many strings are sent per node?
        SuperVector<int> numFeedback = mpiLayer.gather((int)localLengths.size(), root);

        // all lengths of all strings:
        SuperVector<int> allFeedbackLengths(numFeedback.sum());
        mpiLayer.gatherV(&localLengths[0], localLengths.size(), numFeedback, root, &allFeedbackLengths[0]);

        // gather all messages in a single, giant buffer:
        SuperVector<int> charsPerNode = mpiLayer.gather((int)localBuffer.size(), root);
        SuperVector<char> globalBuffer(charsPerNode.sum());

        mpiLayer.gatherV(&localBuffer[0], localBuffer.size(), charsPerNode, root, &globalBuffer[0]);

        // reconstruct strings:
        int cursor = 0;
        steeringFeedback.resize(0);

        for (SuperVector<int>::iterator i = allFeedbackLengths.begin();
             i != allFeedbackLengths.end();
             ++i) {
            int nextCursor = cursor + *i;
            steeringFeedback << std::string(&globalBuffer[cursor],
                                            &globalBuffer[nextCursor]);
            cursor = nextCursor;
        }

        LOG(DEBUG, "  notifying... steeringFeedback.size(" << MPILayer().rank() << ") == " << steeringFeedback.size() << "\n");
        signal.notify_one();
    }
#endif
};

}

}

#endif
