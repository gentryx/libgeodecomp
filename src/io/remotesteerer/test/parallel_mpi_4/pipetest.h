#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#include <hpx/config.hpp>
#endif

#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/io/remotesteerer/pipe.h>

#include <boost/thread.hpp>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;
using namespace LibGeoDecomp::RemoteSteererHelpers;

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

class PipeTest : public CxxTest::TestSuite
{
public:
    void testSyncSteeringRequests()
    {
        MPILayer mpiLayer;
        Pipe pipe;

        if (mpiLayer.rank() == 0) {
            pipe.addSteeringRequest("set heat 0.1 100 120 110");
            pipe.addSteeringRequest("set flow 6.9 100 120 110");
        }

        pipe.sync();

        TS_ASSERT_EQUALS(pipe.steeringRequests.size(), std::size_t(2));
        TS_ASSERT_EQUALS(pipe.steeringFeedback.size(), std::size_t(0));

        TS_ASSERT_EQUALS(pipe.steeringRequests[0], "set heat 0.1 100 120 110");
        TS_ASSERT_EQUALS(pipe.steeringRequests[1], "set flow 6.9 100 120 110");

        TS_ASSERT_EQUALS(pipe.retrieveSteeringRequests().size(), unsigned(2));
        TS_ASSERT_EQUALS(pipe.steeringRequests.size(), std::size_t(0));
    }

    void testSyncSteeringFeedback()
    {
        MPILayer mpiLayer;
        Pipe pipe;

        pipe.addSteeringFeedback("node " + StringOps::itoa(mpiLayer.rank()) + " starting");
        if (mpiLayer.rank() == 2) {
            pipe.addSteeringFeedback("node 2 encountered error");
        }
        pipe.addSteeringFeedback("node " + StringOps::itoa(mpiLayer.rank()) + " shutting down");

        pipe.sync();
        // 4 ranks with 2x feedback each ("starting" + "shutting down"), plus rank 2 with an error message
        unsigned expectedSize = (mpiLayer.rank() == 0)? 9 : 0;
        TS_ASSERT_EQUALS(pipe.steeringFeedback.size(),           expectedSize);
        TS_ASSERT_EQUALS(pipe.copySteeringFeedback().size(),     expectedSize);
        TS_ASSERT_EQUALS(pipe.retrieveSteeringFeedback().size(), expectedSize);
        TS_ASSERT_EQUALS(pipe.steeringFeedback.size(), std::size_t(0));
    }

    class Runner
    {
    public:
        Runner(Pipe *pipe) :
            pipe(pipe)
        {}

        int operator()()
        {
            sleep(2);
            pipe->addSteeringFeedback("bingobongo\n");
            return 0;
        }
    private:
        Pipe *pipe;
    };

    void testWaitForFeedback()
    {
        MPILayer mpiLayer;
        Pipe pipe;

        if (mpiLayer.rank() == 0) {
            boost::thread myThread((Runner(&pipe)));
            pipe.waitForFeedback();
            StringVec actual = pipe.retrieveSteeringFeedback();
            StringVec expected;
            expected << "bingobongo\n";
            TS_ASSERT_EQUALS(actual, expected);

            myThread.join();
        }
    }
};

}

}
