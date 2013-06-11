#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/remotesteerer/pipe.h>
#include <libgeodecomp/mpilayer/mpilayer.h>

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

        TS_ASSERT_EQUALS(pipe.steeringRequests.size(), 2);
        TS_ASSERT_EQUALS(pipe.steeringFeedback.size(), 0);

        TS_ASSERT_EQUALS(pipe.steeringRequests[0], "set heat 0.1 100 120 110");
        TS_ASSERT_EQUALS(pipe.steeringRequests[1], "set flow 6.9 100 120 110");

        TS_ASSERT_EQUALS(pipe.retrieveSteeringRequests().size(), 2);
        TS_ASSERT_EQUALS(pipe.steeringRequests.size(), 0);
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
        int expectedSize = (mpiLayer.rank() == 0)? 9 : 0;
        TS_ASSERT_EQUALS(pipe.steeringFeedback.size(), expectedSize);

        TS_ASSERT_EQUALS(pipe.retrieveSteeringFeedback().size(), expectedSize);
        TS_ASSERT_EQUALS(pipe.steeringFeedback.size(), 0);
    }
};

}

}
