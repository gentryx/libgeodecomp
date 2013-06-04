#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/remotesteerer/commandserver.h>

using namespace LibGeoDecomp;
using namespace LibGeoDecomp::RemoteSteererHelpers;

namespace LibGeoDecomp {

namespace RemoteSteererHelpers {

class CommandServerTest : public CxxTest::TestSuite
{
public:
    class MockAction : public Action<int>
    {
    public:
        MockAction() :
            RemoteSteererHelpers::Action<int>("this is but a dummy action", "mock")
        {}

        virtual void operator()(const StringOps::StringVec& parameters, Pipe& pipe)
        {
            pipe.addSteeringFeedback("MockAction mocks you!");
        }
    };

    // class MockHandler : public Handler<int>
    // {
    // };

    void testActionInvocationAndFeedback()
    {
        boost::shared_ptr<Pipe> pipe(new Pipe);
        {
            CommandServer<int> server(47110, pipe);
            server.addAction(new MockAction());
            CommandServer<int>::sendCommand("mock 1 2 3", "127.0.0.1", 47110);
        }
        StringOps::StringVec feedback = pipe->retrieveSteeringFeedback();
        TS_ASSERT_EQUALS(feedback.size(), 1);
        TS_ASSERT_EQUALS(feedback[0], "MockAction mocks you!");
    }

    // void testActionsInvokingHandlers()
    // {
    //     boost::shared_ptr<Pipe> pipe(new Pipe);
    //     {
    //         CommandServer<int> server(47110, pipe);
    //         server.addHandler(new MockHandler());
    //         CommandServer<int>::sendCommand("mock 1 2 3", "127.0.0.1", 47110);
    //     }
    // }
};

}
}
