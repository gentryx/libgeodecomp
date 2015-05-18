#include <libgeodecomp/io/remotesteerer/commandserver.h>

#include <cxxtest/TestSuite.h>

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
            RemoteSteererHelpers::Action<int>("mock", "this is but a dummy action")
        {}

        virtual void operator()(const StringVec& parameters, Pipe& pipe)
        {
            pipe.addSteeringFeedback("MockAction mocks you!");
        }
    };

    void setUp()
    {
        pipe.reset(new Pipe);
    }

    void tearDown()
    {
        pipe.reset();
    }

    void testActionInvocationAndFeedback()
    {
        int port = 47110;
        CommandServer<int> server(port, pipe);
        server.addAction(new MockAction());
        StringVec feedback = CommandServer<int>::sendCommandWithFeedback("mock 1 2 3", 1, port);
        TS_ASSERT_EQUALS(feedback.size(), std::size_t(1));
        TS_ASSERT_EQUALS(feedback[0], "MockAction mocks you!");
    }

    void testInvalidCommand()
    {
        int port = 47114;
        CommandServer<int> server(port, pipe);
        StringVec feedback = CommandServer<int>::sendCommandWithFeedback("blah", 2, port);

        TS_ASSERT_EQUALS(feedback.size(), std::size_t(2));
        TS_ASSERT_EQUALS(feedback[0], "command not found: blah");
        TS_ASSERT_EQUALS(feedback[1], "try \"help\"");
    }

private:
    boost::shared_ptr<Pipe> pipe;
};

}
}
