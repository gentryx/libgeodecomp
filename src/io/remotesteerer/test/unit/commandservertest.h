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
        CommandServer<int> server(47110, pipe);
        server.addAction(new MockAction());
        StringVec feedback = CommandServer<int>::sendCommandWithFeedback("mock 1 2 3", 1, 47110);
        TS_ASSERT_EQUALS(feedback.size(), size_t(1));
        TS_ASSERT_EQUALS(feedback[0], "MockAction mocks you!");
    }

    void testInvalidCommand()
    {
        CommandServer<int> server(47112, pipe);
        StringVec feedback = CommandServer<int>::sendCommandWithFeedback("blah", 2, 47112);

        TS_ASSERT_EQUALS(feedback.size(), size_t(2));
        TS_ASSERT_EQUALS(feedback[0], "command not found: blah");
        TS_ASSERT_EQUALS(feedback[1], "try \"help\"");
    }

private:
    boost::shared_ptr<Pipe> pipe;
};

}
}
