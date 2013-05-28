#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/remotesteerer/action.h>
#include <libgeodecomp/io/remotesteerer/commandserver.h>

using namespace LibGeoDecomp;
using namespace LibGeoDecomp::RemoteSteererHelpers;;

namespace LibGeoDecomp {

class ActionTest : public CxxTest::TestSuite
{
public:
    class MockAction : public Action<int>
    {
    public:
        MockAction() :
            RemoteSteererHelpers::Action<int>("this is but a dummy action", "mock")
        {}

        virtual void operator()(const StringOps::StringVec& parameters, CommandServer<int> *server)
        {
            server->sendMessage("MockAction mocks you!");
        }
    };

    class MockCommandServer : public CommandServer<int>
    {
    public:
        MockCommandServer() :
            CommandServer<int>(12345, CommandServer<int>::FunctionMap(), 0)
        {}

        void sendMessage(const std::string& message)
        {
            messages << message;
        }

        StringOps::StringVec messages;
    };

    void testBasic()
    {
        MockCommandServer server;
        {
            MockAction action;
            TS_ASSERT_EQUALS("this is but a dummy action", action.helpMessage());
            TS_ASSERT_EQUALS("mock", action.key());
            StringOps::StringVec parameters;
            parameters << "mock"
                       << "arg0"
                       << "arg1";
            std::cout << "calling action\n";
            action(parameters, &server);
            std::cout << "returning\n";
        }
    }
};

}
