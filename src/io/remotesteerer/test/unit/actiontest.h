#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/remotesteerer/action.h>
#include <libgeodecomp/io/remotesteerer/pipe.h>
#include <libgeodecomp/misc/testcell.h>

using namespace LibGeoDecomp;
using namespace LibGeoDecomp::RemoteSteererHelpers;

namespace LibGeoDecomp {

class ActionTest : public CxxTest::TestSuite
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
            pipe.addSteeringFeedback("MockAction mocks you! " + parameters[0]);
        }
    };

    void testBasic()
    {
        Pipe pipe;
        MockAction action;

        TS_ASSERT_EQUALS("this is but a dummy action", action.helpMessage());
        TS_ASSERT_EQUALS("mock", action.key());
        StringVec parameters;
        parameters << "arrrr"
                   << "matey";
        action(parameters, pipe);
        StringVec feedback = pipe.retrieveSteeringFeedback();
        TS_ASSERT_EQUALS(feedback.size(), 1);
        TS_ASSERT_EQUALS(feedback[0], "MockAction mocks you! arrrr");
    }
};

}
