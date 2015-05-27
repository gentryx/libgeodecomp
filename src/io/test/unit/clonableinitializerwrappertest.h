#include <libgeodecomp/io/clonableinitializerwrapper.h>
#include <sstream>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class LoggingTestInit : public Initializer<int>
{
public:
    using Initializer<int>::DIM;

    explicit LoggingTestInit(int arg1 = -1)
    {
        events << "LoggingTestInit("
               << arg1 << ")\n";
    }

    LoggingTestInit(int arg1, int arg2)
    {
        events << "LoggingTestInit("
               << arg1 << ", "
               << arg2 << ")\n";
    }

    LoggingTestInit(int arg1, int arg2, int arg3)
    {
        events << "LoggingTestInit("
               << arg1 << ", "
               << arg2 << ", "
               << arg3 << ")\n";
    }

    LoggingTestInit(int arg1, int arg2, int arg3, int arg4)
    {
        events << "LoggingTestInit("
               << arg1 << ", "
               << arg2 << ", "
               << arg3 << ", "
               << arg4 << ")\n";
    }

    LoggingTestInit(int arg1, int arg2, int arg3, int arg4, int arg5)
    {
        events << "LoggingTestInit("
               << arg1 << ", "
               << arg2 << ", "
               << arg3 << ", "
               << arg4 << ", "
               << arg5 << ")\n";
    }

    LoggingTestInit(int arg1, int arg2, int arg3, int arg4, int arg5, int arg6)
    {
        events << "LoggingTestInit("
               << arg1 << ", "
               << arg2 << ", "
               << arg3 << ", "
               << arg4 << ", "
               << arg5 << ", "
               << arg6 << ")\n";
    }

    LoggingTestInit(int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7)
    {
        events << "LoggingTestInit("
               << arg1 << ", "
               << arg2 << ", "
               << arg3 << ", "
               << arg4 << ", "
               << arg5 << ", "
               << arg6 << ", "
               << arg7 << ")\n";
    }

    LoggingTestInit(int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8)
    {
        events << "LoggingTestInit("
               << arg1 << ", "
               << arg2 << ", "
               << arg3 << ", "
               << arg4 << ", "
               << arg5 << ", "
               << arg6 << ", "
               << arg7 << ", "
               << arg8 << ")\n";
    }

    void grid(GridBase<int, DIM> *target)
    {
        events << "grid(" << target << ")\n";
    }

    CoordBox<DIM> gridBox()
    {
        events << "gridBox()\n";
        return CoordBox<2>(Coord<2>(10, 20), Coord<2>(40, 30));
    }

    Coord<DIM> gridDimensions() const
    {
        events << "gridDimensions()\n";
        return Coord<2>(40, 30);
    }

    unsigned startStep() const
    {
        events << "startStep()\n";
        return 11;
    }

    unsigned maxSteps() const
    {
        events << "maxSteps()\n";
        return 47;
    }

    static std::stringstream events;
};

std::stringstream LoggingTestInit::events;

class ClonableInitializerWrapperTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        expected.str("");
        LoggingTestInit::events.str("");
    }

    void testDelegate()
    {
        LoggingTestInit init;
        ClonableInitializerWrapper<LoggingTestInit> wrapper(init);

        GridBase<int, 2> *pointer = 0;
        wrapper.grid(pointer);

        TS_ASSERT_EQUALS(Coord<2>(40, 30),                                wrapper.gridDimensions());
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(10, 20), Coord<2>(40, 30)), wrapper.gridBox());
        TS_ASSERT_EQUALS(std::size_t(47), wrapper.maxSteps());
        TS_ASSERT_EQUALS(std::size_t(11), wrapper.startStep());

        int *dummy = 0;
        expected << "LoggingTestInit(-1)\n"
                 << "grid(" << dummy << ")\n"
                 << "gridDimensions()\n"
                 << "gridBox()\n"
                 << "maxSteps()\n"
                 << "startStep()\n";

        TS_ASSERT_EQUALS(LoggingTestInit::events.str(), expected.str());
    }

    void testWrapWithCopyConstructor()
    {
        LoggingTestInit init(1, 2, 3, 4);
        Initializer<int> *foo = ClonableInitializerWrapper<LoggingTestInit>::wrap(init);
        delete foo;

        expected << "LoggingTestInit(1, 2, 3, 4)\n";

        TS_ASSERT_EQUALS(LoggingTestInit::events.str(), expected.str());
    }

    void testForwardingOfConstructorArguments()
    {
        Initializer<int> *foo = 0;

        foo = ClonableInitializerWrapper<LoggingTestInit>::wrap(1);
        expected << "LoggingTestInit(1)\n";
        delete foo;

        foo = ClonableInitializerWrapper<LoggingTestInit>::wrap(2, 3);
        expected << "LoggingTestInit(2, 3)\n";
        delete foo;

        foo = ClonableInitializerWrapper<LoggingTestInit>::wrap(4, 5, 6);
        expected << "LoggingTestInit(4, 5, 6)\n";
        delete foo;

        foo = ClonableInitializerWrapper<LoggingTestInit>::wrap(7, 8, 9, 10);
        expected << "LoggingTestInit(7, 8, 9, 10)\n";
        delete foo;

        foo = ClonableInitializerWrapper<LoggingTestInit>::wrap(11, 12, 13, 14, 15);
        expected << "LoggingTestInit(11, 12, 13, 14, 15)\n";
        delete foo;

        foo = ClonableInitializerWrapper<LoggingTestInit>::wrap(16, 17, 18, 19, 20, 21);
        expected << "LoggingTestInit(16, 17, 18, 19, 20, 21)\n";
        delete foo;

        foo = ClonableInitializerWrapper<LoggingTestInit>::wrap(22, 23, 24, 25, 26, 27, 28);
        expected << "LoggingTestInit(22, 23, 24, 25, 26, 27, 28)\n";
        delete foo;

        foo = ClonableInitializerWrapper<LoggingTestInit>::wrap(29, 30, 31, 32, 33, 34, 35, 36);
        expected << "LoggingTestInit(29, 30, 31, 32, 33, 34, 35, 36)\n";
        delete foo;

        TS_ASSERT_EQUALS(LoggingTestInit::events.str(), expected.str());
    }

private:
    std::stringstream expected;
};

}
