#include <libgeodecomp/io/clonableinitializerwrapper.h>
#include <sstream>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class LoggingTestInit : public Initializer<int>
{
public:
    using Initializer<int>::DIM;

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

    void testDelegate()
    {
        LoggingTestInit init;
        ClonableInitializerWrapper<LoggingTestInit> wrapper(init);

        GridBase<int, 2> *pointer = 0;
        wrapper.grid(pointer);

        TS_ASSERT_EQUALS(Coord<2>(40, 30),                                wrapper.gridDimensions());
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(10, 20), Coord<2>(40, 30)), wrapper.gridBox());
        TS_ASSERT_EQUALS(47, wrapper.maxSteps());
        TS_ASSERT_EQUALS(11, wrapper.startStep());

        std::stringstream expected;
        expected << "grid(0)\n"
                 << "gridDimensions()\n"
                 << "gridBox()\n"
                 << "maxSteps()\n"
                 << "startStep()\n";

        TS_ASSERT_EQUALS(LoggingTestInit::events.str(), expected.str());
    }
};

}
