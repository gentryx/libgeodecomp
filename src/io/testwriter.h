#ifndef LIBGEODECOMP_IO_TESTWRITER_H
#define LIBGEODECOMP_IO_TESTWRITER_H

#include <list>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/clonable.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>

namespace LibGeoDecomp {

/**
 * This class serves to verify the callback behavior of
 * implementations of MonolithicSimulator.
 */
template<typename CELL = TestCell<2> >
class TestWriter : public Clonable<Writer<CELL>, TestWriter<CELL> >
{
public:
    using Writer<CELL>::NANO_STEPS;
    using typename Writer<CELL>::GridType;

    TestWriter(
        unsigned period,
        const std::vector<int>& expectedSteps,
        const std::vector<WriterEvent>& expectedEvents)  :
        Clonable<Writer<CELL>, TestWriter<CELL> >("", period),
        expectedSteps(expectedSteps),
        expectedEvents(expectedEvents)
    {}

    virtual void stepFinished(
        const GridType& grid,
        unsigned step,
        WriterEvent event)
    {
        unsigned myExpectedCycle = NANO_STEPS * step;

        TS_ASSERT(!expectedSteps.empty());
        unsigned expectedStep = expectedSteps.front();
        WriterEvent expectedEvent = expectedEvents.front();

        LOG(DBG, "TestWriter::stepFinished()\n"
            << "  expected: " << expectedEvent << "@" << expectedStep << "\n"
            << "    actual: " << event         << "@" << step << "\n")

        expectedSteps.erase(expectedSteps.begin());
        expectedEvents.erase(expectedEvents.begin());
        TS_ASSERT_EQUALS(expectedStep, step);
        TS_ASSERT_EQUALS(expectedEvent, event);

        TS_ASSERT_TEST_GRID2(GridType, grid, myExpectedCycle, typename);
    }

private:
    std::vector<int> expectedSteps;
    std::vector<WriterEvent> expectedEvents;
};

}

#endif
