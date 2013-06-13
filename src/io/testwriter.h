#ifndef LIBGEODECOMP_IO_TESTWRITER_H
#define LIBGEODECOMP_IO_TESTWRITER_H

#include <list>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>

namespace LibGeoDecomp {

/**
 * This class serves to verify the callback behavior of
 * implementations of MonolithicSimulator.
 */
class TestWriter : public Writer<TestCell<2> >
{
public:
    typedef Writer<TestCell<2> >::GridType GridType;

    TestWriter(
        const unsigned& period,
        const SuperVector<unsigned>& expectedSteps,
        const SuperVector<WriterEvent> expectedEvents)  :
        Writer<TestCell<2> >("", period),
        expectedSteps(expectedSteps),
        expectedEvents(expectedEvents)
    {}

    virtual void stepFinished(
        const GridType& grid,
        const unsigned step,
        WriterEvent event)
    {
        unsigned myExpectedCycle = TestCell<2>::nanoSteps() * step;
        TS_ASSERT_TEST_GRID(GridType, grid, myExpectedCycle);

        TS_ASSERT(!expectedSteps.empty());
        unsigned expectedStep = expectedSteps.front();
        WriterEvent expectedEvent = expectedEvents.front();
        expectedSteps.erase(expectedSteps.begin());
        expectedEvents.erase(expectedEvents.begin());
        TS_ASSERT_EQUALS(expectedStep, step);
        TS_ASSERT_EQUALS(expectedEvent, event);
    }

private:
    SuperVector<unsigned> expectedSteps;
    SuperVector<WriterEvent> expectedEvents;
};

}

#endif
