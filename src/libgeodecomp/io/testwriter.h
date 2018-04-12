#ifndef LIBGEODECOMP_IO_TESTWRITER_H
#define LIBGEODECOMP_IO_TESTWRITER_H

#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/clonable.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>

// Kill some warnings in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4711 )
#endif

#include <list>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

/**
 * This class serves to verify the callback behavior of
 * implementations of MonolithicSimulator.
 */
template<typename CELL = TestCell<2> >
class TestWriter : public Clonable<Writer<CELL>, TestWriter<CELL> >
{
public:
    typedef typename Writer<CELL>::GridType GridType;
    using Writer<CELL>::NANO_STEPS;

    TestWriter(
        unsigned period,
        unsigned firstStep,
        unsigned lastStep)  :
        Clonable<Writer<CELL>, TestWriter<CELL> >("", period)
    {
        expectedSteps << firstStep;
        expectedEvents << WRITER_INITIALIZED;
        for (unsigned i = firstStep + period - firstStep % period; i < lastStep; i += period) {
            expectedSteps << i;
            expectedEvents << WRITER_STEP_FINISHED;
        }

        expectedSteps << lastStep;
        expectedEvents << WRITER_ALL_DONE;
    }

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

    bool allEventsDone() const
    {
        return expectedSteps.empty() && expectedEvents.empty();
    }

private:
    std::vector<unsigned> expectedSteps;
    std::vector<WriterEvent> expectedEvents;
};

}

#endif
