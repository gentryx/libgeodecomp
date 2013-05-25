#ifndef LIBGEODECOMP_IO_PARALLELTESTWRITER_H
#define LIBGEODECOMP_IO_PARALLELTESTWRITER_H

#include <list>
#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>

namespace LibGeoDecomp {

/**
 * This class serves to verify the callback behavior of
 * implementations of DistributedSimulator.
 */
class ParallelTestWriter : public ParallelWriter<TestCell<2> >
{
public:
    typedef typename ParallelWriter<TestCell<2> >::GridType GridType;
    using ParallelWriter<TestCell<2> >::region;

    ParallelTestWriter(
        const unsigned& period,
        const SuperVector<unsigned>& expectedSteps,
        const SuperVector<WriterEvent> expectedEvents)  :
        ParallelWriter<TestCell<2> >("", period),
        expectedSteps(expectedSteps),
        expectedEvents(expectedEvents),
        lastStep(-1)
    {}

    virtual void stepFinished(
        const GridType& grid,
        const Region<Topology::DIM>& validRegion,
        const Coord<Topology::DIM>& globalDimensions,
        unsigned step,
        WriterEvent event,
        bool lastCall)
    {
        if (lastStep != step) {
            // step should only change if lastCall has been set before
            TS_ASSERT(unaccountedRegion.empty());
            unaccountedRegion = region;
        }
        unaccountedRegion -= validRegion;

        unsigned myExpectedCycle = TestCell<2>::nanoSteps() * step;
        TS_ASSERT_TEST_GRID_REGION(GridType, grid, validRegion, myExpectedCycle);

        TS_ASSERT(!expectedSteps.empty());
        unsigned expectedStep = expectedSteps.front();
        WriterEvent expectedEvent = expectedEvents.front();
        expectedSteps.erase(expectedSteps.begin());
        expectedEvents.erase(expectedEvents.begin());
        TS_ASSERT_EQUALS(expectedStep, step);
        TS_ASSERT_EQUALS(expectedEvent, event);

        // ensure setRegion() has actually been called
        TS_ASSERT(!region.empty());
        // ensure validRegion is a subset of what was specified via setRegion()
        if (!(validRegion - region).empty())
            std::cout << "deltaRegion: " << (validRegion - region) << "\n";
        TS_ASSERT((validRegion - region).empty());
        // check that all parts of the specified region were actually consumed
        if (lastCall) {
            TS_ASSERT(unaccountedRegion.empty());
        }

        lastStep = step;
    }

private:
    SuperVector<unsigned> expectedSteps;
    SuperVector<WriterEvent> expectedEvents;
    int lastStep;
    Region<2> unaccountedRegion;
};

}

#endif
