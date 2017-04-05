#ifndef LIBGEODECOMP_IO_TESTSTEERER_H
#define LIBGEODECOMP_IO_TESTSTEERER_H

#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/misc/testcell.h>

namespace LibGeoDecomp {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4820 )
#endif

/**
 * The TestSteerer demos how a Steerer can be implemented to modify
 * the grid during the course of the simulation. The idea is to
 * advance the cell's cycleCounter at \p eventStep by \p cycleOffset.
 *
 * The simulation will be terminated at the time step given by \p
 * terminalStep.
 */
template<int DIM>
class TestSteerer : public Steerer<TestCell<DIM> >
{
public:
    typedef typename Steerer<TestCell<DIM> >::SteererFeedback SteererFeedback;
    typedef typename Steerer<TestCell<DIM> >::GridType GridType;
    typedef typename Steerer<TestCell<DIM> >::CoordType CoordType;
    using Steerer<TestCell<DIM> >::region;

    TestSteerer(
        unsigned period,
        unsigned eventStep,
        unsigned cycleOffset,
        unsigned terminalStep = 1000000)  :
        Steerer<TestCell<DIM> >(period),
        eventStep(eventStep),
        cycleOffset(cycleOffset),
        terminalStep(terminalStep),
        lastStep(-1),
        lastEvent(STEERER_ALL_DONE)
    {}

    Steerer<TestCell<DIM> > *clone() const
    {
        return new TestSteerer(*this);
    }

    virtual void nextStep(
        GridType *grid,
        const Region<DIM>& validRegion,
        const CoordType& globalDimensions,
        unsigned step,
        SteererEvent event,
        std::size_t rank,
        bool lastCall,
        SteererFeedback *feedback)
    {
        // ensure setRegion() has actually been called
        TS_ASSERT(!region.empty());

        // lastCall should have been true if we've switched time
        // steps. But only if we weren't initializing before (or are
        // now finishing up). Hence the event check...
        if (lastEvent == event) {
            TS_ASSERT_EQUALS(previousLastCall, (lastStep != step));
        }
        lastEvent = event;
        previousLastCall = lastCall;

        // ensure that all parts of this->region have been accounted for
        if (lastStep != step) {
            TS_ASSERT(unaccountedRegion.empty());
            unaccountedRegion = region;
        }
        unaccountedRegion -= validRegion;

        lastStep = step;

        if (step >= terminalStep) {
            feedback->endSimulation();
        }

        if (step != eventStep) {
            return;
        }

        for (typename Region<DIM>::Iterator i = validRegion.begin();
             i != validRegion.end();
             ++i) {
            TestCell<DIM> cell = grid->get(*i);
            cell.cycleCounter += cycleOffset;
            grid->set(*i, cell);
        }
    }


private:
    unsigned eventStep;
    unsigned cycleOffset;
    unsigned terminalStep;
    unsigned lastStep;
    SteererEvent lastEvent;
    bool previousLastCall;
    Region<DIM> unaccountedRegion;
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
