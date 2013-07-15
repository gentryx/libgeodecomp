#ifndef LIBGEODECOMP_IO_TESTSTEERER_H
#define LIBGEODECOMP_IO_TESTSTEERER_H

#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/misc/testcell.h>

namespace LibGeoDecomp {

/**
 * The TestSteerer demos how a Steerer can be implemented to modify
 * the grid during the course of the simulation. The idea is to
 * advance the cell's cycleCounter at \p eventStep by \p cycleOffset.
 */
template<int DIM>
class TestSteerer : public Steerer<TestCell<DIM> >
{
public:
    typedef typename Steerer<TestCell<DIM> >::GridType GridType;
    typedef typename Steerer<TestCell<DIM> >::CoordType CoordType;
    using Steerer<TestCell<DIM> >::region;

    TestSteerer(
        const unsigned& period,
        const unsigned& eventStep,
        const unsigned& cycleOffset)  :
        Steerer<TestCell<DIM> >(period),
        eventStep(eventStep),
        cycleOffset(cycleOffset),
        lastStep(-1)
    {}

    virtual void nextStep(
        GridType *grid,
        const Region<DIM>& validRegion,
        const CoordType& globalDimensions,
        unsigned step,
        SteererEvent event,
        std::size_t rank,
        bool lastCall)
    {
        // ensure setRegion() has actually been called
        TS_ASSERT(!region.empty());
        // fixme: extend this test according to paralleltestwriter

        if (step != eventStep) {
            return;
        }

        for (typename Region<DIM>::Iterator i = validRegion.begin();
             i != validRegion.end();
             ++i) {
            grid->at(*i).cycleCounter += cycleOffset;
        }
    }


private:
    unsigned eventStep;
    unsigned cycleOffset;
    int lastStep;
};

}

#endif
