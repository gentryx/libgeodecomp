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

    TestSteerer(
        const unsigned& period,
        const unsigned& eventStep,
        const unsigned& cycleOffset)  :
        Steerer<TestCell<DIM> >(period),
        eventStep(eventStep),
        cycleOffset(cycleOffset)
    {}

    virtual void nextStep(
        GridType *grid,
        const Region<DIM>& validRegion,
        unsigned step)
    {
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
};

}

#endif
