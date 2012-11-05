#ifndef _libgeodecomp_io_teststeerer_h_
#define _libgeodecomp_io_teststeerer_h_

#include <sstream>
#include <libgeodecomp/io/steerer.h>
#include <libgeodecomp/misc/testcell.h>

namespace LibGeoDecomp {

/**
 * The TestSteerer demos how a Steerer can be implemented to modify
 * the grid during the course of the simulation. The idea is to
 * advance the TestCell's cycleCounter at _eventStep by \p _cycleOffset.
 */
template<int DIM>
class TestSteerer : public Steerer<TestCell<DIM> >
{
public:
    typedef typename Steerer<TestCell<DIM> >::GridType GridType;

    TestSteerer(
        const unsigned& _period, 
        const unsigned& _eventStep, 
        const unsigned& _cycleOffset)  :
        Steerer<TestCell<DIM> >(_period),
        eventStep(_eventStep),
        cycleOffset(_cycleOffset)
    {}

    virtual void nextStep(
        GridType *grid, 
        const Region<DIM>& validRegion, 
        const unsigned& step) 
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
