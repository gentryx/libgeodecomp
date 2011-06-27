#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_mocksimulator_h_
#define _libgeodecomp_parallelization_mocksimulator_h_

#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>

namespace LibGeoDecomp {

class MockSimulator : public MonolithicSimulator<TestCell<2> >
{
public:
    MockSimulator(Initializer<TestCell<2> > *_init) : 
        MonolithicSimulator<TestCell<2> >(_init) {}
    ~MockSimulator() { events += "deleted\n"; }
    void step() {}
    void run() {}
    Grid<TestCell<2> > *getGrid() { return 0; }

    static std::string events;
};

};

#endif
#endif
