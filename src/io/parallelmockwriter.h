#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_io_parallelmockwriter_h_
#define _libgeodecomp_io_parallelmockwriter_h_

#include <sstream>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/io/parallelwriter.h>

namespace LibGeoDecomp {

class ParallelMockWriter : public ParallelWriter<TestCell<2> > {
public:

    static std::string staticEvents;

    ParallelMockWriter(DistributedSimulator<TestCell<2> > *sim)
        : ParallelWriter<TestCell<2> >("foobar", sim, 1) {}

    ~ParallelMockWriter() { staticEvents += "deleted\n"; }

    void initialized() { myEvents << "initialized()\n"; }

    void stepFinished() { myEvents << "stepFinished(step=" << this->distSim->getStep() << ")\n"; }

    void allDone() { myEvents << "allDone()\n"; }

    std::string events() { return myEvents.str(); }

private:
    std::ostringstream myEvents;
};

};

#endif
#endif
