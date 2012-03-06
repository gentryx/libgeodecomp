#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_io_mockwriter_h_
#define _libgeodecomp_io_mockwriter_h_

#include <sstream>

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/testcell.h>

namespace LibGeoDecomp {

class MockWriter : public Writer<TestCell<2> >, public ParallelWriter<TestCell<2> >
{
public:

    static std::string staticEvents;

    MockWriter(MonolithicSimulator<TestCell<2> > *sim) : 
        Writer<TestCell<2> >("foobar", sim, 1),
        ParallelWriter<TestCell<2> >("foobar", 0, 1) 
    {}

    MockWriter(DistributedSimulator<TestCell<2> > *sim) : 
        Writer<TestCell<2> >("foobar", 0, 1), 
        ParallelWriter<TestCell<2> >("foobar", sim, 1) 
    {}

    ~MockWriter() 
    { 
        staticEvents += "deleted\n"; 
    }

    void initialized() 
    { 
        myEvents << "initialized()\n"; 
    }

    void stepFinished() 
    { 
        unsigned step;
        if (this->sim != 0) {
            step = this->sim->getStep();
        } else {
            step = this->distSim->getStep();
        }

        myEvents << "stepFinished(step=" << step << ")\n"; 
    }

    void allDone() 
    { 
        myEvents << "allDone()\n"; 
    }

    std::string events() 
    { 
        return myEvents.str(); 
    }

private:
    std::ostringstream myEvents;
};

};

#endif
#endif
