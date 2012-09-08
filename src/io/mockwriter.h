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

    using Writer<TestCell<2> >::sim;
    using ParallelWriter<TestCell<2> >::distSim;

    MockWriter(MonolithicSimulator<TestCell<2> > *sim, const unsigned& _period=1) : 
        Writer<TestCell<2> >("foobar", sim, _period),
        ParallelWriter<TestCell<2> >("foobar", 0, _period) 
    {}

    MockWriter(DistributedSimulator<TestCell<2> > *sim, const unsigned& _period=1) : 
        Writer<TestCell<2> >("foobar", 0, _period), 
        ParallelWriter<TestCell<2> >("foobar", sim, _period) 
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
        if (sim != 0) {
            step = sim->getStep();
        } else {
            step = distSim->getStep();
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

}

#endif
