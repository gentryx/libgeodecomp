#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_io_mockwriter_h_
#define _libgeodecomp_io_mockwriter_h_

#include <sstream>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/io/writer.h>

namespace LibGeoDecomp {

class MockWriter : public Writer<TestCell<2> > {
public:

    static std::string staticEvents;

    MockWriter(MonolithicSimulator<TestCell<2> > *sim)
        : Writer<TestCell<2> >("foobar", sim, 1) {}

    ~MockWriter() { staticEvents += "deleted\n"; }

    void initialized() { myEvents << "initialized()\n"; }

    void stepFinished() { myEvents << "stepFinished(step=" << this->sim->getStep() << ")\n"; }

    void allDone() { myEvents << "allDone()\n"; }

    std::string events() { return myEvents.str(); }

private:
    std::ostringstream myEvents;
};

};

#endif
#endif
