#ifndef LIBGEODECOMP_IO_MOCKWRITER_H
#define LIBGEODECOMP_IO_MOCKWRITER_H

#include <sstream>

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>
#include <libgeodecomp/parallelization/distributedsimulator.h>
#include <libgeodecomp/misc/testcell.h>

namespace LibGeoDecomp {

class MockWriter : public Writer<TestCell<2> >, public ParallelWriter<TestCell<2> >
{
public:
    static std::string staticEvents;

    MockWriter(const unsigned& period=1) :
        Writer<TestCell<2> >("", period),
        ParallelWriter<TestCell<2> >("", period)
    {}

    ~MockWriter()
    {
        staticEvents += "deleted\n";
    }

    void stepFinished(
        const Writer<TestCell<2> >::GridType& grid,
        unsigned step,
        WriterEvent event)
    {
        stepFinished(step, event);
    }

    void stepFinished(
        const ParallelWriter<TestCell<2> >::GridType& grid,
        const Region<2>& validRegion,
        const Coord<2>& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
        stepFinished(step, event);
    }

    std::string events()
    {
        return myEvents.str();
    }

private:
    std::ostringstream myEvents;

    void stepFinished(unsigned step, WriterEvent event)
    {
        switch (event) {
        case WRITER_INITIALIZED:
            myEvents << "initialized()\n";
            break;
        case WRITER_STEP_FINISHED:
            myEvents << "stepFinished(step=" << step << ")\n";
            break;
        case WRITER_ALL_DONE:
            myEvents << "allDone()\n";
            break;
        default:
            myEvents << "unknown event\n";
        }
    }
};

}

#endif
