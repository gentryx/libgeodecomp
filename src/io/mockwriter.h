#ifndef LIBGEODECOMP_IO_MOCKWRITER_H
#define LIBGEODECOMP_IO_MOCKWRITER_H

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>
#include <libgeodecomp/parallelization/distributedsimulator.h>
#include <libgeodecomp/misc/clonable.h>
#include <libgeodecomp/misc/testcell.h>

#include <sstream>

namespace LibGeoDecomp {

namespace MockWriterHelpers {

class MockWriterEvent
{
public:
    MockWriterEvent(unsigned step, WriterEvent event, std::size_t rank, bool lastCall) :
        step(step),
        event(event),
        rank(rank),
        lastCall(lastCall)
    {}

    bool operator==(const MockWriterEvent& other) const
    {
        return
            (other.step     == step    ) &&
            (other.event    == event   ) &&
            (other.rank     == rank    ) &&
            (other.lastCall == lastCall);
    }

    std::string toString() const
    {
        std::stringstream buf;
        buf << "MockWriterEvent(" << step << ", ";
        switch(event) {
        case WRITER_INITIALIZED:
            buf << "WRITER_INITIALIZED";
            break;
        case WRITER_STEP_FINISHED:
            buf << "WRITER_STEP_FINISHED";
            break;
        case WRITER_ALL_DONE:
            if (step != -1) {
                buf << "WRITER_ALL_DONE";
            } else {
                buf << "DELETED";
            }
            break;
        default:
            buf << "unknown event";
            break;
        }
        buf << ", " << rank << ", " << lastCall << ")\n";

        return buf.str();
    }

    int step;
    WriterEvent event;
    std::size_t rank;
    bool lastCall;
};

}

template<typename CELL_TYPE=TestCell<2> >
class MockWriter :
        public Clonable<Writer<        CELL_TYPE>, MockWriter<CELL_TYPE> >,
        public Clonable<ParallelWriter<CELL_TYPE>, MockWriter<CELL_TYPE> >
{
public:
    typedef MockWriterHelpers::MockWriterEvent Event;
    typedef std::vector<Event> EventVec;

    explicit MockWriter(boost::shared_ptr<EventVec> events, const unsigned& period = 1) :
        Clonable<Writer<CELL_TYPE>, MockWriter>("", period),
        Clonable<ParallelWriter<CELL_TYPE>, MockWriter>("", period),
        events(events)
    {}

    ~MockWriter()
    {
        *events << Event(-1, WRITER_ALL_DONE, -1, true);
    }

    void stepFinished(
        const typename Writer<CELL_TYPE>::GridType& grid,
        unsigned step,
        WriterEvent event)
    {
        stepFinished(step, event, 0, true);
    }

    void stepFinished(
        const typename ParallelWriter<CELL_TYPE>::GridType& grid,
        const Region<2>& validRegion,
        const Coord<2>& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
        stepFinished(step, event, rank, lastCall);
    }

private:
    boost::shared_ptr<MockWriter::EventVec> events;

    void stepFinished(unsigned step, WriterEvent event, std::size_t rank, bool lastCall)

    {
        *events << Event(step, event, rank, lastCall);
    }
};

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const MockWriter<>::Event& event)
{
    __os << event.toString();
    return __os;
}

}

#endif
