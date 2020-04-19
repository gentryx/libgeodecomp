#ifndef LIBGEODECOMP_IO_MOCKWRITER_H
#define LIBGEODECOMP_IO_MOCKWRITER_H

#include <libgeodecomp/io/parallelwriter.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>
#include <libgeodecomp/parallelization/distributedsimulator.h>
#include <libgeodecomp/misc/clonable.h>
#include <libgeodecomp/misc/testcell.h>

#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/include/threads.hpp>
#include <hpx/concurrency/spinlock.hpp>
#endif

#include <sstream>

namespace LibGeoDecomp {

namespace MockWriterHelpers {

/**
 * internal helper class
 */
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

#ifdef LIBGEODECOMP_WITH_HPX

/**
 * Internal helper class, records events received by a writer; for use
 * in unit test.
 */
class ThreadSafeEventsStore
{
public:
    typedef std::vector<MockWriterEvent>::iterator iterator;

    ThreadSafeEventsStore() :
        insertMutex(new hpx::lcos::local::spinlock())
    {}

    ThreadSafeEventsStore& operator<<(const MockWriterEvent& event)
    {
        std::unique_lock<hpx::lcos::local::spinlock> l(*insertMutex);
        delegate[event.rank] << event;
        return *this;
    }

    std::size_t size() const
    {
        return delegate.size();
    }

    void clear()
    {
        delegate.clear();
    }

    iterator begin()
    {
        return delegate.begin()->second.begin();
    }

    iterator end()
    {
        return delegate.begin()->second.end();
    }

    const MockWriterEvent& operator[](std::size_t index)
    {
        return delegate.begin()->second[index];
    }

    bool operator==(const ThreadSafeEventsStore& other) const
    {
        return delegate == other.delegate;
    }

    bool operator!=(const ThreadSafeEventsStore& other) const
    {
        return !(*this == other);
    }

    std::map<std::size_t, std::vector<MockWriterEvent> > delegate;

private:
    typename SharedPtr<hpx::lcos::local::spinlock>::Type insertMutex;
};

#endif

}

/**
 * This class implements the Writer interface, but simply records all
 * calls it receives so that unit tests can later evaluate the
 * correctness. Useful to check that a given Simulator basically
 * adheres to the required interface.
 */
template<typename CELL_TYPE=TestCell<2> >
class MockWriter :
        public Clonable<Writer<        CELL_TYPE>, MockWriter<CELL_TYPE> >,
        public Clonable<ParallelWriter<CELL_TYPE>, MockWriter<CELL_TYPE> >
{
public:
    typedef MockWriterHelpers::MockWriterEvent Event;
#ifdef LIBGEODECOMP_WITH_HPX
    typedef MockWriterHelpers::ThreadSafeEventsStore EventsStore;
#else
    typedef std::vector<Event> EventsStore;
#endif
    using Writer<CELL_TYPE>::DIM;

    explicit MockWriter(
        SharedPtr<EventsStore>::Type events,
        unsigned period = 1) :
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
        const Region<DIM>& validRegion,
        const Coord<DIM>& globalDimensions,
        unsigned step,
        WriterEvent event,
        std::size_t rank,
        bool lastCall)
    {
        stepFinished(step, event, rank, lastCall);
    }

private:
    SharedPtr<MockWriter::EventsStore>::Type events;

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
