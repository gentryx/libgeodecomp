#ifndef LIBGEODECOMP_IO_MOCKSTEERER_H
#define LIBGEODECOMP_IO_MOCKSTEERER_H

#include <sstream>
#include <libgeodecomp/io/steerer.h>

#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/include/threads.hpp>
#include <hpx/concurrency/spinlock.hpp>
#endif

namespace LibGeoDecomp {

namespace MockSteererHelpers {

/**
 * Internal helper class which records a single invocation of the
 * MockWriter's interface.
 */
class MockSteererEvent
{
public:
    MockSteererEvent(unsigned step, SteererEvent event, std::size_t rank, bool lastCall) :
        step(step),
        event(event),
        rank(rank),
        lastCall(lastCall)
    {}

    bool operator==(const MockSteererEvent& other) const
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
        buf << "MockSteererEvent(" << step << ", ";
        switch(event) {
        case STEERER_INITIALIZED:
            buf << "STEERER_INITIALIZED";
            break;
        case STEERER_NEXT_STEP:
            buf << "STEERER_NEXT_STEP";
            break;
        case STEERER_ALL_DONE:
            if (step != -1) {
                buf << "STEERER_ALL_DONE";
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
    SteererEvent event;
    std::size_t rank;
    bool lastCall;
};

#ifdef LIBGEODECOMP_WITH_HPX

/**
 * Records events received by the MockWriter for later inspection.
 * Obviously strives to be thread-safe, at least while recording.
 */
class ThreadSafeEventsStore
{
public:
    typedef std::vector<MockSteererEvent>::iterator iterator;

    ThreadSafeEventsStore() :
        insertMutex(new hpx::lcos::local::spinlock())
    {}

    ThreadSafeEventsStore& operator<<(const MockSteererEvent& event)
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

    const MockSteererEvent& operator[](std::size_t index)
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

    std::map<std::size_t, std::vector<MockSteererEvent> > delegate;

private:
    SharedPtr<hpx::lcos::local::spinlock>::Type insertMutex;
};

#endif

}

/**
 * Like MockWriter, this class records all calls to its Writer
 * interface. Useful when testing new Simulators.
 */
template<typename CELL_TYPE=TestCell<2> >
class MockSteerer : public Steerer<CELL_TYPE>
{
public:
    typedef typename Steerer<CELL_TYPE>::SteererFeedback SteererFeedback;
    typedef typename Steerer<CELL_TYPE>::Topology Topology;
    typedef typename Steerer<CELL_TYPE>::GridType GridType;
    typedef MockSteererHelpers::MockSteererEvent Event;
#ifdef LIBGEODECOMP_WITH_HPX
    typedef MockSteererHelpers::ThreadSafeEventsStore EventsStore;
#else
    typedef std::vector<Event> EventsStore;
#endif
    static const int DIM = Topology::DIM;

    MockSteerer(unsigned period, SharedPtr<EventsStore>::Type events)  :
        Steerer<CELL_TYPE>(period),
        events(events)
    {}

    virtual ~MockSteerer()
    {
        *events << Event(-1, STEERER_ALL_DONE, -1, true);
    }

    Steerer<CELL_TYPE> *clone() const
    {
        return new MockSteerer(*this);
    }

    virtual void nextStep(
        GridType *grid,
        const Region<DIM>& validRegion,
        const Coord<DIM>& globalDimensions,
        unsigned step,
        SteererEvent event,
        std::size_t rank,
        bool lastCall,
        SteererFeedback *feedback)
    {
        *events << Event(step, event, rank, lastCall);
    }

private:
    SharedPtr<EventsStore>::Type events;
};

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const typename MockSteererHelpers::MockSteererEvent& event)
{
    __os << event.toString();
    return __os;
}

}

#endif
