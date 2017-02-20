#ifndef LIBGEODECOMP_PARALLELIZATION_HIERARCHICALSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_HIERARCHICALSIMULATOR_H

#include <libgeodecomp/parallelization/distributedsimulator.h>
#include <libgeodecomp/parallelization/nesting/eventpoint.h>

namespace LibGeoDecomp {

/**
 * This is the base class for all simulators that are using our nested
 * parallelism architecture. The latter is built on UpdateGroups and
 * Steppers.
 */
template<typename CELL>
class HierarchicalSimulator : public DistributedSimulator<CELL>
{
protected:
    using DistributedSimulator<CELL>::NANO_STEPS;
    using DistributedSimulator<CELL>::initializer;
    EventMap events;
    unsigned loadBalancingPeriod;
    bool enableFineGrainedParallelism;

    HierarchicalSimulator(
        Initializer<CELL> *initializer,
        unsigned loadBalancingPeriod,
        bool enableFineGrainedParallelism) :
        DistributedSimulator<CELL>(initializer),
        loadBalancingPeriod(loadBalancingPeriod),
        enableFineGrainedParallelism(enableFineGrainedParallelism)
    {}

    inline void initEvents()
    {
        events.clear();
        long lastNanoStep = initializer->maxSteps() * NANO_STEPS;
        events[lastNanoStep] << END;

        insertNextLoadBalancingEvent();
    }

    inline void handleEvents()
    {
        if (currentNanoStep() > events.begin()->first) {
            throw std::logic_error("stale event found, should have been handled previously");
        }
        if (currentNanoStep() < events.begin()->first) {
            // don't need to handle future events now
            return;
        }

        const EventSet& curEvents = events.begin()->second;
        for (EventSet::const_iterator i = curEvents.begin(); i != curEvents.end(); ++i) {
            if (*i == LOAD_BALANCING) {
                balanceLoad();
                insertNextLoadBalancingEvent();
            }
        }
        events.erase(events.begin());
    }

    inline void insertNextLoadBalancingEvent()
    {
        long nextLoadBalancing = currentNanoStep() + loadBalancingPeriod;
        long lastNanoStep = initializer->maxSteps() * NANO_STEPS;

        if (nextLoadBalancing <= lastNanoStep) {
            events[nextLoadBalancing] << LOAD_BALANCING;
        }
    }

    virtual long currentNanoStep() const = 0;

    virtual void balanceLoad() = 0;

    /**
     * returns the number of nano steps until the next event needs to be handled.
     */
    inline long timeToNextEvent() const
    {
        return events.begin()->first - currentNanoStep();
    }

    /**
     * returns the number of nano steps until simulation end.
     */
    inline long timeToLastEvent() const
    {
        return  events.rbegin()->first - currentNanoStep();
    }
};

}

#endif
