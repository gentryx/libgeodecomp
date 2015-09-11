#ifndef LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/set.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/lcos/broadcast.hpp>

#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/geometry/partitions/stripingpartition.h>
#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <libgeodecomp/parallelization/distributedsimulator.h>
#include <libgeodecomp/parallelization/hpxsimulator/hpxstepper.h>
#include <libgeodecomp/parallelization/hpxsimulator/updategroup.h>

namespace LibGeoDecomp {
namespace HpxSimulator {

namespace HpxSimulatorHelpers {

// fixme: return these from gatherAndBroadcastLocalityIndices
extern std::map<std::string, hpx::lcos::local::promise<std::vector<double> > > globalUpdateGroupWeights;
extern std::map<std::string, hpx::lcos::local::promise<std::vector<std::size_t> > > localityIndices;


void gatherAndBroadcastLocalityIndices(
    const std::string& basename,
    const std::vector<double> updateGroupWeights);

}

}
}

namespace LibGeoDecomp {
namespace HpxSimulator {


enum EventPoint {LOAD_BALANCING, END};
typedef std::set<EventPoint> EventSet;
typedef std::map<long, EventSet> EventMap;

template<
    class CELL_TYPE,
    class PARTITION,
    class STEPPER=LibGeoDecomp::HiParSimulator::HpxStepper<CELL_TYPE>
>
class HpxSimulator : public DistributedSimulator<CELL_TYPE>
{
public:
    friend class HpxSimulatorTest;
    using DistributedSimulator<CELL_TYPE>::NANO_STEPS;
    typedef typename DistributedSimulator<CELL_TYPE>::Topology Topology;
    typedef LibGeoDecomp::DistributedSimulator<CELL_TYPE> ParentType;
    typedef UpdateGroup<CELL_TYPE> UpdateGroupType;
    typedef typename ParentType::GridType GridType;

    static const int DIM = Topology::DIM;

public:
    /**
     * Creates an HpxSimulator. Parameters are essentially the same as
     * for the HiParSimulator. The vector updateGroupWeights controls
     * how many UpdateGroups will be created and how large their
     * individual domain should be.
     */
    inline HpxSimulator(
        Initializer<CELL_TYPE> *initializer,
        const std::vector<double> updateGroupWeights = std::vector<double>(1, 1.0),
        LoadBalancer *balancer = 0,
        const unsigned loadBalancingPeriod = 1,
        const unsigned ghostZoneWidth = 1) :
        ParentType(initializer),
        updateGroupWeights(updateGroupWeights),
        balancer(balancer),
        loadBalancingPeriod(loadBalancingPeriod * NANO_STEPS),
        ghostZoneWidth(ghostZoneWidth)
    {
        std::string basename = "fixme";
        HpxSimulatorHelpers::gatherAndBroadcastLocalityIndices(basename, updateGroupWeights);
        std::cout << "indices: " << HpxSimulatorHelpers::localityIndices[basename].get_future().get() << "\n";
        std::cout << "weights: " << HpxSimulatorHelpers::globalUpdateGroupWeights[basename].get_future().get() << "\n";
    }

    inline void run()
    {
        initSimulation();

        statistics = nanoStep(timeToLastEvent());
    }

    void stop()
    {
        // hpx::lcos::broadcast<typename UpdateGroupType::ComponentType::StopAction>(
        //     updateGroupsIds
        // ).wait();
    }

    inline void step()
    {
        // init();
        // nanoStep(NANO_STEPS);
    }

    virtual unsigned getStep() const
    {
        if (initialized) {
            // return typename UpdateGroupType::ComponentType::CurrentStepAction()(updateGroupsIds[0]).first;
        } else {
            return initializer->startStep();
        }

        return 0;
    }

    virtual void addSteerer(Steerer<CELL_TYPE> *steerer)
    {
        DistributedSimulator<CELL_TYPE>::addSteerer(steerer);
    }

    virtual void addWriter(ParallelWriter<CELL_TYPE> *writer)
    {
        DistributedSimulator<CELL_TYPE>::addWriter(writer);
    }

    std::size_t numUpdateGroups() const
    {
        return updateGroups.size();
    }

    std::vector<Chronometer> gatherStatistics()
    {
        return statistics;
    }

private:
    using DistributedSimulator<CELL_TYPE>::initializer;
    using DistributedSimulator<CELL_TYPE>::steerers;
    using DistributedSimulator<CELL_TYPE>::writers;

    std::vector<double> updateGroupWeights;
    boost::shared_ptr<LoadBalancer> balancer;
    unsigned loadBalancingPeriod;
    unsigned ghostZoneWidth;
    EventMap events;
    PartitionManager<Topology> partitionManager;
    std::vector<boost::shared_ptr<UpdateGroupType> > updateGroups;


    boost::atomic<bool> initialized;
    std::vector<Chronometer> statistics;

    inline void initSimulation()
    {
        if (updateGroups.size() != 0) {
            return;
        }

        CoordBox<DIM> box = initializer->gridBox();

        double myBaseSpeed = APITraits::SelectSpeedGuide<CELL_TYPE>::value();
        std::vector<double> mySpeeds;
        for (double weight: updateGroupWeights) {
            mySpeeds << myBaseSpeed * weight;
        }
        // fixme: allgather myspeeds here!

        // std::vector<std::size_t> weights = initialWeights(
        //     box.dimensions.prod(),
        //     rankSpeeds);

        // boost::shared_ptr<PARTITION> partition(
        //     new PARTITION(
        //         box.origin,
        //         box.dimensions,
        //         0,
        //         weights));

        // updateGroup.reset(
        //     new UpdateGroupType(
        //         partition,
        //         box,
        //         ghostZoneWidth,
        //         initializer,
        //         static_cast<STEPPER*>(0),
        //         writerAdaptersGhost,
        //         writerAdaptersInner,
        //         steererAdaptersGhost,
        //         steererAdaptersInner,
        //         mpiLayer.communicator()));

        // writerAdaptersGhost.clear();
        // writerAdaptersInner.clear();

        // initEvents();
    }

    // fixme: reduce duplication from HiParSimulator
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
        events[nextLoadBalancing] << LOAD_BALANCING;
    }

    inline long currentNanoStep() const
    {
        std::pair<int, int> now = updateGroups[0]->currentStep();
        return (long)now.first * NANO_STEPS + now.second;
    }

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

    inline void balanceLoad()
    {
        // fixme: do we need this after all?
    }

    std::vector<Chronometer> nanoStep(std::size_t remainingNanoSteps)
    {
        return std::vector<Chronometer>();
    }

    std::vector<std::size_t> initialWeights(const std::size_t items, const std::size_t size) const
    {
        std::vector<std::size_t> ret(size);

        return ret;
    }
};

}
}

#endif
#endif
