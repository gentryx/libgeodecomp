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

void gatherAndBroadcastLocalityIndices(
    double speedGuide,
    std::vector<double> *globalUpdateGroupSpeeds,
    std::vector<std::size_t> *localityIndices,
    const std::string& basename,
    const std::vector<double> updateGroupSpeeds);

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
     * for the HiParSimulator. The vector updateGroupSpeeds controls
     * how many UpdateGroups will be created and how large their
     * individual domain should be.
     */
    inline HpxSimulator(
        Initializer<CELL_TYPE> *initializer,
        const std::vector<double> updateGroupSpeeds = std::vector<double>(1, 1.0),
        LoadBalancer *balancer = 0,
        const unsigned loadBalancingPeriod = 1,
        const unsigned ghostZoneWidth = 1,
        std::string basename = "HPXSimulator") :
        ParentType(initializer),
        updateGroupSpeeds(updateGroupSpeeds),
        balancer(balancer),
        loadBalancingPeriod(loadBalancingPeriod * NANO_STEPS),
        ghostZoneWidth(ghostZoneWidth),
        basename(basename)
    {
        HpxSimulatorHelpers::gatherAndBroadcastLocalityIndices(
            APITraits::SelectSpeedGuide<CELL_TYPE>::value(),
            &globalUpdateGroupSpeeds,
            &localityIndices,
            basename,
            updateGroupSpeeds);

        std::cout << "speeds: " << globalUpdateGroupSpeeds << "\n";
        std::cout << "indices: " << localityIndices << "\n";
    }

    inline void run()
    {
        initSimulation();
        // statistics = nanoStep(timeToLastEvent());
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
        // if (initialized) {
            // return typename UpdateGroupType::ComponentType::CurrentStepAction()(updateGroupsIds[0]).first;
        // } else {
        //     return initializer->startStep();
        // }

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

    std::vector<double> updateGroupSpeeds;
    boost::shared_ptr<LoadBalancer> balancer;
    unsigned loadBalancingPeriod;
    unsigned ghostZoneWidth;
    std::string basename;
    EventMap events;
    PartitionManager<Topology> partitionManager;

    std::vector<boost::shared_ptr<UpdateGroupType> > updateGroups;
    std::vector<double> globalUpdateGroupSpeeds;
    std::vector<std::size_t> localityIndices;

    std::vector<Chronometer> statistics;

    inline void initSimulation()
    {
        if (updateGroups.size() != 0) {
            return;
        }

        CoordBox<DIM> box = initializer->gridBox();

        std::vector<std::size_t> weights = initialWeights(
            box.dimensions.prod(),
            globalUpdateGroupSpeeds);

        std::cout << "weights: " << weights << "\n";
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

    /**
     * computes an initial weight distribution of the work items (i.e.
     * number of cells in the simulation space). rankSpeeds gives an
     * estimate of the relative performance of the different ranks
     * (good when running on heterogeneous systems, e.g. clusters
     * comprised of multiple genrations of nodes or x86 clusters with
     * additional Xeon Phi accelerators).
     */
    // fixme: stolen from HiParSimulator
    std::vector<std::size_t> initialWeights(std::size_t items, const std::vector<double> rankSpeeds) const
    {
        std::size_t size = rankSpeeds.size();
        double totalSum = sum(rankSpeeds);
        std::vector<std::size_t> ret(size);

        std::size_t lastPos = 0;
        double partialSum = 0.0;
        for (std::size_t i = 0; i < size - 1; ++i) {
            partialSum += rankSpeeds[i];
            std::size_t nextPos = items * partialSum / totalSum;
            ret[i] = nextPos - lastPos;
            lastPos = nextPos;
        }
        ret[size - 1] = items - lastPos;

        return ret;
    }

};

}
}

#endif
#endif
