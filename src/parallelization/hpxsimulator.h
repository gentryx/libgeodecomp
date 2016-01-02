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
#include <libgeodecomp/parallelization/nesting/eventpoint.h>
#include <libgeodecomp/parallelization/nesting/hpxupdategroup.h>
#include <libgeodecomp/parallelization/nesting/parallelwriteradapter.h>
#include <libgeodecomp/parallelization/nesting/steereradapter.h>
#include <libgeodecomp/parallelization/nesting/stepper.h>
#include <libgeodecomp/parallelization/nesting/vanillastepper.h>

namespace LibGeoDecomp {
namespace HpxSimulatorHelpers {

void gatherAndBroadcastLocalityIndices(
    double speedGuide,
    std::vector<double> *globalUpdateGroupSpeeds,
    std::vector<std::size_t> *localityIndices,
    const std::string& basename,
    const std::vector<double> updateGroupSpeeds);

}

template<
    class CELL_TYPE,
    class PARTITION,
    class STEPPER=LibGeoDecomp::VanillaStepper<CELL_TYPE, UpdateFunctorHelpers::ConcurrencyEnableHPX>
>
class HpxSimulator : public DistributedSimulator<CELL_TYPE>
{
public:
    friend class HpxSimulatorTest;
    using DistributedSimulator<CELL_TYPE>::NANO_STEPS;
    typedef typename DistributedSimulator<CELL_TYPE>::Topology Topology;
    typedef LibGeoDecomp::DistributedSimulator<CELL_TYPE> ParentType;
    typedef HPXUpdateGroup<CELL_TYPE> UpdateGroupType;
    typedef typename ParentType::GridType GridType;
    typedef LibGeoDecomp::ParallelWriterAdapter<typename UpdateGroupType::GridType, CELL_TYPE> ParallelWriterAdapterType;
    typedef LibGeoDecomp::SteererAdapter<typename UpdateGroupType::GridType, CELL_TYPE> SteererAdapterType;
    typedef typename UpdateGroupType::PatchAccepterVec PatchAccepterVec;
    typedef typename UpdateGroupType::PatchProviderVec PatchProviderVec;
    typedef typename UpdateGroupType::PatchAccepterPtr PatchAccepterPtr;
    typedef typename UpdateGroupType::PatchProviderPtr PatchProviderPtr;

    static const int DIM = Topology::DIM;

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
        std::string basename = "/HPXSimulator") :
        ParentType(initializer),
        updateGroupSpeeds(updateGroupSpeeds),
        balancer(balancer),
        loadBalancingPeriod(loadBalancingPeriod * NANO_STEPS),
        ghostZoneWidth(ghostZoneWidth),
        basename(basename),
        rank(hpx::get_locality_id())
    {
        HpxSimulatorHelpers::gatherAndBroadcastLocalityIndices(
            APITraits::SelectSpeedGuide<CELL_TYPE>::value(),
            &globalUpdateGroupSpeeds,
            &localityIndices,
            basename,
            updateGroupSpeeds);

        for (std::size_t i = localityIndices[rank + 0]; i < localityIndices[rank + 1]; ++i) {
            steererAdaptersGhost[i] = PatchProviderVec();
            steererAdaptersInner[i] = PatchProviderVec();

            writerAdaptersGhost[i] = PatchAccepterVec();
            writerAdaptersInner[i] = PatchAccepterVec();
        }
    }

    inline void run()
    {
        initSimulation();
        nanoStep(timeToLastEvent());
    }

    inline void step()
    {
        initSimulation();
        nanoStep(NANO_STEPS);
    }

    virtual unsigned getStep() const
    {
        if (updateGroups.size() == 0) {
            return initializer->startStep();
        }

        return updateGroups[0]->currentStep().first;
    }

    virtual void addSteerer(Steerer<CELL_TYPE> *steerer)
    {
        for (std::size_t i = localityIndices[rank + 0]; i < localityIndices[rank + 1]; ++i) {
            // fixme: check events just like writer events
            boost::shared_ptr<Steerer<CELL_TYPE> > steererSharedPointer(steerer);

            // two adapters needed, just as for the writers
            typename UpdateGroupType::PatchProviderPtr adapterGhost(
                new SteererAdapterType(
                    steererSharedPointer,
                    initializer->startStep(),
                    initializer->maxSteps(),
                    false));

            typename UpdateGroupType::PatchProviderPtr adapterInnerSet(
                new SteererAdapterType(
                    steererSharedPointer,
                    initializer->startStep(),
                    initializer->maxSteps(),
                    true));

            steererAdaptersGhost[i].push_back(adapterGhost);
            steererAdaptersInner[i].push_back(adapterInnerSet);

            if ((i + 1) < localityIndices[rank + 1]) {
                steerer = steerer->clone();
            }
        }
    }

    virtual void addWriter(ParallelWriter<CELL_TYPE> *writer)
    {
        for (std::size_t i = localityIndices[rank + 0]; i < localityIndices[rank + 1]; ++i) {
            boost::shared_ptr<ParallelWriter<CELL_TYPE> > writerSharedPointer(writer);
            // we need two adapters as each ParallelWriter needs to be
            // notified twice: once for the (inner) ghost zone, and once
            // for the inner set.
            PatchAccepterPtr adapterGhost(
                new ParallelWriterAdapterType(
                    writerSharedPointer,
                    initializer->startStep(),
                    initializer->maxSteps(),
                    false));
            PatchAccepterPtr adapterInnerSet(
                new ParallelWriterAdapterType(
                    writerSharedPointer,
                    initializer->startStep(),
                    initializer->maxSteps(),
                    true));

            writerAdaptersGhost[i].push_back(adapterGhost);
            writerAdaptersInner[i].push_back(adapterInnerSet);

            if ((i + 1) < localityIndices[rank + 1]) {
                writer = writer->clone();
            }
        }
    }

    std::size_t numUpdateGroups() const
    {
        return updateGroups.size();
    }

    std::vector<Chronometer> gatherStatistics()
    {
        // fixme
        Chronometer statistics;
        return std::vector<Chronometer>(1, statistics);
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

    std::vector<boost::shared_ptr<UpdateGroupType> > updateGroups;
    std::vector<double> globalUpdateGroupSpeeds;
    std::vector<std::size_t> localityIndices;
    std::size_t rank;

    std::map<std::size_t, typename UpdateGroupType::PatchProviderVec> steererAdaptersGhost;
    std::map<std::size_t, typename UpdateGroupType::PatchProviderVec> steererAdaptersInner;
    std::map<std::size_t, typename UpdateGroupType::PatchAccepterVec> writerAdaptersGhost;
    std::map<std::size_t, typename UpdateGroupType::PatchAccepterVec> writerAdaptersInner;

    inline void initSimulation()
    {
        if (updateGroups.size() != 0) {
            return;
        }

        CoordBox<DIM> box = initializer->gridBox();

        std::vector<std::size_t> weights = initialWeights(
            box.dimensions.prod(),
            globalUpdateGroupSpeeds);

        boost::shared_ptr<PARTITION> partition(
            new PARTITION(
                box.origin,
                box.dimensions,
                0,
                weights));

        std::vector<hpx::future<boost::shared_ptr<UpdateGroupType> > > updateGroupCreationFutures;

        for (std::size_t i = localityIndices[rank + 0]; i < localityIndices[rank + 1]; ++i) {
            updateGroupCreationFutures << hpx::async(&HpxSimulator::createUpdateGroup, this, i, partition);
        }
        updateGroups = hpx::util::unwrapped(std::move(updateGroupCreationFutures));

        for (std::size_t i = localityIndices[rank + 0]; i < localityIndices[rank + 1]; ++i) {
            writerAdaptersGhost[i].clear();
            writerAdaptersInner[i].clear();
            steererAdaptersGhost[i].clear();
            steererAdaptersInner[i].clear();
        }

        initEvents();
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

    void nanoStep(std::size_t remainingNanoSteps)
    {
        std::vector<hpx::future<void> > updateFutures;
        updateFutures.reserve(updateGroups.size());

        for (auto& i: updateGroups) {
            updateFutures << hpx::async(&UpdateGroupType::update, i, remainingNanoSteps);
        }

        hpx::lcos::wait_all(std::move(updateFutures));
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

    boost::shared_ptr<UpdateGroupType> createUpdateGroup(
        std::size_t rank,
        boost::shared_ptr<PARTITION> partition)
    {
        CoordBox<DIM> box = initializer->gridBox();

        boost::shared_ptr<UpdateGroupType> ret;
        ret.reset(new UpdateGroupType(
                      partition,
                      box,
                      ghostZoneWidth,
                      initializer,
                      reinterpret_cast<STEPPER*>(0),
                      writerAdaptersGhost[rank],
                      writerAdaptersInner[rank],
                      steererAdaptersGhost[rank],
                      steererAdaptersInner[rank],
                      basename,
                      rank));
        return ret;
    }
};

}

#endif
#endif
