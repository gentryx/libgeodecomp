
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_PARALLELIZATION_SERVER_HPXUPDATEGROUP_H
#define LIBGEODECOMP_PARALLELIZATION_SERVER_HPXUPDATEGROUP_H

#include <libgeodecomp/parallelization/hiparsimulator/parallelwriteradapter.h>
#include <libgeodecomp/parallelization/hiparsimulator/steereradapter.h>
#include <libgeodecomp/parallelization/hpxsimulator/patchlink.h>

#include <hpx/include/components.hpp>
#include <hpx/util/high_resolution_timer.hpp>

namespace LibGeoDecomp {
namespace HpxSimulator {

template <class CELL_TYPE, class PARTITION, class STEPPER>
class UpdateGroup;

enum EventPoint {LOAD_BALANCING, END};
typedef std::set<EventPoint> EventSet;
typedef std::map<std::size_t, EventSet> EventMap;

template <class CELL_TYPE, class PARTITION, class STEPPER>
class UpdateGroupServer : public hpx::components::managed_component_base<
    UpdateGroupServer<CELL_TYPE, PARTITION, STEPPER> >
{
public:
    typedef typename STEPPER::Topology Topology;
    const static int DIM = Topology::DIM;
    static const unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL_TYPE>::VALUE;

    typedef
        UpdateGroup<CELL_TYPE, PARTITION, STEPPER> ClientType;

    typedef DisplacedGrid<CELL_TYPE, Topology, true> GridType;
    typedef
        typename DistributedSimulator<CELL_TYPE>::WriterVector
        WriterVector;
    typedef
        typename DistributedSimulator<CELL_TYPE>::SteererVector
        SteererVector;


    typedef typename STEPPER::PatchType PatchType;
    typedef typename STEPPER::PatchProviderPtr PatchProviderPtr;
    typedef typename STEPPER::PatchAccepterPtr PatchAccepterPtr;
    typedef boost::shared_ptr<typename PatchLink<GridType, ClientType>::Link> PatchLinkPtr;
    typedef typename STEPPER::PatchAccepterVec PatchAccepterVec;
    typedef typename STEPPER::PatchProviderVec PatchProviderVec;

    typedef
        typename PatchLink<GridType, ClientType>::Provider
        PatchLinkProviderType;

    typedef boost::shared_ptr<PatchLinkProviderType> PatchLinkProviderPtr;

    typedef
        typename PatchLink<GridType, ClientType>::Accepter
        PatchLinkAccepterType;

    typedef boost::shared_ptr<PatchLinkAccepterType> PatchLinkAccepterPtr;

    typedef
        PartitionManager<Topology>
        PartitionManagerType;
    typedef typename PartitionManagerType::RegionVecMap RegionVecMap;

    typedef
        HiParSimulator::ParallelWriterAdapter<GridType, CELL_TYPE>
        ParallelWriterAdapterType;
    typedef HiParSimulator::SteererAdapter<GridType, CELL_TYPE> SteererAdapterType;

    typedef std::pair<std::size_t, std::size_t> StepPairType;

    UpdateGroupServer()
      : stopped(false)
    {}

    void initPartitions(const typename ClientType::InitData& initData, std::size_t global_idx)
    {
        initializer = initData.initializer;
        loadBalancingPeriod = initData.loadBalancingPeriod;
        ghostZoneWidth = initData.ghostZoneWidth;
        writers = initData.writers;
        steerers = initData.steerers;
        //this->balancer = balancer;
        rank = global_idx;
        ////////////////////////////////////////////////////////////////////////
        // Registering name.
        std::string name = "LibGeoDecomp.UpdateGroup.";
        name += boost::lexical_cast<std::string>(rank);
        hpx::agas::register_name_sync(name, this->get_gid());
        ////////////////////////////////////////////////////////////////////////

        partitionManager.reset(new PartitionManagerType());
        CoordBox<DIM> box = initializer->gridBox();

        boost::shared_ptr<PARTITION> partition(
            new PARTITION(
                box.origin,
                box.dimensions,
                0,
                initData.initialWeights));

        partitionManager->resetRegions(
                box,
                partition,
                rank,
                ghostZoneWidth
            );

        partitionManager->resetGhostZones(initData.boundingBoxes);
        
        long firstSyncPoint =
            initializer->startStep() * NANO_STEPS + ghostZoneWidth;

        const RegionVecMap& outerMap = partitionManager->getOuterGhostZoneFragments();
        for (typename RegionVecMap::const_iterator i = outerMap.begin(); i != outerMap.end(); ++i) {
            if (!i->second.empty() && !i->second.back().empty()) {
                if(i->second.back().empty()) {
                    continue;
                }

                PatchLinkProviderPtr link(
                    new PatchLinkProviderType(
                        i->second.back()));
                link->charge(
                    firstSyncPoint,
                    PatchLink<GridType, ClientType>::ENDLESS,
                    ghostZoneWidth);

                patchlinkProviderMap.insert(std::make_pair(i->first, link));
            }
        }
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(UpdateGroupServer, initPartitions, InitPartitionsAction);

    void init()
    {
        long firstSyncPoint =
            initializer->startStep() * NANO_STEPS + ghostZoneWidth;

        // we have to hand over a list of all ghostzone senders as the
        // stepper will perform an initial update of the ghostzones
        // upon creation and we have to send those over to our neighbors.
        PatchAccepterVec patchAcceptersGhost;
        const RegionVecMap& innerMap = partitionManager->getInnerGhostZoneFragments();
        for (typename RegionVecMap::const_iterator i = innerMap.begin(); i != innerMap.end(); ++i) {
            if (!i->second.empty() && !i->second.back().empty()) {
                if(i->second.back().empty()) {
                    continue;
                }

                PatchLinkAccepterPtr link(
                    new PatchLinkAccepterType(
                        i->second.back(),
                        rank,
                        getUpdateGroup(i->first)));
                patchAcceptersGhost.push_back(link);
                patchLinks << link;

                link->charge(
                    firstSyncPoint,
                    PatchLink<GridType, ClientType>::ENDLESS,
                    ghostZoneWidth);

                link->setRegion(partitionManager->ownRegion());
            }
        }

        PatchAccepterVec patchAcceptersInner;

        // Convert writers to patch accepters
        BOOST_FOREACH(const typename WriterVector::value_type& writer, writers) {
            boost::shared_ptr<ParallelWriter<CELL_TYPE> > writerPtr(writer->clone());
            PatchAccepterPtr adapterGhost(
                new ParallelWriterAdapterType(
                    writerPtr,
                    initializer->startStep(),
                    initializer->maxSteps(),
                    initializer->gridDimensions(),
                    rank,
                    false));
            PatchAccepterPtr adapterInnerSet(
                new ParallelWriterAdapterType(
                    writerPtr,
                    initializer->startStep(),
                    initializer->maxSteps(),
                    initializer->gridDimensions(),
                    rank,
                    true));
            // notify PatchAccepters of the updategroups region:
            adapterGhost->setRegion(partitionManager->ownRegion());
            adapterInnerSet->setRegion(partitionManager->ownRegion());

            patchAcceptersGhost.push_back(adapterGhost);
            patchAcceptersInner.push_back(adapterInnerSet);
        }

        typedef typename std::map<std::size_t, PatchLinkProviderPtr>::iterator patchlinkIter;
        for(patchlinkIter it = patchlinkProviderMap.begin(); it != patchlinkProviderMap.end(); ++it) {
        }

        stepper.reset(new STEPPER(
                          partitionManager,
                          initializer,
                          patchAcceptersGhost,
                          patchAcceptersInner));

        // the ghostzone receivers may be safely added after
        // initialization as they're only really needed when the next
        // ghostzone generation is being received.
        for(patchlinkIter it = patchlinkProviderMap.begin(); it != patchlinkProviderMap.end(); ++it) {
            addPatchProvider(it->second, HiParSimulator::Stepper<CELL_TYPE>::GHOST);
            patchLinks << it->second;

            it->second->setRegion(partitionManager->ownRegion());
        }

        // Convert steerer to patch accepters
        BOOST_FOREACH(const typename SteererVector::value_type& steerer, steerers) {
            // two adapters needed, just as for the writers
            PatchProviderPtr adapterGhost(
                new SteererAdapterType(
                    steerer,
                    initializer->startStep(),
                    initializer->maxSteps(),
                    initializer->gridDimensions(),
                    rank,
                    false));
            PatchProviderPtr adapterInnerSet(
                new SteererAdapterType(
                    steerer,
                    initializer->startStep(),
                    initializer->maxSteps(),
                    initializer->gridDimensions(),
                    rank,
                    false));

            adapterGhost->setRegion(partitionManager->ownRegion());
            adapterInnerSet->setRegion(partitionManager->ownRegion());

            addPatchProvider(adapterGhost, HiParSimulator::Stepper<CELL_TYPE>::GHOST);
            addPatchProvider(adapterInnerSet, HiParSimulator::Stepper<CELL_TYPE>::INNER_SET);
        }

        initEvents();
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(UpdateGroupServer, init, InitAction);

    void addPatchProvider(
        const PatchProviderPtr& patchProvider,
        const PatchType& patchType)
    {
        stepper->addPatchProvider(patchProvider, patchType);
    }

    void addPatchAccepter(
        const PatchAccepterPtr& patchAccepter,
        const PatchType& patchType)
    {
        stepper->addPatchAccepter(patchAccepter, patchType);
    }

    StepPairType currentStep() const
    {
        if(stepper) {
            return stepper->currentStep();
        }
        else {
            return std::make_pair(initializer->startStep(), 0u);
        }
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(UpdateGroupServer, currentStep, CurrentStepAction);

    std::size_t getStep() const
    {
        return currentStep().first;
    }

    Chronometer nanoStep(std::size_t remainingNanoSteps)
    {
        Chronometer chrono;
        {
            TimeTotal t(&chrono);
            stopped = false;
            while (remainingNanoSteps > 0 && !stopped) {
                std::size_t hop = std::min(remainingNanoSteps, timeToNextEvent());
                stepper->update(hop);
                handleEvents();
                remainingNanoSteps -= hop;
            }
        }
        chrono += stepper->statistics();
        return chrono;
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(UpdateGroupServer, nanoStep, NanoStepAction);

    void stop()
    {
        stopped = true;
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(UpdateGroupServer, stop, StopAction);

    void setOuterGhostZone(
        std::size_t srcRank,
        boost::shared_ptr<std::vector<CELL_TYPE> > buffer,
        long nanoStep)
    {
        typename std::map<std::size_t, PatchLinkProviderPtr>::iterator patchlinkIter;
        patchlinkIter = patchlinkProviderMap.find(srcRank);
        if(patchlinkIter == patchlinkProviderMap.end()) {
            std::cerr << rank << " setting outer ghostzone from unknown rank: " << srcRank << "\ngot these ranks:\n";
            typedef std::pair<std::size_t, PatchLinkProviderPtr> pair_type;
            BOOST_FOREACH(const pair_type& p, patchlinkProviderMap) {
                std::cerr << rank << " " << p.first << "\n";
            }
            return;
        }
        BOOST_ASSERT(patchlinkIter != patchlinkProviderMap.end());

        patchlinkIter->second->setBuffer(buffer, nanoStep);
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(UpdateGroupServer, setOuterGhostZone, SetOuterGhostZoneAction);

    double getCellSpeed(APITraits::FalseType) const
    {
        return 1.0;
    }

    double getCellSpeed(APITraits::TrueType) const
    {
        return CELL_TYPE::speed();
    }

    double speed()
    {
        return getCellSpeed(typename APITraits::SelectSpeed<CELL_TYPE>::Value());
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(UpdateGroupServer, speed, SpeedAction);

    std::size_t getRank() const
    {
        return rank;
    }
private:
    std::map<std::size_t, ClientType> updateGroups;

    boost::shared_ptr<HiParSimulator::Stepper<CELL_TYPE> > stepper;
    boost::shared_ptr<PartitionManagerType> partitionManager;
    std::vector<PatchLinkPtr> patchLinks;
    std::map<std::size_t, PatchLinkProviderPtr> patchlinkProviderMap;
    boost::shared_ptr<Initializer<CELL_TYPE> > initializer;
    boost::shared_ptr<LoadBalancer> balancer;
    unsigned loadBalancingPeriod;
    unsigned ghostZoneWidth;
    WriterVector writers;
    SteererVector steerers;
    std::size_t rank;
    EventMap events;

    boost::atomic<bool> stopped;

    ClientType getUpdateGroup(std::size_t dstRank)
    {
        typename std::map<std::size_t, ClientType>::iterator it = updateGroups.find(dstRank);
        if(it == updateGroups.end()) {
            hpx::id_type updateGroupId;
            std::string name = "LibGeoDecomp.UpdateGroup.";
            name += boost::lexical_cast<std::string>(dstRank);
            while(true) {
                updateGroupId = hpx::agas::resolve_name_sync(name);

                if(!updateGroupId) {
                    hpx::this_thread::suspend(boost::posix_time::seconds(1));
                }
                else {
                    break;
                }
            }
            it = updateGroups.insert(it, std::make_pair(dstRank, ClientType(updateGroupId)));
        }
        return it->second;
    }

    void initEvents()
    {
        events.clear();
        long lastNanoStep = initializer->maxSteps() * NANO_STEPS;
        events[lastNanoStep].insert(END);

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
        events[nextLoadBalancing].insert(LOAD_BALANCING);
    }

    std::size_t currentNanoStep() const
    {
        std::pair<std::size_t, std::size_t> now = currentStep();
        return now.first * NANO_STEPS + now.second;
    }

    std::size_t timeToNextEvent()
    {
        return events.begin()->first - currentNanoStep();
    }

    std::size_t timeToLastEvent()
    {
        return events.rbegin()->first - currentNanoStep();
    }

    void balanceLoad()
    {
    }
};

}
}

#endif
#endif
