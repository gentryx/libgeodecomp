
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_PARALLELIZATION_SERVER_HPXUPDATEGROUP_H
#define LIBGEODECOMP_PARALLELIZATION_SERVER_HPXUPDATEGROUP_H

#include <libgeodecomp/parallelization/hiparsimulator/parallelwriteradapter.h>
#include <libgeodecomp/parallelization/hiparsimulator/steereradapter.h>
#include <libgeodecomp/parallelization/hpxsimulator/patchlink.h>

#include <hpx/include/components.hpp>

namespace LibGeoDecomp {
namespace HpxSimulator {

template <class CELL_TYPE, class PARTITION, class STEPPER>
class UpdateGroup;

enum EventPoint {LOAD_BALANCING, END};
typedef SuperSet<EventPoint> EventSet;
typedef SuperMap<std::size_t, EventSet> EventMap;

template <class CELL_TYPE, class PARTITION, class STEPPER>
class UpdateGroupServer : public hpx::components::managed_component_base<
    UpdateGroupServer<CELL_TYPE, PARTITION, STEPPER> >
{
public:
    const static int DIM = CELL_TYPE::Topology::DIM;
    static const unsigned NANO_STEPS = CellAPITraitsFixme::SelectNanoSteps<CELL_TYPE>::VALUE;

    typedef
        UpdateGroup<CELL_TYPE, PARTITION, STEPPER> ClientType;

    typedef DisplacedGrid<
        CELL_TYPE, typename CELL_TYPE::Topology, true> GridType;
    typedef
        typename DistributedSimulator<CELL_TYPE>::WriterVector
        WriterVector;
    typedef
        typename DistributedSimulator<CELL_TYPE>::SteererVector
        SteererVector;
    typedef typename HiParSimulator::Stepper<CELL_TYPE>::PatchType PatchType;
    typedef
        typename HiParSimulator::Stepper<CELL_TYPE>::PatchProviderPtr
        PatchProviderPtr;
    typedef
        typename HiParSimulator::Stepper<CELL_TYPE>::PatchAccepterPtr
        PatchAccepterPtr;

    typedef boost::shared_ptr<typename PatchLink<GridType, ClientType>::Link> PatchLinkPtr;

    typedef
        typename HiParSimulator::Stepper<CELL_TYPE>::PatchAccepterVec
        PatchAccepterVec;
    typedef
        typename HiParSimulator::Stepper<CELL_TYPE>::PatchProviderVec
        PatchProviderVec;

    typedef
        typename PatchLink<GridType, ClientType>::Provider
        PatchLinkProviderType;

    typedef boost::shared_ptr<PatchLinkProviderType> PatchLinkProviderPtr;

    typedef
        typename PatchLink<GridType, ClientType>::Accepter
        PatchLinkAccepterType;

    typedef boost::shared_ptr<PatchLinkAccepterType> PatchLinkAccepterPtr;

    typedef
        HiParSimulator::PartitionManager<typename CELL_TYPE::Topology>
        PartitionManagerType;
    typedef typename PartitionManagerType::RegionVecMap RegionVecMap;

    typedef
        HiParSimulator::ParallelWriterAdapter<GridType, CELL_TYPE>
        ParallelWriterAdapterType;
    typedef HiParSimulator::SteererAdapter<GridType, CELL_TYPE> SteererAdapterType;

    typedef std::pair<std::size_t, std::size_t> StepPairType;

    UpdateGroupServer()
      : boundingBoxFuture(boundingBoxPromise.get_future())
      , initFuture(initPromise.get_future())
      , stopped(false)
    {}

    void init(
        const std::vector<ClientType>& updateGroups,
        //boost::shared_ptr<LoadBalancer> balancer,
        unsigned loadBalancingPeriod,
        unsigned ghostZoneWidth,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        const WriterVector& writers,
        const SteererVector& steerers
    )
    {
        this->updateGroups = updateGroups;
        this->initializer = initializer;
        this->loadBalancingPeriod = loadBalancingPeriod;
        //this->balancer = balancer;
        setRank();

        partitionManager.reset(new PartitionManagerType());
        CoordBox<DIM> box = initializer->gridBox();
        std::size_t numPartitions = updateGroups.size();

        boost::shared_ptr<PARTITION> partition(
            new PARTITION(
                box.origin,
                box.dimensions,
                0,
                initialWeights(box.dimensions.prod(), numPartitions)));

        partitionManager->resetRegions(
                box,
                partition,
                rank,
                ghostZoneWidth
            );

        boundingBoxPromise.set_value(partitionManager->ownRegion().boundingBox());
        SuperVector<hpx::future<CoordBox<DIM> > > boundingBoxesFutures;
        boundingBoxesFutures.reserve(numPartitions);
        // TODO: replace with proper all gather function
        BOOST_FOREACH(const ClientType& ug, updateGroups) {
            if(ug.gid() == this->get_gid()) {
                boundingBoxesFutures << boundingBoxFuture;
            } else {
                boundingBoxesFutures << ug.boundingBox();
            }
        }

        SuperVector<CoordBox<DIM> > boundingBoxes;
        boundingBoxes.reserve(numPartitions);

        BOOST_FOREACH(hpx::future<CoordBox<DIM> >& f, boundingBoxesFutures) {
            boundingBoxes << f.get();
        }
        partitionManager->resetGhostZones(boundingBoxes);

        long firstSyncPoint =
            initializer->startStep() * NANO_STEPS + ghostZoneWidth;

        // we have to hand over a list of all ghostzone senders as the
        // stepper will perform an initial update of the ghostzones
        // upon creation and we have to send those over to our neighbors.
        PatchAccepterVec patchAcceptersGhost;
        RegionVecMap map = partitionManager->getInnerGhostZoneFragments();
        for (typename RegionVecMap::iterator i = map.begin(); i != map.end(); ++i) {
            if (!i->second.back().empty()) {
                PatchLinkAccepterPtr link(
                    new PatchLinkAccepterType(
                        i->second.back(),
                        rank,
                        updateGroups[i->first]));
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
            PatchAccepterPtr adapterGhost(
                new ParallelWriterAdapterType(
                    boost::shared_ptr<ParallelWriter<CELL_TYPE> >(writer->clone()),
                    initializer->startStep(),
                    initializer->maxSteps(),
                    initializer->gridDimensions(),
                    rank,
                    false));
            PatchAccepterPtr adapterInnerSet(
                new ParallelWriterAdapterType(
                    boost::shared_ptr<ParallelWriter<CELL_TYPE> >(writer->clone()),
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

        stepper.reset(new STEPPER(
                          partitionManager,
                          initializer,
                          patchAcceptersGhost,
                          patchAcceptersInner));

        // the ghostzone receivers may be safely added after
        // initialization as they're only really needed when the next
        // ghostzone generation is being received.
        map = partitionManager->getOuterGhostZoneFragments();
        for (typename RegionVecMap::iterator i = map.begin(); i != map.end(); ++i) {
            if (!i->second.back().empty()) {
                PatchLinkProviderPtr link(
                    new PatchLinkProviderType(
                        i->second.back()));

                patchlinkProviderMap.insert(std::make_pair(i->first, link));

                addPatchProvider(link, HiParSimulator::Stepper<CELL_TYPE>::GHOST);
                patchLinks << link;

                link->charge(
                    firstSyncPoint,
                    PatchLink<GridType, ClientType>::ENDLESS,
                    ghostZoneWidth);

                link->setRegion(partitionManager->ownRegion());
            }
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
        initPromise.set_value();
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
        hpx::wait(boundingBoxFuture);
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

    void nanoStep(std::size_t remainingNanoSteps)
    {
        hpx::wait(initFuture);
        stopped = false;
        while (remainingNanoSteps > 0 && !stopped) {
            std::size_t hop = std::min(remainingNanoSteps, timeToNextEvent());
            stepper->update(hop);
            handleEvents();
            remainingNanoSteps -= hop;
        }
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(UpdateGroupServer, nanoStep, NanoStepAction);

    void stop()
    {
        stopped = true;
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(UpdateGroupServer, stop, StopAction);

    CoordBox<DIM> boundingBox()
    {
        return boundingBoxFuture.get();
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(UpdateGroupServer, boundingBox, BoundingBoxAction);

    void setOuterGhostZone(
        std::size_t srcRank,
        boost::shared_ptr<SuperVector<CELL_TYPE> > buffer,
        long nanoStep)
    {
        hpx::wait(initFuture);
        typename std::map<std::size_t, PatchLinkProviderPtr>::iterator patchlinkIter;
        patchlinkIter = patchlinkProviderMap.find(srcRank);
        BOOST_ASSERT(patchlinkIter != patchlinkProviderMap.end());

        patchlinkIter->second->setBuffer(buffer, nanoStep);
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(UpdateGroupServer, setOuterGhostZone, SetOuterGhostZoneAction);

    std::size_t getRank() const
    {
        return rank;
    }
private:
    std::vector<ClientType> updateGroups;

    boost::shared_ptr<HiParSimulator::Stepper<CELL_TYPE> > stepper;
    boost::shared_ptr<PartitionManagerType> partitionManager;
    SuperVector<PatchLinkPtr> patchLinks;
    SuperMap<std::size_t, PatchLinkProviderPtr> patchlinkProviderMap;
    boost::shared_ptr<Initializer<CELL_TYPE> > initializer;
    boost::shared_ptr<LoadBalancer> balancer;
    unsigned loadBalancingPeriod;
    std::size_t rank;
    EventMap events;

    hpx::lcos::local::promise<CoordBox<DIM> > boundingBoxPromise;
    hpx::future<CoordBox<DIM> > boundingBoxFuture;

    hpx::lcos::local::promise<void> initPromise;
    hpx::future<void> initFuture;

    boost::atomic<bool> stopped;

    void setRank()
    {
        rank = 0;
        BOOST_FOREACH(const ClientType& ug, updateGroups) {
            if(ug.gid() == this->get_gid()) {
                break;
            }
            ++rank;
        }
    }

    SuperVector<long> initialWeights(const long items, const long size) const
    {
        SuperVector<long> ret(size);
        long lastPos = 0;

        for (long i = 0; i < size; i++) {
            long currentPos = items * (i + 1) / size;
            ret[i] = currentPos - lastPos;
            lastPos = currentPos;
        }

        return ret;
    }

    void initEvents()
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
