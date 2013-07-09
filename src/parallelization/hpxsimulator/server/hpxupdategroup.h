
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX
#ifndef LIBGEODECOMP_PARALLELIZATION_SERVER_HPXUPDATEGROUP_H
#define LIBGEODECOMP_PARALLELIZATION_SERVER_HPXUPDATEGROUP_H

#include <libgeodecomp/parallelization/hiparsimulator/parallelwriteradapter.h>
#include <libgeodecomp/parallelization/hiparsimulator/steereradapter.h>
#include <libgeodecomp/parallelization/hpxsimulator/hpxpatchlinks.h>

#include <hpx/include/components.hpp>

namespace LibGeoDecomp {
namespace HpxSimulator {

template <class CELL_TYPE, class PARTITION, class STEPPER>
class HpxUpdateGroup;

namespace Server {

template <class CELL_TYPE, class PARTITION, class STEPPER>
class HpxUpdateGroup
  : public hpx::components::managed_component_base<
        HpxUpdateGroup<CELL_TYPE, PARTITION, STEPPER>
    >
{
public:
    const static int DIM = CELL_TYPE::Topology::DIM;

    typedef
        LibGeoDecomp::HpxSimulator::HpxUpdateGroup<CELL_TYPE, PARTITION, STEPPER> ClientType;

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
    
    typedef boost::shared_ptr<typename HpxPatchLink<GridType, ClientType>::Link> PatchLinkPtr;

    typedef
        typename HiParSimulator::Stepper<CELL_TYPE>::PatchAccepterVec
        PatchAccepterVec;
    typedef
        typename HiParSimulator::Stepper<CELL_TYPE>::PatchProviderVec
        PatchProviderVec;

    typedef
        typename HpxPatchLink<GridType, ClientType>::Provider
        PatchLinkProviderType;
    
    typedef boost::shared_ptr<PatchLinkProviderType> PatchLinkProviderPtr;

    typedef
        typename HpxPatchLink<GridType, ClientType>::Accepter
        PatchLinkAccepterType;

    typedef boost::shared_ptr<PatchLinkAccepterType> PatchLinkAccepterPtr;
    
    typedef
        HiParSimulator::PartitionManager<DIM, typename CELL_TYPE::Topology> 
        PartitionManagerType;
    typedef typename PartitionManagerType::RegionVecMap RegionVecMap;
    
    typedef
        HiParSimulator::ParallelWriterAdapter<GridType, CELL_TYPE, HpxUpdateGroup> 
        ParallelWriterAdapterType;
    typedef HiParSimulator::SteererAdapter<GridType, CELL_TYPE> SteererAdapterType;
    
    typedef std::pair<std::size_t, std::size_t> StepPairType;

    HpxUpdateGroup()
      : boundingBoxFuture(boundingBoxPromise.get_future())
      , initFuture(initPromise.get_future())
    {}
    
    void init(
        std::vector<ClientType> const & updateGroups,
        unsigned ghostZoneWidth,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        WriterVector const & writers,
        SteererVector const & steerers
    )
    {
        this->updateGroups = updateGroups;
        this->initializer = initializer;
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
        BOOST_FOREACH(ClientType const & ug, updateGroups)
        {
            if(ug.gid() == this->get_gid())
            {
                boundingBoxesFutures << boundingBoxFuture;
            }
            else
            {
                boundingBoxesFutures << ug.boundingBox();
            }
        }
        
        SuperVector<CoordBox<DIM> > boundingBoxes;
        boundingBoxes.reserve(numPartitions);

        BOOST_FOREACH(hpx::future<CoordBox<DIM> > & f, boundingBoxesFutures)
        {
            boundingBoxes << f.get();
        }
        partitionManager->resetGhostZones(boundingBoxes);

        long firstSyncPoint =
            initializer->startStep() * CELL_TYPE::nanoSteps() + ghostZoneWidth;

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
                    HpxPatchLink<GridType, ClientType>::ENDLESS,
                    ghostZoneWidth);

                link->setRegion(partitionManager->ownRegion());
            }
        }

        PatchAccepterVec patchAcceptersInner;

        // Convert writers to patch accepters
        BOOST_FOREACH(typename WriterVector::value_type const & writer, writers)
        {
            PatchAccepterPtr adapterGhost(
                new ParallelWriterAdapterType(
                    this,
                    boost::shared_ptr<ParallelWriter<CELL_TYPE> >(writer->clone()),
                    initializer->startStep(),
                    initializer->maxSteps(),
                    initializer->gridDimensions(),
                    false));
            PatchAccepterPtr adapterInnerSet(
                new ParallelWriterAdapterType(
                    this,
                    boost::shared_ptr<ParallelWriter<CELL_TYPE> >(writer->clone()),
                    initializer->startStep(),
                    initializer->maxSteps(),
                    initializer->gridDimensions(),
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
                    HpxPatchLink<GridType, ClientType>::ENDLESS,
                    ghostZoneWidth);

                link->setRegion(partitionManager->ownRegion());
            }
        }

        // Convert steerer to patch accepters
        BOOST_FOREACH(typename SteererVector::value_type const & steerer, steerers)
        {
            // two adapters needed, just as for the writers
            PatchProviderPtr adapterGhost(
                new SteererAdapterType(steerer));
            PatchProviderPtr adapterInnerSet(
                new SteererAdapterType(steerer));

            adapterGhost->setRegion(partitionManager->ownRegion());
            adapterInnerSet->setRegion(partitionManager->ownRegion());

            addPatchProvider(adapterGhost, HiParSimulator::Stepper<CELL_TYPE>::GHOST);
            addPatchProvider(adapterInnerSet, HiParSimulator::Stepper<CELL_TYPE>::INNER_SET);
        }

        initPromise.set_value();
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(HpxUpdateGroup, init, InitAction);

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
        if(stepper)
        {
            return stepper->currentStep();
        }
        else
        {
            return std::make_pair(initializer->startStep(), 0u);
        }
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(HpxUpdateGroup, currentStep, CurrentStepAction);

    std::size_t getStep() const
    {
        return currentStep().first;
    }

    void nanoStep(std::size_t remainingNanoSteps)
    {
        hpx::wait(initFuture);
        stepper->update(remainingNanoSteps);
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(HpxUpdateGroup, nanoStep, NanoStepAction);

    //hpx::future<CoordBox<DIM> > boundingBox()
    CoordBox<DIM> boundingBox()
    {
        return boundingBoxFuture.get();
    }
    HPX_DEFINE_COMPONENT_ACTION_TPL(HpxUpdateGroup, boundingBox, BoundingBoxAction);
    
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
    HPX_DEFINE_COMPONENT_ACTION_TPL(HpxUpdateGroup, setOuterGhostZone, SetOuterGhostZoneAction);

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
    std::size_t rank;

    hpx::lcos::local::promise<CoordBox<DIM> > boundingBoxPromise;
    hpx::future<CoordBox<DIM> > boundingBoxFuture;

    hpx::lcos::local::promise<void> initPromise;
    hpx::future<void> initFuture;

    void setRank()
    {
        rank = 0;
        BOOST_FOREACH(ClientType const & ug, updateGroups)
        {
            if(ug.gid() == this->get_gid())
            {
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
};

}}}

#endif
#endif
