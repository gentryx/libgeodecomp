#ifndef LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_UPDATEGROUP_H
#define LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_UPDATEGROUP_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/communication/hpxpatchlink.h>
#include <libgeodecomp/geometry/partitionmanager.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/parallelization/hpxsimulator/updategroupserver.h>
#include <libgeodecomp/storage/displacedgrid.h>

namespace LibGeoDecomp {


// fixme: move to LibGeoDecomp namespace
namespace HpxSimulator {

// fixme: can't we reuse the MPI UpdateGroup here? 
template <class CELL_TYPE>
class UpdateGroup
{
public:
    typedef typename LibGeoDecomp::HiParSimulator::Stepper<CELL_TYPE> StepperType;
    typedef typename StepperType::Topology Topology;
    typedef typename APITraits::SelectSoA<CELL_TYPE>::Value SupportsSoA;
    typedef typename GridTypeSelector<CELL_TYPE, Topology, true, SupportsSoA>::Value GridType;
    typedef typename StepperType::PatchType PatchType;
    typedef typename StepperType::PatchProviderPtr PatchProviderPtr;
    typedef typename StepperType::PatchAccepterPtr PatchAccepterPtr;
    typedef boost::shared_ptr<typename LibGeoDecomp::HPXPatchLink<GridType>::Link> PatchLinkPtr;
    typedef PartitionManager<Topology> PartitionManagerType;
    typedef typename PartitionManagerType::RegionVecMap RegionVecMap;
    typedef typename StepperType::PatchAccepterVec PatchAccepterVec;
    typedef typename StepperType::PatchProviderVec PatchProviderVec;

    const static int DIM = Topology::DIM;

    // friend class hpx::serialization::access;
    // friend class UpdateGroupServer<CELL_TYPE, PARTITION, STEPPER>;

    // typedef typename STEPPER::Topology Topology;
    // typedef DisplacedGrid<CELL_TYPE, Topology, true> GridType;
    // typedef typename STEPPER::PatchType PatchType;
    // typedef typename STEPPER::PatchProviderPtr PatchProviderPtr;
    // typedef typename STEPPER::PatchAccepterPtr PatchAccepterPtr;
    // typedef typename STEPPER::PatchAccepterVec PatchAccepterVec;
    // typedef typename STEPPER::PatchProviderVec PatchProviderVec;
    // const static int DIM = Topology::DIM;

    // typedef
    //     typename DistributedSimulator<CELL_TYPE>::WriterVector
    //     WriterVector;
    // typedef
    //     typename DistributedSimulator<CELL_TYPE>::SteererVector
    //     SteererVector;

    // typedef UpdateGroupServer<CELL_TYPE, PARTITION, STEPPER> ComponentType;

    // typedef std::pair<std::size_t, std::size_t> StepPairType;

    template<typename STEPPER>
    UpdateGroup(
        boost::shared_ptr<Partition<DIM> > partition,
        const CoordBox<DIM>& box,
        const unsigned& ghostZoneWidth,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        STEPPER *stepperType,
        PatchAccepterVec patchAcceptersGhost = PatchAccepterVec(),
        PatchAccepterVec patchAcceptersInner = PatchAccepterVec(),
        PatchProviderVec patchProvidersGhost = PatchProviderVec(),
        PatchProviderVec patchProvidersInner = PatchProviderVec(),
        std::string basename = "HPXSimulator::UpdateGroup",
        int rank = hpx::get_locality_id()) :
        ghostZoneWidth(ghostZoneWidth),
        initializer(initializer),
        basename(basename),
        rank(rank)
    {
        partitionManager.reset(new PartitionManagerType());
        partitionManager->resetRegions(
            box,
            partition,
            rank,
            ghostZoneWidth);
        CoordBox<DIM> ownBoundingBox(partitionManager->ownRegion().boundingBox());

        // fixme
        std::size_t size = partition->getWeights().size();
        std::string broadcastName = basename + "/boundingBoxes";
        std::vector<CoordBox<DIM> > boundingBoxes;

        boundingBoxes = HPXReceiver<CoordBox<DIM> >::allGather(ownBoundingBox, rank, size, broadcastName);
        partitionManager->resetGhostZones(boundingBoxes);

        long firstSyncPoint =
            initializer->startStep() * APITraits::SelectNanoSteps<CELL_TYPE>::VALUE +
            ghostZoneWidth;

        // fixme
        // We need to create the patch providers first, as the patch
        // accepters will look up their IDs upon creation:
        PatchProviderVec patchLinkProviders;
        const RegionVecMap& map1 = partitionManager->getOuterGhostZoneFragments();
        for (typename RegionVecMap::const_iterator i = map1.begin(); i != map1.end(); ++i) {
            if (!i->second.back().empty()) {
                boost::shared_ptr<typename HPXPatchLink<GridType>::Provider> link(
                    new typename HPXPatchLink<GridType>::Provider(
                        i->second.back(),
                        basename,
                        i->first,
                        rank));
                patchLinkProviders << link;
                patchLinks << link;

                link->charge(
                    firstSyncPoint,
                    PatchProvider<GridType>::infinity(),
                    ghostZoneWidth);

                link->setRegion(partitionManager->ownRegion());
            }
        }

        // we have to hand over a list of all ghostzone senders as the
        // stepper will perform an initial update of the ghostzones
        // upon creation and we have to send those over to our neighbors.
        PatchAccepterVec ghostZoneAccepterLinks;
        const RegionVecMap& map2 = partitionManager->getInnerGhostZoneFragments();
        for (typename RegionVecMap::const_iterator i = map2.begin(); i != map2.end(); ++i) {
            if (!i->second.back().empty()) {
                // fixme
                boost::shared_ptr<typename HPXPatchLink<GridType>::Accepter> link(
                    new typename HPXPatchLink<GridType>::Accepter(
                        i->second.back(),
                        basename,
                        rank,
                        i->first));
                ghostZoneAccepterLinks << link;
                patchLinks << link;

                link->charge(
                    firstSyncPoint,
                    PatchAccepter<GridType>::infinity(),
                    ghostZoneWidth);

                link->setRegion(partitionManager->ownRegion());
            }
        }

        // notify all PatchAccepters of the process' region:
        for (std::size_t i = 0; i < patchAcceptersGhost.size(); ++i) {
            patchAcceptersGhost[i]->setRegion(partitionManager->ownRegion());
        }
        for (std::size_t i = 0; i < patchAcceptersInner.size(); ++i) {
            patchAcceptersInner[i]->setRegion(partitionManager->ownRegion());
        }

        stepper.reset(new STEPPER(
                          partitionManager,
                          this->initializer,
                          patchAcceptersGhost + ghostZoneAccepterLinks,
                          patchAcceptersInner));

        // the ghostzone receivers may be safely added after
        // initialization as they're only really needed when the next
        // ghostzone generation is being received.
        for (typename PatchProviderVec::iterator i = patchLinkProviders.begin(); i != patchLinkProviders.end(); ++i) {
            addPatchProvider(*i, StepperType::GHOST);
        }

        // add external PatchProviders last to allow them to override
        // the local ghost zone providers (a.k.a. PatchLink::Source).
        for (typename PatchProviderVec::iterator i = patchProvidersGhost.begin();
             i != patchProvidersGhost.end();
             ++i) {
            (*i)->setRegion(partitionManager->ownRegion());
            addPatchProvider(*i, StepperType::GHOST);
        }

        for (typename PatchProviderVec::iterator i = patchProvidersInner.begin();
             i != patchProvidersInner.end();
             ++i) {
            (*i)->setRegion(partitionManager->ownRegion());
            addPatchProvider(*i, StepperType::INNER_SET);
        }
    }

    virtual ~UpdateGroup()
    {
        for (typename std::vector<PatchLinkPtr>::iterator i = patchLinks.begin();
             i != patchLinks.end();
             ++i) {
            (*i)->cleanup();
        }
    }

    const Chronometer& statistics() const
    {
        return stepper->statistics();
    }

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

    inline void update(int nanoSteps)
    {
        stepper->update(nanoSteps);
    }

    const GridType& grid() const
    {
        return stepper->grid();
    }

    inline virtual std::pair<int, int> currentStep() const
    {
        return stepper->currentStep();
    }

    inline const std::vector<std::size_t>& getWeights() const
    {
        return partitionManager->getWeights();
    }

    inline double computeTimeInner() const
    {
        return stepper->computeTimeInner;
    }

    inline double computeTimeGhost() const
    {
        return stepper->computeTimeGhost;
    }

    inline double patchAcceptersTime() const
    {
        return stepper->patchAcceptersTime;
    }

    inline double patchProvidersTime() const
    {
        return stepper->patchAcceptersTime;
    }

private:
    boost::shared_ptr<StepperType> stepper;
    boost::shared_ptr<PartitionManagerType> partitionManager;
    std::vector<PatchLinkPtr> patchLinks;
    unsigned ghostZoneWidth;
    boost::shared_ptr<Initializer<CELL_TYPE> > initializer;
    std::string basename;
    unsigned rank;
};

}

}

#endif
#endif
