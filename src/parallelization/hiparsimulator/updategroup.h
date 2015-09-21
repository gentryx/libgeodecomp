#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_MPI
#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_UPDATEGROUP_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_UPDATEGROUP_H

#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/communication/patchlink.h>
#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/geometry/partitionmanager.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/gridtypeselector.h>
#include <libgeodecomp/storage/patchaccepter.h>
#include <libgeodecomp/storage/patchprovider.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<class CELL_TYPE>
class UpdateGroup
{
public:
    friend class HiParSimulatorTest;
    friend class UpdateGroupPrototypeTest;
    friend class UpdateGroupTest;

    typedef Stepper<CELL_TYPE> StepperType;
    typedef typename StepperType::Topology Topology;
    typedef typename APITraits::SelectSoA<CELL_TYPE>::Value SupportsSoA;
    typedef typename GridTypeSelector<CELL_TYPE, Topology, true, SupportsSoA>::Value GridType;
    typedef typename StepperType::PatchType PatchType;
    typedef typename StepperType::PatchProviderPtr PatchProviderPtr;
    typedef typename StepperType::PatchAccepterPtr PatchAccepterPtr;
    typedef boost::shared_ptr<typename PatchLink<GridType>::Link> PatchLinkPtr;
    typedef PartitionManager<Topology> PartitionManagerType;
    typedef typename PartitionManagerType::RegionVecMap RegionVecMap;
    typedef typename StepperType::PatchAccepterVec PatchAccepterVec;
    typedef typename StepperType::PatchProviderVec PatchProviderVec;

    const static int DIM = Topology::DIM;

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
        MPI_Comm communicator = MPI_COMM_WORLD) :
        ghostZoneWidth(ghostZoneWidth),
        initializer(initializer),
        mpiLayer(communicator),
        rank(mpiLayer.rank())
    {
        partitionManager.reset(new PartitionManagerType());
        partitionManager->resetRegions(
            box,
            partition,
            rank,
            ghostZoneWidth);
        CoordBox<DIM> ownBoundingBox(partitionManager->ownRegion().boundingBox());

        std::vector<CoordBox<DIM> > boundingBoxes(mpiLayer.size());
        mpiLayer.allGather(ownBoundingBox, &boundingBoxes);
        partitionManager->resetGhostZones(boundingBoxes);

        long firstSyncPoint =
            initializer->startStep() * APITraits::SelectNanoSteps<CELL_TYPE>::VALUE +
            ghostZoneWidth;

        // For conformance with the HPX UpdateGroup, we're creating
        // the PatchProviders early:
        PatchProviderVec patchLinkProviders;
        const RegionVecMap& map1 = partitionManager->getOuterGhostZoneFragments();
        for (typename RegionVecMap::const_iterator i = map1.begin(); i != map1.end(); ++i) {
            if (!i->second.back().empty()) {
                boost::shared_ptr<typename PatchLink<GridType>::Provider> link(
                    new typename PatchLink<GridType>::Provider(
                        i->second.back(),
                        i->first,
                        MPILayer::PATCH_LINK,
                        SerializationBuffer<CELL_TYPE>::cellMPIDataType(),
                        mpiLayer.communicator()));
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
                boost::shared_ptr<typename PatchLink<GridType>::Accepter> link(
                    new typename PatchLink<GridType>::Accepter(
                        i->second.back(),
                        i->first,
                        MPILayer::PATCH_LINK,
                        SerializationBuffer<CELL_TYPE>::cellMPIDataType(),
                        mpiLayer.communicator()));
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

        // notify all PatchProviders of the process' region:
        for (std::size_t i = 0; i < patchProvidersGhost.size(); ++i) {
            patchProvidersGhost[i]->setRegion(partitionManager->ownRegion());
        }
        for (std::size_t i = 0; i < patchProvidersInner.size(); ++i) {
            patchProvidersInner[i]->setRegion(partitionManager->ownRegion());
        }

        stepper.reset(new STEPPER(
                          partitionManager,
                          this->initializer,
                          patchAcceptersGhost + ghostZoneAccepterLinks,
                          patchAcceptersInner,
                          // add external PatchProviders last to allow them to override
                          // the local ghost zone providers (a.k.a. PatchLink::Source).
                          patchLinkProviders + patchProvidersGhost,
                          patchProvidersInner));
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
    MPILayer mpiLayer;
    unsigned rank;
};

}
}

#endif
#endif
