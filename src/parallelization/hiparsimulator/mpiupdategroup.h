#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_UPDATEGROUP_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_UPDATEGROUP_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_MPI

#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/communication/patchlink.h>
#include <libgeodecomp/parallelization/updategroup.h>

namespace LibGeoDecomp {

namespace HiParSimulator {
class HiParSimulatorTest;
}

template<class CELL_TYPE>
class MPIUpdateGroup : public UpdateGroup<CELL_TYPE, PatchLink>
{
public:
    friend class LibGeoDecomp::HiParSimulator::HiParSimulatorTest;
    friend class UpdateGroupPrototypeTest;
    friend class UpdateGroupTest;

    using typename UpdateGroup<CELL_TYPE, PatchLink>::GridType;
    using typename UpdateGroup<CELL_TYPE, PatchLink>::PatchAccepterVec;
    using typename UpdateGroup<CELL_TYPE, PatchLink>::PatchProviderVec;
    using typename UpdateGroup<CELL_TYPE, PatchLink>::PatchLinkPtr;
    using typename UpdateGroup<CELL_TYPE, PatchLink>::RegionVecMap;
    using typename UpdateGroup<CELL_TYPE, PatchLink>::StepperType;
    using typename UpdateGroup<CELL_TYPE, PatchLink>::Topology;

    using UpdateGroup<CELL_TYPE, PatchLink>::partitionManager;
    using UpdateGroup<CELL_TYPE, PatchLink>::patchLinks;
    using UpdateGroup<CELL_TYPE, PatchLink>::rank;
    using UpdateGroup<CELL_TYPE, PatchLink>::stepper;
    using UpdateGroup<CELL_TYPE, PatchLink>::DIM;

    typedef typename StepperType::PatchType PatchType;
    typedef typename StepperType::PatchProviderPtr PatchProviderPtr;
    typedef typename StepperType::PatchAccepterPtr PatchAccepterPtr;

    template<typename STEPPER>
    MPIUpdateGroup(
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
        UpdateGroup<CELL_TYPE, PatchLink>(ghostZoneWidth, initializer, MPILayer(communicator).rank()),
        mpiLayer(communicator)
    {
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

        // We need to create the patch providers first, as the HPX patch
        // accepters will look up their IDs upon creation:
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

    virtual ~MPIUpdateGroup()
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
    MPILayer mpiLayer;
};

}

#endif
#endif
