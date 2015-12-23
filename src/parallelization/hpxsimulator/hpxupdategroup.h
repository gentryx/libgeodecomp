#ifndef LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_HPXUPDATEGROUP_H
#define LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_HPXUPDATEGROUP_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/communication/hpxpatchlink.h>
#include <libgeodecomp/parallelization/updategroup.h>

namespace LibGeoDecomp {

// fixme: can't we reuse the MPI UpdateGroup here?
template <class CELL_TYPE>
class HPXUpdateGroup : public UpdateGroup<CELL_TYPE, HPXPatchLink>
{
public:
    friend class HiParSimulatorTest;
    friend class UpdateGroupPrototypeTest;
    friend class UpdateGroupTest;

    using typename UpdateGroup<CELL_TYPE, HPXPatchLink>::GridType;
    using typename UpdateGroup<CELL_TYPE, HPXPatchLink>::PatchAccepterVec;
    using typename UpdateGroup<CELL_TYPE, HPXPatchLink>::PatchProviderVec;
    using typename UpdateGroup<CELL_TYPE, HPXPatchLink>::PatchLinkPtr;
    using typename UpdateGroup<CELL_TYPE, HPXPatchLink>::RegionVecMap;
    using typename UpdateGroup<CELL_TYPE, HPXPatchLink>::StepperType;
    using typename UpdateGroup<CELL_TYPE, HPXPatchLink>::Topology;

    using UpdateGroup<CELL_TYPE, HPXPatchLink>::partitionManager;
    using UpdateGroup<CELL_TYPE, HPXPatchLink>::patchLinks;
    using UpdateGroup<CELL_TYPE, HPXPatchLink>::rank;
    using UpdateGroup<CELL_TYPE, HPXPatchLink>::stepper;
    using UpdateGroup<CELL_TYPE, HPXPatchLink>::DIM;

    template<typename STEPPER>
    HPXUpdateGroup(
        boost::shared_ptr<Partition<DIM> > partition,
        const CoordBox<DIM>& box,
        const unsigned& ghostZoneWidth,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        STEPPER *stepperType,
        PatchAccepterVec patchAcceptersGhost = PatchAccepterVec(),
        PatchAccepterVec patchAcceptersInner = PatchAccepterVec(),
        PatchProviderVec patchProvidersGhost = PatchProviderVec(),
        PatchProviderVec patchProvidersInner = PatchProviderVec(),
        std::string basename = "/HPXSimulator::UpdateGroup",
        int rank = hpx::get_locality_id()) :
        UpdateGroup<CELL_TYPE, HPXPatchLink>(ghostZoneWidth, initializer, rank),
        basename(basename)
    {
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

        // We need to create the patch providers first, as the HPX patch
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

private:
    std::string basename;
};

}

#endif
#endif
