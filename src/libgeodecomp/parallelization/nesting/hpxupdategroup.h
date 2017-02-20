#ifndef LIBGEODECOMP_PARALLELIZATION_NESTING_HPXUPDATEGROUP_H
#define LIBGEODECOMP_PARALLELIZATION_NESTING_HPXUPDATEGROUP_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/communication/hpxpatchlink.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/parallelization/nesting/updategroup.h>

namespace LibGeoDecomp {

/**
 * This implementation of UpdateGroup is used by the HPXSimulator.
 * This means that the ghost zone (halo) communication is handled by
 * HPX-based PatchLinks.
 */
template <class CELL_TYPE>
class HPXUpdateGroup : public UpdateGroup<CELL_TYPE, HPXPatchLink>
{
public:
    friend class HiParSimulatorTest;
    friend class UpdateGroupPrototypeTest;
    friend class UpdateGroupTest;

    typedef typename UpdateGroup<CELL_TYPE, HPXPatchLink>::GridType GridType;
    typedef typename UpdateGroup<CELL_TYPE, HPXPatchLink>::PatchAccepterVec PatchAccepterVec;
    typedef typename UpdateGroup<CELL_TYPE, HPXPatchLink>::PatchProviderVec PatchProviderVec;
    typedef typename UpdateGroup<CELL_TYPE, HPXPatchLink>::PatchLinkAccepter PatchLinkAccepter;
    typedef typename UpdateGroup<CELL_TYPE, HPXPatchLink>::PatchLinkProvider PatchLinkProvider;
    typedef typename UpdateGroup<CELL_TYPE, HPXPatchLink>::InitPtr InitPtr;
    typedef typename UpdateGroup<CELL_TYPE, HPXPatchLink>::SteererPtr SteererPtr;
    typedef typename UpdateGroup<CELL_TYPE, HPXPatchLink>::PartitionPtr PartitionPtr;
    typedef typename UpdateGroup<CELL_TYPE, HPXPatchLink>::PatchLinkAccepterPtr PatchLinkAccepterPtr;
    typedef typename UpdateGroup<CELL_TYPE, HPXPatchLink>::PatchLinkProviderPtr PatchLinkProviderPtr;

    using UpdateGroup<CELL_TYPE, HPXPatchLink>::init;
    using UpdateGroup<CELL_TYPE, HPXPatchLink>::rank;

    const static int DIM = UpdateGroup<CELL_TYPE, HPXPatchLink>::DIM;

    template<typename STEPPER>
    HPXUpdateGroup(
        PartitionPtr partition,
        const CoordBox<DIM>& box,
        unsigned ghostZoneWidth,
        InitPtr initializer,
        STEPPER *stepperType,
        PatchAccepterVec patchAcceptersGhost = PatchAccepterVec(),
        PatchAccepterVec patchAcceptersInner = PatchAccepterVec(),
        PatchProviderVec patchProvidersGhost = PatchProviderVec(),
        PatchProviderVec patchProvidersInner = PatchProviderVec(),
        bool enableFineGrainedParallelism = false,
        std::string basename = "/HPXSimulator::UpdateGroup",
        int rank = hpx::get_locality_id()):
        UpdateGroup<CELL_TYPE, HPXPatchLink>(ghostZoneWidth, initializer, rank),
        basename(basename)
    {
        init(
            partition,
            box,
            ghostZoneWidth,
            initializer,
            stepperType,
            patchAcceptersGhost,
            patchAcceptersInner,
            patchProvidersGhost,
            patchProvidersInner,
            enableFineGrainedParallelism);
    }

private:
    std::string basename;

    std::vector<CoordBox<DIM> > gatherBoundingBoxes(
        const CoordBox<DIM>& ownBoundingBox,
        std::size_t size,
        std::size_t tag) const
    {
        std::string broadcastName = basename + "/boundingBoxes" + StringOps::itoa(tag);
        std::vector<CoordBox<DIM> > boundingBoxes;
        return HPXReceiver<CoordBox<DIM> >::allGather(ownBoundingBox, rank, size, broadcastName);
    }

    virtual PatchLinkAccepterPtr makePatchLinkAccepter(int target, const Region<DIM>& region)
    {
        return PatchLinkAccepterPtr(
            new typename HPXPatchLink<GridType>::Accepter(
                region,
                basename,
                rank,
                target));

    }

    virtual PatchLinkProviderPtr makePatchLinkProvider(int source, const Region<DIM>& region)
    {
        return PatchLinkProviderPtr(
            new typename HPXPatchLink<GridType>::Provider(
                region,
                basename,
                source,
                rank));
    }

};

}

#endif
#endif
