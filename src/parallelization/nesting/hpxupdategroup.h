#ifndef LIBGEODECOMP_PARALLELIZATION_NESTING_HPXUPDATEGROUP_H
#define LIBGEODECOMP_PARALLELIZATION_NESTING_HPXUPDATEGROUP_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/communication/hpxpatchlink.h>
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

    using UpdateGroup<CELL_TYPE, HPXPatchLink>::init;
    using UpdateGroup<CELL_TYPE, HPXPatchLink>::rank;
    const static int DIM = UpdateGroup<CELL_TYPE, HPXPatchLink>::DIM;

    template<typename STEPPER>
    HPXUpdateGroup(
        boost::shared_ptr<Partition<DIM> > partition,
        const CoordBox<DIM>& box,
        unsigned ghostZoneWidth,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
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
        boost::shared_ptr<Partition<DIM> > partition) const
    {
        std::size_t size = partition->getWeights().size();
        std::string broadcastName = basename + "/boundingBoxes";
        std::vector<CoordBox<DIM> > boundingBoxes;
        return HPXReceiver<CoordBox<DIM> >::allGather(ownBoundingBox, rank, size, broadcastName);
    }

    virtual boost::shared_ptr<PatchLinkAccepter> makePatchLinkAccepter(int target, const Region<DIM>& region)
    {
        return boost::shared_ptr<PatchLinkAccepter>(
            new typename HPXPatchLink<GridType>::Accepter(
                region,
                basename,
                rank,
                target));

    }

    virtual boost::shared_ptr<PatchLinkProvider> makePatchLinkProvider(int source, const Region<DIM>& region)
    {
        return boost::shared_ptr<PatchLinkProvider>(
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
