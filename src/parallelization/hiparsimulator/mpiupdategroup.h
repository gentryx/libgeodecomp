#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_MPIUPDATEGROUP_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_MPIUPDATEGROUP_H

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
    using typename UpdateGroup<CELL_TYPE, PatchLink>::PatchLinkAccepter;
    using typename UpdateGroup<CELL_TYPE, PatchLink>::PatchLinkProvider;

    using UpdateGroup<CELL_TYPE, PatchLink>::init;
    using UpdateGroup<CELL_TYPE, PatchLink>::rank;
    using UpdateGroup<CELL_TYPE, PatchLink>::DIM;

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
        init(
            partition,
            box,
            ghostZoneWidth,
            initializer,
            stepperType,
            patchAcceptersGhost,
            patchAcceptersInner,
            patchProvidersGhost,
            patchProvidersInner);
    }

private:
    MPILayer mpiLayer;

    std::vector<CoordBox<DIM> > gatherBoundingBoxes(
        const CoordBox<DIM>& ownBoundingBox,
        boost::shared_ptr<Partition<DIM> > partition) const
    {
        std::vector<CoordBox<DIM> > boundingBoxes(mpiLayer.size());
        mpiLayer.allGather(ownBoundingBox, &boundingBoxes);
        return boundingBoxes;
    }

    virtual boost::shared_ptr<PatchLinkAccepter> makePatchLinkAccepter(int target, const Region<DIM>& region)
    {
        return boost::shared_ptr<typename PatchLink<GridType>::Accepter>(
            new typename PatchLink<GridType>::Accepter(
                region,
                target,
                MPILayer::PATCH_LINK,
                SerializationBuffer<CELL_TYPE>::cellMPIDataType(),
                mpiLayer.communicator()));

    }

    virtual boost::shared_ptr<PatchLinkProvider> makePatchLinkProvider(int source, const Region<DIM>& region)
    {
        return boost::shared_ptr<typename PatchLink<GridType>::Provider>(
            new typename PatchLink<GridType>::Provider(
                region,
                source,
                MPILayer::PATCH_LINK,
                SerializationBuffer<CELL_TYPE>::cellMPIDataType(),
                mpiLayer.communicator()));
    }
};

}

#endif
#endif
