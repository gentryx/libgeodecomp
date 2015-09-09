#ifndef LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_UPDATEGROUP_H
#define LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_UPDATEGROUP_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX

#include <libgeodecomp/communication/hpxpatchlink.h>
#include <libgeodecomp/geometry/partitionmanager.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/parallelization/hpxsimulator/updategroupserver.h>
#include <libgeodecomp/storage/displacedgrid.h>

#include <hpx/apply.hpp>

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
        PatchProviderVec patchProvidersInner = PatchProviderVec()) :
        ghostZoneWidth(ghostZoneWidth),
        initializer(initializer),
        rank(hpx::get_locality_id())
    {
        partitionManager.reset(new PartitionManagerType());
        partitionManager->resetRegions(
            box,
            partition,
            rank,
            ghostZoneWidth);
        CoordBox<DIM> ownBoundingBox(partitionManager->ownRegion().boundingBox());

        // fixme
        // std::vector<CoordBox<DIM> > boundingBoxes(mpiLayer.size());
        // mpiLayer.allGather(ownBoundingBox, &boundingBoxes);
        // partitionManager->resetGhostZones(boundingBoxes);

        long firstSyncPoint =
            initializer->startStep() * APITraits::SelectNanoSteps<CELL_TYPE>::VALUE +
            ghostZoneWidth;

        // we have to hand over a list of all ghostzone senders as the
        // stepper will perform an initial update of the ghostzones
        // upon creation and we have to send those over to our neighbors.
        PatchAccepterVec ghostZoneAccepterLinks;
        // RegionVecMap map = partitionManager->getInnerGhostZoneFragments();
        // for (typename RegionVecMap::iterator i = map.begin(); i != map.end(); ++i) {
        //     if (!i->second.back().empty()) {
        //         boost::shared_ptr<typename PatchLink<GridType>::Accepter> link(
        //             new typename PatchLink<GridType>::Accepter(
        //                 i->second.back(),
        //                 i->first,
        //                 MPILayer::PATCH_LINK,
        //                 SerializationBuffer<CELL_TYPE>::cellMPIDataType(),
        //                 mpiLayer.communicator()));
        //         ghostZoneAccepterLinks << link;
        //         patchLinks << link;

        //         link->charge(
        //             firstSyncPoint,
        //             PatchAccepter<GridType>::infinity(),
        //             ghostZoneWidth);

        //         link->setRegion(partitionManager->ownRegion());
        //     }
        // }
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

        // UpdateGroup(hpx::id_type thisId = hpx::id_type()) :
    //     thisId(thisId)
    // {}

    // struct InitData
    // {
    //     unsigned loadBalancingPeriod;
    //     unsigned ghostZoneWidth;
    //     boost::shared_ptr<Initializer<CELL_TYPE> > initializer;
    //     WriterVector writers;
    //     SteererVector steerers;
    //     std::vector<CoordBox<DIM> > boundingBoxes;
    //     std::vector<std::size_t> initialWeights;

    //     template <typename ARCHIVE>
    //     void serialize(ARCHIVE& ar, unsigned)
    //     {
    //         ar & loadBalancingPeriod;
    //         ar & ghostZoneWidth;
    //         ar & initializer;
    //         ar & writers;
    //         ar & steerers;
    //         ar & boundingBoxes;
    //         ar & initialWeights;
    //     }
    // };

    // hpx::naming::id_type gid() const
    // {
    //     return thisId;
    // }

    // hpx::future<void> setOuterGhostZone(
    //     std::size_t srcRank,
    //     boost::shared_ptr<std::vector<CELL_TYPE> > buffer,
    //     long nanoStep)
    // {
    //     return
    //         hpx::async<typename ComponentType::SetOuterGhostZoneAction>(
    //             thisId,
    //             srcRank,
    //             buffer,
    //             nanoStep
    //         );
    // }


private:
    boost::shared_ptr<StepperType> stepper;
    boost::shared_ptr<PartitionManagerType> partitionManager;
    std::vector<PatchLinkPtr> patchLinks;
    unsigned ghostZoneWidth;
    boost::shared_ptr<Initializer<CELL_TYPE> > initializer;
    unsigned rank;
};

}

}

#endif
#endif
