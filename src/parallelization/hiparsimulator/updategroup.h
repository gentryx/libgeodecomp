#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_updategroup_h_
#define _libgeodecomp_parallelization_hiparsimulator_updategroup_h_

#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/hiparsimulator/intersectingregionaccumulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitionmanager.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchaccepter.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchprovider.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillaregionaccumulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<class CELL_TYPE, class PARTITION, class STEPPER=VanillaStepper<CELL_TYPE> >
class UpdateGroup
{
    friend class UpdateGroupPrototypeTest;
    friend class UpdateGroupTest;
public:
    const static int DIM = CELL_TYPE::Topology::DIMENSIONS;

    UpdateGroup(
        const PARTITION& _partition, 
        const SuperVector<unsigned>& _weights, 
        const unsigned& _offset,
        const CoordBox<DIM>& box, 
        const unsigned& _ghostZoneWidth,
        boost::shared_ptr<Initializer<CELL_TYPE> > _initializer,
        MPI::Comm *communicator = &MPI::COMM_WORLD) : 
        partition(_partition),
        weights(_weights),
        offset(_offset),
        ghostZoneWidth(_ghostZoneWidth),
        initializer(_initializer),
        mpiLayer(communicator),
        rank(mpiLayer.rank())
    {
        partitionManager.reset(new PartitionManager<DIM, typename CELL_TYPE::Topology>());
        partitionManager->resetRegions(
            box,
            new VanillaRegionAccumulator<PARTITION>(
                partition,
                offset,
                weights),
            rank,
            ghostZoneWidth);
        SuperVector<CoordBox<DIM> > boundingBoxes(mpiLayer.size());
        CoordBox<DIM> ownBoundingBox(partitionManager->ownRegion().boundingBox());
        mpiLayer.allGather(ownBoundingBox, &boundingBoxes);
        partitionManager->resetGhostZones(boundingBoxes);

        stepper.reset(
            new STEPPER(
                partitionManager,
                initializer));
    }
    
    inline void update(int nanoSteps) 
    {
        // fixme
    }

private:
    // fixme: kill this dead code
    // UpdateGroup(
    //     const Region<2>& baseRegion,
    //     const PARTITION& _partition, 
    //     const SuperVector<unsigned>& _weights, 
    //     const unsigned& _offset,
    //     const CoordBox<2>& box, 
    //     const unsigned& _ghostZoneWidth,
    //     Initializer<CELL_TYPE> *initializer,
    //     PatchProvider<DisplacedGrid<CELL_TYPE> > *ghostZonePatchProvider = 0,
    //     PatchAccepter<DisplacedGrid<CELL_TYPE> > *ghostZonePatchAccepter = 0,
    //     MPI::Comm *communicator = &MPI::COMM_WORLD) : 
    //     // stepper(communicator, ghostZonePatchProvider, ghostZonePatchAccepter),
    //     partition(_partition),
    //     weights(_weights),
    //     offset(_offset),
    //     ghostZoneWidth(_ghostZoneWidth),
    //     nanoStepCounter(initializer->startStep() * CELL_TYPE::nanoSteps() + initializer->startNanoStep()),
    //     mpiLayer(communicator),
    //     rank(mpiLayer.rank())
    // {
    //      partitionManager.resetRegions(
    //         box,
    //         new IntersectingRegionAccumulator<PARTITION, 2>(
    //             baseRegion,
    //             partition,
    //             offset,
    //             weights),
    //         rank,
    //         ghostZoneWidth);
    //     SuperVector<CoordBox<2> > boundingBoxes(mpiLayer.size());
    //     CoordBox<2> ownBoundingBox(partitionManager.ownRegion().boundingBox());
    //     mpiLayer.allGather(ownBoundingBox, &boundingBoxes);
    //     partitionManager.resetGhostZones(boundingBoxes);

    //     // stepper.resetRegions(
    //     //     &partitionManager,
    //     //     initializer);
    // }

    // inline void nanoStep(
    //     const unsigned& curHopLenght, 
    //     const unsigned& nextHopLength)
    // {
    //     unsigned curStop = nanoStepCounter + curHopLenght;
    //     unsigned nextStop = curStop + nextHopLength;
    //     // stepper.nanoStep(curStop, nextStop, nanoStepCounter);
    //     nanoStepCounter = curStop;
    // }

    // inline Region<2>& getOuterOutgroupGhostZone()
    // {
    //     return partitionManager.getOuterOutgroupGhostZoneFragment();
    // }

    // inline unsigned getNanoStep()
    // {
    //     return nanoStepCounter;
    // }

    // fixme: make return type const
    DisplacedGrid<CELL_TYPE> *getGrid() const
    {
        return 0;
    }

    // fixme: remove
    DisplacedGrid<CELL_TYPE> *getNewGrid() const
    {
        return 0;
    }


private:
    boost::shared_ptr<Stepper<CELL_TYPE> > stepper;
    boost::shared_ptr<PartitionManager<DIM, typename CELL_TYPE::Topology> > partitionManager;
    PARTITION partition;
    SuperVector<unsigned> weights;
    unsigned offset;
    unsigned ghostZoneWidth;
    boost::shared_ptr<Initializer<CELL_TYPE> > initializer;
    MPILayer mpiLayer;
    unsigned rank;
};

}
}

#endif
#endif
