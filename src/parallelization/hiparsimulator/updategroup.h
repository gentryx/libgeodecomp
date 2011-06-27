#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_updategroup_h_
#define _libgeodecomp_parallelization_hiparsimulator_updategroup_h_

#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/hiparsimulator/intersectingregionaccumulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillaregionaccumulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/parallelstepper.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitionmanager.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

// updategroup in its current form is deprecated and because of
// oldsteppers failings out of order
template<class CELL_TYPE, class PARTITION>
class UpdateGroup
{
    friend class UpdateGroupPrototypeTest;
    friend class UpdateGroupTest;
public:
    UpdateGroup(
        const PARTITION& _partition, 
        const SuperVector<unsigned>& _weights, 
        const unsigned& _offset,
        const CoordBox<2>& rect, 
        const unsigned& _ghostZoneWidth,
        Initializer<CELL_TYPE> *initializer,
        PatchProvider<DisplacedGrid<CELL_TYPE> > *ghostZonePatchProvider = 0,
        PatchAccepter<DisplacedGrid<CELL_TYPE> > *ghostZonePatchAccepter = 0,
        MPI::Comm *communicator = &MPI::COMM_WORLD) : 
        stepper(communicator, ghostZonePatchProvider, ghostZonePatchAccepter),
        partition(_partition),
        weights(_weights),
        offset(_offset),
        // fixme: move ghostZoneWidth to 
        ghostZoneWidth(_ghostZoneWidth),
        nanoStepCounter(initializer->startStep() * CELL_TYPE::nanoSteps() + initializer->startNanoStep()),
        mpiLayer(communicator),
        rank(mpiLayer.rank())
    {
        partitionManager.resetRegions(
            rect,
            new VanillaRegionAccumulator<PARTITION, 2>(
                partition,
                offset,
                weights),
            rank,
            ghostZoneWidth);
        SuperVector<CoordBox<2> > boundingBoxes(mpiLayer.size());
        CoordBox<2> ownBoundingBox(partitionManager.ownRegion().boundingBox());
        mpiLayer.allGather(ownBoundingBox, &boundingBoxes);
        partitionManager.resetGhostZones(boundingBoxes);

        stepper.resetRegions(
            &partitionManager,
            initializer);
    }

    UpdateGroup(
        const Region<2>& baseRegion,
        const PARTITION& _partition, 
        const SuperVector<unsigned>& _weights, 
        const unsigned& _offset,
        const CoordBox<2>& rect, 
        const unsigned& _ghostZoneWidth,
        Initializer<CELL_TYPE> *initializer,
        PatchProvider<DisplacedGrid<CELL_TYPE> > *ghostZonePatchProvider = 0,
        PatchAccepter<DisplacedGrid<CELL_TYPE> > *ghostZonePatchAccepter = 0,
        MPI::Comm *communicator = &MPI::COMM_WORLD) : 
        stepper(communicator, ghostZonePatchProvider, ghostZonePatchAccepter),
        partition(_partition),
        weights(_weights),
        offset(_offset),
        ghostZoneWidth(_ghostZoneWidth),
        nanoStepCounter(initializer->startStep() * CELL_TYPE::nanoSteps() + initializer->startNanoStep()),
        mpiLayer(communicator),
        rank(mpiLayer.rank())
    {
         partitionManager.resetRegions(
            rect,
            new IntersectingRegionAccumulator<PARTITION, 2>(
                baseRegion,
                partition,
                offset,
                weights),
            rank,
            ghostZoneWidth);
        SuperVector<CoordBox<2> > boundingBoxes(mpiLayer.size());
        CoordBox<2> ownBoundingBox(partitionManager.ownRegion().boundingBox());
        mpiLayer.allGather(ownBoundingBox, &boundingBoxes);
        partitionManager.resetGhostZones(boundingBoxes);

        stepper.resetRegions(
            &partitionManager,
            initializer);
    }

    inline void nanoStep(
        const unsigned& curHopLenght, 
        const unsigned& nextHopLength)
    {
        unsigned curStop = nanoStepCounter + curHopLenght;
        unsigned nextStop = curStop + nextHopLength;
        stepper.nanoStep(curStop, nextStop, nanoStepCounter);
        nanoStepCounter = curStop;
    }

    inline Region<2>& getOuterOutgroupGhostZone()
    {
        return partitionManager.getOuterOutgroupGhostZoneFragment();
    }

    inline unsigned getNanoStep()
    {
        return nanoStepCounter;
    }

    // fixme: make return type const
    DisplacedGrid<CELL_TYPE> *getGrid() const
    {
        return stepper.getGrid();
    }

    // fixme: remove
    DisplacedGrid<CELL_TYPE> *getNewGrid() const
    {
        return stepper.getNewGrid();
    }


private:
    ParallelStepper<CELL_TYPE> stepper;
    PartitionManager<2> partitionManager;
    PARTITION partition;
    SuperVector<unsigned> weights;
    unsigned offset;
    unsigned ghostZoneWidth;
    unsigned nanoStepCounter;
    MPILayer mpiLayer;
    unsigned rank;
};

}
}

#endif
#endif
