#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_parallelstepper_h_
#define _libgeodecomp_parallelization_hiparsimulator_parallelstepper_h_

#include <boost/shared_ptr.hpp>
#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchaccepter.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchprovider.h>
#include <libgeodecomp/parallelization/hiparsimulator/innersetmarker.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitionmanager.h>
#include <libgeodecomp/parallelization/hiparsimulator/rimmarker.h>
#include <libgeodecomp/parallelization/hiparsimulator/oldstepper.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

enum MPITag {DEFAULT=0, GHOST_ZONE=1};

template<class CELL_TYPE>
class ParallelStepper 
{
    friend class ParallelStepperTest;
    // fixme: remove
    friend class HiParSimulatorTest;
    // fixme: remove once we have a real CoordinationGroup
    friend class UpdateGroupTest;
public:
    typedef DisplacedGrid<CELL_TYPE> GridType;
    typedef MPILayer::MPIRegionPointer MPIRegionPointer;
    typedef SuperMap<int, SuperVector<Region<2> > > RegionVecMap;
    typedef SuperMap<int, SuperVector<MPIRegionPointer> > MPIRegionVecMap;

    inline ParallelStepper(
        MPI::Comm *communicator = &MPI::COMM_WORLD, 
        PatchProvider<GridType> *ghostZonePatchProvider_ = 0, 
        PatchAccepter<GridType> *ghostZonePatchAccepter_ = 0) :
        mpiLayer(communicator),
        ghostZonePatchProvider(ghostZonePatchProvider_),
        ghostZonePatchAccepter(ghostZonePatchAccepter_)
    {}

    inline ~ParallelStepper()
    {
        waitForGhostZones();
    }
 
    // fixme: curStop is misleading and so is nextStep
    //fixme: test with curStop == startNanoStep or curStop == nextStop
    inline void nanoStep(
        const unsigned& curStop, 
        const unsigned& nextStop,
        const unsigned& startNanoStep)
    {
        if (startNanoStep > curStop || curStop > nextStop)
            throw std::logic_error(
                "Expected curStop to be after startNanoStep and nextStop after curStop");
        unsigned deltaT = curStop - startNanoStep;
        unsigned nextDeltaT = nextStop - curStop;
        unsigned nanoStepCounter = startNanoStep;

        while (deltaT > 0) {
            if (ghostZonePatchProvider) 
                ghostZonePatchProvider->get(
                    *oldGrid, 
                    partitionManager->getOuterOutgroupGhostZoneFragment(), 
                    nanoStepCounter);
            waitForGhostZones();
            unsigned thisHopLength = 
                std::min(deltaT, 
                         std::min(ghostZoneWidth(), validGhostZoneWidth));
            deltaT -= thisHopLength;
            unsigned nextHopLength = deltaT? 
                std::min(ghostZoneWidth(), deltaT) : 
                std::min(ghostZoneWidth(), nextDeltaT);
            updateGhostZones(thisHopLength, nanoStepCounter);
            sendGhostZones(nextHopLength);
            recvGhostZones(nextHopLength);
            validGhostZoneWidth = nextHopLength;
            updateInnerSet(0, thisHopLength, nanoStepCounter); 
            nanoStepCounter += thisHopLength;
            std::swap(oldGrid, newGrid); 
        }
    }

    inline void resetRegions(
        PartitionManager<2> *_partitionManager,
        Initializer<CELL_TYPE> *initializer)
    {
        partitionManager = _partitionManager;
        validGhostZoneWidth = ghostZoneWidth();
        SuperVector<CoordBox<2> > boundingBoxes(mpiLayer.size());
        CoordBox<2> ownBoundingBox(
            partitionManager->ownRegion().boundingBox());
        mpiLayer.allGather(ownBoundingBox, &boundingBoxes);
        partitionManager->resetGhostZones(boundingBoxes);
        resetGrids(initializer);
        registerGhostZones();
    }

    // fixme: make return type const
    GridType *getGrid() const
    {
        return &*oldGrid;
    }

    // fixme: remove
    GridType *getNewGrid() const
    {
        return &*newGrid;
    }

private:
    OldStepper<CELL_TYPE> stepper;
    unsigned validGhostZoneWidth;
    MPILayer mpiLayer;
    PartitionManager<2> *partitionManager;
    MPIRegionVecMap innerGhostZoneFragments;
    MPIRegionVecMap outerGhostZoneFragments;
    boost::shared_ptr<GridType> oldGrid;
    boost::shared_ptr<GridType> newGrid;
    boost::shared_ptr<GridType> workGrid1;
    boost::shared_ptr<GridType> workGrid2;
    PatchProvider<GridType> *ghostZonePatchProvider;
    PatchAccepter<GridType> *ghostZonePatchAccepter;

    inline const unsigned& ghostZoneWidth()
    {
        return partitionManager->getGhostZoneWidth();
    }

    inline void resetGrids(Initializer<CELL_TYPE> *initializer)
    {
        // finish transfers before deleting grids
        waitForGhostZones();

        CoordBox<2> boundingBox = 
            partitionManager->ownExpandedRegion().boundingBox();
        oldGrid.reset(new GridType(boundingBox));
        newGrid.reset(new GridType(boundingBox));
        workGrid1.reset(new GridType(boundingBox));
        workGrid2.reset(new GridType(boundingBox));
        initializer->grid(&*oldGrid);
        newGrid->getEdgeCell() = oldGrid->getEdgeCell();
        workGrid1->getEdgeCell() = oldGrid->getEdgeCell();
        workGrid2->getEdgeCell() = oldGrid->getEdgeCell();
    }

    inline void updateGhostZones(
        const unsigned& nanoSteps, 
        const unsigned& nanoStepCounter)
    {
        if (nanoSteps > ghostZoneWidth())
            throw std::logic_error(
                "Cannot update ghost zones for nanoSteps > ghostZoneWidth()");
        unsigned startStep = ghostZoneWidth() + 1 - nanoSteps;
        unsigned endStep = ghostZoneWidth() + 1;
        // fixme: can we remove this if-statement somehow?
        if (ghostZonePatchAccepter) {
            stepper.update(
                oldGrid, 
                workGrid1, 
                workGrid2, 
                newGrid, 
                RimMarker<2>(*partitionManager),
                startStep,
                endStep,
                nanoStepCounter, 
                ghostZonePatchAccepter);
        } else {
            stepper.update(
                oldGrid, 
                workGrid1, 
                workGrid2, 
                newGrid, 
                RimMarker<2>(*partitionManager),
                startStep,
                endStep,
                nanoStepCounter);
        }
    }

    inline void updateInnerSet(
        const unsigned& startWidth, 
        const unsigned& nanoSteps, 
        const unsigned& nanoStepCounter)
    {
        // if the final stage (the innermost kernel) of the inner set
        // is empty, then there is no need to update it at all as the
        // ghost zone updates will cover it.
        if (partitionManager->innerSet(ghostZoneWidth()).empty())
            return;
        if (nanoSteps > ghostZoneWidth())
            throw std::logic_error(
                "Cannot update inner set for nanoSteps > ghostZoneWidth()");
        stepper.update(
            oldGrid, 
            workGrid1, 
            workGrid2, 
            newGrid, 
            InnerSetMarker<2>(*partitionManager),
            startWidth,
            startWidth + nanoSteps,
            nanoStepCounter);
    }

    inline void sendGhostZones(const unsigned& width)
    {
        if (width > ghostZoneWidth())
            throw std::logic_error(
                "Cannot send ghost zones of width > ghostZoneWidth()");

        for (MPIRegionVecMap::iterator i = innerGhostZoneFragments.begin(); 
             i != innerGhostZoneFragments.end(); ++i)
            if (i->first != PartitionManager<2>::OUTGROUP)
                mpiLayer.sendRegion(
                    newGrid->baseAddress(), 
                    i->second[width], 
                    i->first, 
                    GHOST_ZONE);
    }    

    inline void recvGhostZones(const unsigned& width)
    {
        if (width > ghostZoneWidth())
            throw std::logic_error(
                "Cannot receive ghost zones of width > ghostZoneWidth()");
        for (MPIRegionVecMap::iterator i = 
                 outerGhostZoneFragments.begin(); 
             i != outerGhostZoneFragments.end(); ++i) 
            mpiLayer.recvRegion(newGrid->baseAddress(), 
                                i->second[width], i->first, GHOST_ZONE);

    }    

    inline void waitForGhostZones()       
    {
        mpiLayer.wait(GHOST_ZONE);
    }    

    inline void registerGhostZones()
    {
        const RegionVecMap& innerGhostZones = 
            partitionManager->getInnerGhostZoneFragments();
        const RegionVecMap& outerGhostZones = 
            partitionManager->getOuterGhostZoneFragments();
        for (RegionVecMap::const_iterator i = innerGhostZones.begin(); 
             i != innerGhostZones.end(); ++i) {
            if (i->first != PartitionManager<2>::OUTGROUP) {
                innerGhostZoneFragments[i->first].resize(ghostZoneWidth() + 1);
                for (int width = 0; width <= ghostZoneWidth(); ++width)
                    innerGhostZoneFragments[i->first][width] = 
                        mpiLayer.registerRegion(*oldGrid, i->second[width]);
            }
        }
        for (RegionVecMap::const_iterator i = outerGhostZones.begin(); 
             i != outerGhostZones.end(); ++i) {
            if (i->first != PartitionManager<2>::OUTGROUP) {
                outerGhostZoneFragments[i->first].resize(ghostZoneWidth() + 1);
                for (int width = 0; width <= ghostZoneWidth(); ++width) 
                    outerGhostZoneFragments[i->first][width] = 
                        mpiLayer.registerRegion(*oldGrid, i->second[width]);
            }
        }
    }
};

}
}

#endif
#endif
