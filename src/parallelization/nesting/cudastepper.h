#ifndef LIBGEODECOMP_PARALLELIZATION_NESTING_CUDASTEPPER_H
#define LIBGEODECOMP_PARALLELIZATION_NESTING_CUDASTEPPER_H

#include <libgeodecomp/geometry/cudaregion.h>
#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/parallelization/nesting/commonstepper.h>
#include <libgeodecomp/storage/cudagrid.h>
#include <libgeodecomp/storage/updatefunctor.h>

namespace LibGeoDecomp {

namespace CUDAStepperHelpers {

/**
 * Internal helper class
 */
template<int DIM>
class LoadAbsoluteCoord;

/**
 * see above
 */
template<>
class LoadAbsoluteCoord<2>
{
public:
    __device__
    Coord<2> operator()(int regionIndex, const int *coords, int regionSize)
    {
        int x = coords[regionIndex + 0 * regionSize];
        int y = coords[regionIndex + 1 * regionSize];

        return Coord<2>(x, y);
    }
};

/**
 * see above
 */
template<>
class LoadAbsoluteCoord<3>
{
public:
    __device__
    Coord<3> operator()(int regionIndex, const int *coords, int regionSize)
    {
        int x = coords[regionIndex + 0 * regionSize];
        int y = coords[regionIndex + 1 * regionSize];
        int z = coords[regionIndex + 2 * regionSize];

        return Coord<3>(x, y, z);
    }
};

// fixme: inspect resulting PTX of this implementation and benchmark the code

/**
 * A simple/stupid implementation of a neighborhood object for CUDA.
 */
template<typename CELL_TYPE, typename TOPOLOGY, int DIM>
class SimpleHood;

/**
 * see above
 */
template<typename CELL_TYPE, typename TOPOLOGY>
class SimpleHood<CELL_TYPE, TOPOLOGY, 2>
{
public:
    typedef typename TOPOLOGY::RawTopologyType RawTopo;

    __device__
    SimpleHood(const Coord<2> *dim, const Coord<2> *index, const CELL_TYPE *grid, const CELL_TYPE *edgeCell) :
        dim(dim),
        index(index),
        grid(grid),
        edgeCell(edgeCell)
    {}

    template<int X, int Y, int Z>
    __device__
    const CELL_TYPE& operator[](FixedCoord<X, Y, Z> coord) const
    {
        int x = index->x() + X;
        int y = index->y() + Y;

        if (x < 0) {
            if (RawTopo::WRAP_AXIS0) {
                x += dim->x();
            } else {
                return *edgeCell;
            }
        }
        if (x >= dim->x()) {
            if (RawTopo::WRAP_AXIS0) {
                x -= dim->x();
            } else {
                return *edgeCell;
            }
        }

        if (y < 0) {
            if (RawTopo::WRAP_AXIS1) {
                y += dim->y();
            } else {
                return *edgeCell;
            }
        }
        if (y >= dim->y()) {
            if (RawTopo::WRAP_AXIS1) {
                y -= dim->y();
            } else {
                return *edgeCell;
            }
        }

        return grid[(y * dim->x()) + x];
    }


private:
    const Coord<2> *dim;
    const Coord<2> *index;
    const CELL_TYPE *grid;
    const CELL_TYPE *edgeCell;
};

/**
 * see above
 */
template<typename CELL_TYPE, typename TOPOLOGY>
class SimpleHood<CELL_TYPE, TOPOLOGY, 3>
{
public:
    typedef typename TOPOLOGY::RawTopologyType RawTopo;

    __device__
    SimpleHood(const Coord<3> *dim, const Coord<3> *index, const CELL_TYPE *grid, const CELL_TYPE *edgeCell) :
        dim(dim),
        index(index),
        grid(grid),
        edgeCell(edgeCell)
    {}

    template<int X, int Y, int Z>
    __device__
    const CELL_TYPE& operator[](FixedCoord<X, Y, Z> coord) const
    {
        int x = index->x() + X;
        int y = index->y() + Y;
        int z = index->z() + Z;

        if (x < 0) {
            if (RawTopo::WRAP_AXIS0) {
                x += dim->x();
            } else {
                return *edgeCell;
            }
        }
        if (x >= dim->x()) {
            if (RawTopo::WRAP_AXIS0) {
                x -= dim->x();
            } else {
                return *edgeCell;
            }
        }

        if (y < 0) {
            if (RawTopo::WRAP_AXIS1) {
                y += dim->y();
            } else {
                return *edgeCell;
            }
        }
        if (y >= dim->y()) {
            if (RawTopo::WRAP_AXIS1) {
                y -= dim->y();
            } else {
                return *edgeCell;
            }
        }

        if (z < 0) {
            if (RawTopo::WRAP_AXIS2) {
                z += dim->z();
            } else {
                return *edgeCell;
            }
        }
        if (z >= dim->z()) {
            if (RawTopo::WRAP_AXIS2) {
                z -= dim->z();
            } else {
                return *edgeCell;
            }
        }

        return grid[(z * dim->x() * dim->y()) + (y * dim->x()) + x];
    }


private:
    const Coord<3> *dim;
    const Coord<3> *index;
    const CELL_TYPE *grid;
    const CELL_TYPE *edgeCell;
};

template<int DIM, typename CELL_TYPE>
__global__
void copyKernel(CELL_TYPE *gridDataOld, CELL_TYPE *gridDataNew, int *coords, int regionSize, CoordBox<DIM> boundingBox)
{
    // fixme:
    int regionIndex = blockIdx.x;
    if (regionIndex >= regionSize) {
        return;
    }

    Coord<DIM> relativeCoord =
        LoadAbsoluteCoord<DIM>()(regionIndex, coords, regionSize) - boundingBox.origin;
    int gridIndex = relativeCoord.toIndex(boundingBox.dimensions);

    gridDataNew[gridIndex] = gridDataOld[gridIndex];
}

template<int DIM, typename CELL_TYPE>
__global__
void updateKernel(CELL_TYPE *gridDataOld, CELL_TYPE *edgeCell, CELL_TYPE *gridDataNew, int nanoStep, const int *coords, int regionSize, CoordBox<DIM> boundingBox)
{
    // fixme:
    int regionIndex = blockIdx.x;
    if (regionIndex >= regionSize) {
        return;
    }

    Coord<DIM> relativeCoord =
        LoadAbsoluteCoord<DIM>()(regionIndex, coords, regionSize) - boundingBox.origin;
    int gridIndex = relativeCoord.toIndex(boundingBox.dimensions);

    gridDataNew[gridIndex].updateCUDA(
        SimpleHood<CELL_TYPE, typename APITraits::SelectTopology<CELL_TYPE>::Value, DIM>(
            &boundingBox.dimensions, &relativeCoord, gridDataOld, edgeCell),
        nanoStep);
}

}

/**
 * The CUDAStepper offloads cell updates to a CUDA enabled GPU.
 *
 * FIXME: add option to select CUDA device in c-tor. we'll need to use a custom stream in this case.
 * FIXME: add 2d unit test
 */
template<typename CELL_TYPE>
class CUDAStepper : public CommonStepper<CELL_TYPE>
{
public:
    friend class CUDAStepperRegionTest;
    friend class CUDAStepperBasicTest;
    friend class CUDAStepperTest;

    typedef typename Stepper<CELL_TYPE>::Topology Topology;
    const static int DIM = Topology::DIM;
    const static unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL_TYPE>::VALUE;

    typedef class Stepper<CELL_TYPE> ParentType;
    typedef typename ParentType::GridType GridType;
    typedef CUDAGrid<CELL_TYPE, Topology, true> CUDAGridType;
    typedef PartitionManager<Topology> PartitionManagerType;
    typedef PatchBufferFixed<GridType, GridType, 1> PatchBufferType1;
    typedef PatchBufferFixed<GridType, GridType, 2> PatchBufferType2;
    typedef typename ParentType::PatchAccepterVec PatchAccepterVec;
    typedef typename ParentType::PatchProviderVec PatchProviderVec;

    using CommonStepper<CELL_TYPE>::initializer;
    using CommonStepper<CELL_TYPE>::patchAccepters;
    using CommonStepper<CELL_TYPE>::patchProviders;
    using CommonStepper<CELL_TYPE>::partitionManager;
    using CommonStepper<CELL_TYPE>::chronometer;
    using CommonStepper<CELL_TYPE>::notifyPatchAccepters;
    using CommonStepper<CELL_TYPE>::notifyPatchProviders;

    using CommonStepper<CELL_TYPE>::innerSet;
    using CommonStepper<CELL_TYPE>::saveKernel;
    using CommonStepper<CELL_TYPE>::restoreRim;
    using CommonStepper<CELL_TYPE>::globalNanoStep;
    using CommonStepper<CELL_TYPE>::rim;
    using CommonStepper<CELL_TYPE>::resetValidGhostZoneWidth;
    using CommonStepper<CELL_TYPE>::initGridsCommon;
    using CommonStepper<CELL_TYPE>::getVolatileKernel;
    using CommonStepper<CELL_TYPE>::saveRim;
    using CommonStepper<CELL_TYPE>::getInnerRim;
    using CommonStepper<CELL_TYPE>::restoreKernel;

    using CommonStepper<CELL_TYPE>::curStep;
    using CommonStepper<CELL_TYPE>::curNanoStep;
    using CommonStepper<CELL_TYPE>::validGhostZoneWidth;
    using CommonStepper<CELL_TYPE>::ghostZoneWidth;
    using CommonStepper<CELL_TYPE>::oldGrid;
    using CommonStepper<CELL_TYPE>::newGrid;
    using CommonStepper<CELL_TYPE>::rimBuffer;
    using CommonStepper<CELL_TYPE>::kernelBuffer;
    using CommonStepper<CELL_TYPE>::kernelFraction;

    inline CUDAStepper(
        boost::shared_ptr<PartitionManagerType> partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        const PatchAccepterVec& ghostZonePatchAccepters = PatchAccepterVec(),
        const PatchAccepterVec& innerSetPatchAccepters = PatchAccepterVec(),
        const PatchProviderVec& ghostZonePatchProviders = PatchProviderVec(),
        const PatchProviderVec& innerSetPatchProviders = PatchProviderVec()) :
        CommonStepper<CELL_TYPE>(
            partitionManager,
            initializer,
            ghostZonePatchAccepters,
            innerSetPatchAccepters,
            ghostZonePatchProviders,
            innerSetPatchProviders)
    {
        initGrids();
    }

private:
    boost::shared_ptr<CUDAGridType> oldDeviceGrid;
    boost::shared_ptr<CUDAGridType> newDeviceGrid;
    std::vector<boost::shared_ptr<CUDARegion<DIM> > > deviceInnerSets;

    inline void update1()
    {
        using std::swap;

        unsigned index = ghostZoneWidth() - --validGhostZoneWidth;
        const Region<DIM>& region = innerSet(index);
        {
            TimeComputeInner t(&chronometer);

            const CUDARegion<DIM>& cudaRegion = deviceInnerSet(index);
            // fixme: choose grid-/blockDim in a better way
            // dim3 gridDim(region.size());
            // dim3 blockDim(CELL_TYPE::SIZE);

            dim3 gridDim(200);
            dim3 blockDim(512);

            CUDAStepperHelpers::updateKernel<<<gridDim, blockDim>>>(
                oldDeviceGrid->data(),
                oldDeviceGrid->edgeCell(),
                newDeviceGrid->data(),
                curNanoStep,
                cudaRegion.data(),
                region.size(),
                oldDeviceGrid->boundingBox());

            swap(oldDeviceGrid, newDeviceGrid);
            swap(oldGrid, newGrid);

            ++curNanoStep;
            if (curNanoStep == NANO_STEPS) {
                curNanoStep = 0;
                curStep++;
            }
        }

        notifyPatchAcceptersInnerSet(region, globalNanoStep());

        if (validGhostZoneWidth == 0) {
            updateGhost();
            resetValidGhostZoneWidth();
        }

        notifyPatchProvidersInnerSet(region, globalNanoStep());
    }

    inline void notifyPatchAcceptersGhostZones(
        const Region<DIM>& region,
        std::size_t nanoStep)
    {
        notifyPatchAccepters(region, ParentType::GHOST, nanoStep);
    }

    inline void notifyPatchProvidersGhostZones(
        const Region<DIM>& region,
        std::size_t nanoStep)
    {
        notifyPatchProviders(region, ParentType::GHOST, nanoStep);
    }

    inline void notifyPatchAcceptersInnerSet(
        const Region<DIM>& region,
        std::size_t nanoStep)
    {
        TimePatchAccepters t(&chronometer);

        bool copyRequired = false;

        for (typename ParentType::PatchAccepterList::iterator i =
                 patchAccepters[ParentType::INNER_SET].begin();
             i != patchAccepters[ParentType::INNER_SET].end();
             ++i) {
            if (nanoStep == (*i)->nextRequiredNanoStep()) {
                copyRequired = true;
            }
        }

        if (!copyRequired) {
            return;
        }
        oldDeviceGrid->saveRegion(&*oldGrid, region);

        for (typename ParentType::PatchAccepterList::iterator i =
                 patchAccepters[ParentType::INNER_SET].begin();
             i != patchAccepters[ParentType::INNER_SET].end();
             ++i) {
            if (nanoStep == (*i)->nextRequiredNanoStep()) {
                (*i)->put(
                    *oldGrid,
                    region,
                    partitionManager->getSimulationArea(),
                    nanoStep,
                    partitionManager->rank());
            }
        }
    }

    inline void notifyPatchProvidersInnerSet(
        const Region<DIM>& region,
        std::size_t nanoStep)
    {
        TimePatchProviders t(&chronometer);

        bool copyRequired = false;

        for (typename ParentType::PatchProviderList::iterator i =
                 patchProviders[ParentType::INNER_SET].begin();
             i != patchProviders[ParentType::INNER_SET].end();
             ++i) {
            if (nanoStep == (*i)->nextAvailableNanoStep()) {
                copyRequired = true;
            }
        }

        if (!copyRequired) {
            return;
        }

        std::cout << "copy out for PatchProviders at nanoStep " << nanoStep << "\n";
        oldDeviceGrid->saveRegion(&*oldGrid, region);

        for (typename ParentType::PatchProviderList::iterator i =
                 patchProviders[ParentType::INNER_SET].begin();
             i != patchProviders[ParentType::INNER_SET].end();
             ++i) {
            if (nanoStep == (*i)->nextAvailableNanoStep()) {
                (*i)->get(
                    &*oldGrid,
                    region,
                    partitionManager->getSimulationArea(),
                    nanoStep,
                    partitionManager->rank());
            }
        }

        std::cout << "copy in for PatchProviders at nanoStep " << nanoStep << "\n";
        oldDeviceGrid->loadRegion(*oldGrid, region);
    }

    inline void initGrids()
    {
        Coord<DIM> topoDim = initializer->gridDimensions();
        CoordBox<DIM> gridBox = initGridsCommon();

        oldDeviceGrid.reset(new CUDAGridType(gridBox, topoDim));
        newDeviceGrid.reset(new CUDAGridType(gridBox, topoDim));

        Region<DIM> gridRegion;
        gridRegion << gridBox;
        oldDeviceGrid->loadRegion(*oldGrid, gridRegion);
        oldDeviceGrid->setEdge(oldGrid->getEdgeCell());
        newDeviceGrid->setEdge(oldGrid->getEdgeCell());

        deviceInnerSets.resize(0);
        for (std::size_t i = 0; i <= ghostZoneWidth(); ++i) {
            deviceInnerSets.push_back(
                boost::shared_ptr<CUDARegion<DIM> >(
                    new CUDARegion<DIM>(innerSet(i))));
        }

        notifyPatchAcceptersGhostZones(
            rim(),
            globalNanoStep());
        notifyPatchAcceptersInnerSet(
            innerSet(0),
            globalNanoStep());

        updateGhost();
    }

    /**
     * computes the next ghost zone at time "t_1 = globalNanoStep() +
     * ghostZoneWidth()". Expects that oldGrid has its kernel and its
     * outer ghostzone updated to time "globalNanoStep()" and that the
     * inner ghostzones (rim) at time t_1 can be retrieved from the
     * internal patch buffer. Will leave oldgrid in a state so that
     * its whole ownRegion() will be at time t_1 and the rim will be
     * saved to the patchBuffer at "t2 = t1 + ghostZoneWidth()".
     */
    inline void updateGhost()
    {
        using std::swap;
        {
            TimeComputeGhost t(&chronometer);

            // fixme: skip all this ghost zone buffering for
            // ghostZoneWidth == 1?

            // 1: Prepare grid. The following update of the ghostzone will
            // destroy parts of the kernel, which is why we'll
            // save/restore those.

            // We need to restore the rim since it got destroyed while the
            // kernel was updated.

            oldDeviceGrid->saveRegion(&*oldGrid, rim());
        }

        // 2: actual ghostzone update
        std::size_t oldNanoStep = curNanoStep;
        std::size_t oldStep = curStep;
        std::size_t curGlobalNanoStep = globalNanoStep();

        for (std::size_t t = 0; t < ghostZoneWidth(); ++t) {
            notifyPatchProvidersGhostZones(rim(t), globalNanoStep());

            {
                TimeComputeGhost timer(&chronometer);

                const Region<DIM>& region = rim(t + 1);
                UpdateFunctor<CELL_TYPE>()(
                    region,
                    Coord<DIM>(),
                    Coord<DIM>(),
                    *oldGrid,
                    &*newGrid,
                    curNanoStep);

                ++curNanoStep;
                if (curNanoStep == NANO_STEPS) {
                    curNanoStep = 0;
                    curStep++;
                }

                swap(oldGrid, newGrid);

                ++curGlobalNanoStep;
            }

            notifyPatchAcceptersGhostZones(rim(), curGlobalNanoStep);
        }

        {
            TimeComputeGhost t(&chronometer);
            curNanoStep = oldNanoStep;
            curStep = oldStep;

            // fixme: we don't need this any longer, as the kernel is handled on the device
            // saveRim(curGlobalNanoStep);
            if (ghostZoneWidth() % 2) {
                swap(oldGrid, newGrid);
            }

            // 3: restore grid for kernel update

            oldDeviceGrid->loadRegion(*oldGrid, getInnerRim());
        }
    }

    inline const CUDARegion<DIM>& deviceInnerSet(unsigned offset) const
    {
        return *deviceInnerSets[offset];
    }
};

}

#endif
