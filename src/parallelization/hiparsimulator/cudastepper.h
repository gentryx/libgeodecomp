#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_CUDASTEPPER_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_CUDASTEPPER_H

#include <libgeodecomp/geometry/cudaregion.h>
#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/storage/cudagrid.h>
#include <libgeodecomp/storage/patchbufferfixed.h>
#include <libgeodecomp/storage/updatefunctor.h>

namespace LibGeoDecomp {

namespace HiParSimulator {

namespace CUDAStepperHelpers {

template<int DIM>
class LoadAbsoluteCoord;

template<>
class LoadAbsoluteCoord<2>
{
public:
    __device__
    Coord<2> operator()(int regionIndex, int *coords, int regionSize)
    {
        int x = coords[regionIndex + 0 * regionSize];
        int y = coords[regionIndex + 1 * regionSize];

        return Coord<2>(x, y);
    }
};

template<>
class LoadAbsoluteCoord<3>
{
public:
    __device__
    Coord<3> operator()(int regionIndex, int *coords, int regionSize)
    {
        int x = coords[regionIndex + 0 * regionSize];
        int y = coords[regionIndex + 1 * regionSize];
        int z = coords[regionIndex + 2 * regionSize];

        return Coord<3>(x, y, z);
    }
};

// fixme: inspect resulting PTX of this implementation and benchmark the code
template<typename CELL_TYPE, typename TOPOLOGY, int DIM>
class SimpleHood;

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
    int regionIndex = blockIdx.x * blockDim.x + threadIdx.x;
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
void updateKernel(CELL_TYPE *gridDataOld, CELL_TYPE *edgeCell, CELL_TYPE *gridDataNew, int nanoStep, int *coords, int regionSize, CoordBox<DIM> boundingBox)
{
    int regionIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (regionIndex >= regionSize) {
        return;
    }

    Coord<DIM> relativeCoord =
        LoadAbsoluteCoord<DIM>()(regionIndex, coords, regionSize) - boundingBox.origin;
    int gridIndex = relativeCoord.toIndex(boundingBox.dimensions);

    gridDataNew[gridIndex].update(
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
class CUDAStepper : public Stepper<CELL_TYPE>
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

    using Stepper<CELL_TYPE>::addPatchAccepter;
    using Stepper<CELL_TYPE>::initializer;
    using Stepper<CELL_TYPE>::guessOffset;
    using Stepper<CELL_TYPE>::patchAccepters;
    using Stepper<CELL_TYPE>::patchProviders;
    using Stepper<CELL_TYPE>::partitionManager;

    using Stepper<CELL_TYPE>::chronometer;

    inline CUDAStepper(
        boost::shared_ptr<PartitionManagerType> partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        const PatchAccepterVec& ghostZonePatchAccepters = PatchAccepterVec(),
        const PatchAccepterVec& innerSetPatchAccepters = PatchAccepterVec()) :
        ParentType(partitionManager, initializer)
    {
        curStep = initializer->startStep();
        curNanoStep = 0;

        for (std::size_t i = 0; i < ghostZonePatchAccepters.size(); ++i) {
            addPatchAccepter(ghostZonePatchAccepters[i], ParentType::GHOST);
        }
        for (std::size_t i = 0; i < innerSetPatchAccepters.size(); ++i) {
            addPatchAccepter(innerSetPatchAccepters[i], ParentType::INNER_SET);
        }

        initGrids();
    }

    inline void update(std::size_t nanoSteps)
    {
        for (std::size_t i = 0; i < nanoSteps; ++i)
        {
            update();
        }
    }

    inline virtual std::pair<std::size_t, std::size_t> currentStep() const
    {
        return std::make_pair(curStep, curNanoStep);
    }

    inline virtual const GridType& grid() const
    {
        return *oldGrid;
    }

private:
    std::size_t curStep;
    std::size_t curNanoStep;
    unsigned validGhostZoneWidth;
    boost::shared_ptr<GridType> oldGrid;
    boost::shared_ptr<GridType> newGrid;
    // fixme: get rid of dummyGrid
    boost::shared_ptr<GridType> dummyGrid;
    boost::shared_ptr<CUDAGridType> oldDeviceGrid;
    boost::shared_ptr<CUDAGridType> newDeviceGrid;
    PatchBufferType2 rimBuffer;
    PatchBufferType1 kernelBuffer;
    Region<DIM> kernelFraction;

    inline void update()
    {
        std::cout << "update()\n";
        unsigned index = ghostZoneWidth() - --validGhostZoneWidth;
        const Region<DIM>& region = innerSet(index);
        {
            TimeComputeInner t(&chronometer);

            // fixme:
            // UpdateFunctor<CELL_TYPE>()(
            //     region,
            //     Coord<DIM>(),
            //     Coord<DIM>(),
            //     *oldGrid,
            //     &*dummyGrid,
            //     curNanoStep);

            oldDeviceGrid->loadRegion(*oldGrid, region);
            {
                CUDARegion<DIM> cudaRegion(region);
                // fixme: choose grid-/blockDim in a better way
                dim3 gridDim(512);
                dim3 blockDim(32);
                CUDAStepperHelpers::updateKernel<<<gridDim, blockDim>>>(
                    oldDeviceGrid->data(),
                    oldDeviceGrid->edgeCell(),
                    newDeviceGrid->data(),
                    curNanoStep,
                    cudaRegion.data(),
                    region.size(),
                    oldDeviceGrid->boundingBox());
            }
            newDeviceGrid->saveRegion(&*newGrid,  region);

            std::swap(oldGrid, newGrid);

            ++curNanoStep;
            if (curNanoStep == NANO_STEPS) {
                curNanoStep = 0;
                curStep++;
            }
        }

        notifyPatchAccepters(region, ParentType::INNER_SET, globalNanoStep());

        if (validGhostZoneWidth == 0) {
            updateGhost();
            resetValidGhostZoneWidth();
        }

        notifyPatchProviders(region, ParentType::INNER_SET, globalNanoStep());
    }

    inline void notifyPatchAccepters(
        const Region<DIM>& region,
        const typename ParentType::PatchType& patchType,
        std::size_t nanoStep)
    {
        TimePatchAccepters t(&chronometer);

        for (typename ParentType::PatchAccepterList::iterator i =
                 patchAccepters[patchType].begin();
             i != patchAccepters[patchType].end();
             ++i) {
            if (nanoStep == (*i)->nextRequiredNanoStep()) {
                (*i)->put(*oldGrid, region, nanoStep);
            }
        }
    }

    inline void notifyPatchProviders(
        const Region<DIM>& region,
        const typename ParentType::PatchType& patchType,
        std::size_t nanoStep)
    {
        TimePatchProviders t(&chronometer);

        for (typename ParentType::PatchProviderList::iterator i =
                 patchProviders[patchType].begin();
             i != patchProviders[patchType].end();
             ++i) {
            (*i)->get(
                &*oldGrid,
                region,
                nanoStep);
        }
    }

    inline std::size_t globalNanoStep() const
    {
        return curStep * NANO_STEPS + curNanoStep;
    }

    inline void initGrids()
    {
        Coord<DIM> topoDim = initializer->gridDimensions();
        CoordBox<DIM> gridBox;
        guessOffset(&gridBox.origin, &gridBox.dimensions);

        oldGrid.reset(new GridType(gridBox, CELL_TYPE(), CELL_TYPE(), topoDim));
        newGrid.reset(new GridType(gridBox, CELL_TYPE(), CELL_TYPE(), topoDim));
        dummyGrid.reset(new GridType(gridBox, CELL_TYPE(), CELL_TYPE(), topoDim));
        oldDeviceGrid.reset(new CUDAGridType(gridBox, topoDim));
        newDeviceGrid.reset(new CUDAGridType(gridBox, topoDim));

        initializer->grid(&*oldGrid);
        newGrid->getEdgeCell() = oldGrid->getEdgeCell();
        resetValidGhostZoneWidth();

        notifyPatchAccepters(
            rim(),
            ParentType::GHOST,
            globalNanoStep());
        notifyPatchAccepters(
            innerSet(0),
            ParentType::INNER_SET,
            globalNanoStep());

        kernelBuffer = PatchBufferType1(getVolatileKernel());
        rimBuffer = PatchBufferType2(rim());
        saveRim(globalNanoStep());
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
        std::cout << "updateGhost()\n";
        {
            TimeComputeGhost t(&chronometer);

            // fixme: skip all this ghost zone buffering for
            // ghostZoneWidth == 1?

            // 1: Prepare grid. The following update of the ghostzone will
            // destroy parts of the kernel, which is why we'll
            // save/restore those.
            saveKernel();
            // We need to restore the rim since it got destroyed while the
            // kernel was updated.
            restoreRim(false);
        }

        // 2: actual ghostzone update
        std::size_t oldNanoStep = curNanoStep;
        std::size_t oldStep = curStep;
        std::size_t curGlobalNanoStep = globalNanoStep();

        for (std::size_t t = 0; t < ghostZoneWidth(); ++t) {
            notifyPatchProviders(rim(t), ParentType::GHOST, globalNanoStep());

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

                std::swap(oldGrid, newGrid);

                ++curGlobalNanoStep;
            }

            notifyPatchAccepters(rim(), ParentType::GHOST, curGlobalNanoStep);
        }

        {
            TimeComputeGhost t(&chronometer);
            curNanoStep = oldNanoStep;
            curStep = oldStep;

            saveRim(curGlobalNanoStep);
            if (ghostZoneWidth() % 2) {
                std::swap(oldGrid, newGrid);
            }

            // 3: restore grid for kernel update
            restoreRim(true);
            restoreKernel();
        }
    }

    inline unsigned ghostZoneWidth() const
    {
        return partitionManager->getGhostZoneWidth();
    }

    inline const Region<DIM>& rim(unsigned offset) const
    {
        return partitionManager->rim(offset);
    }

    inline const Region<DIM>& rim() const
    {
        return rim(ghostZoneWidth());
    }

    inline const Region<DIM>& innerSet(unsigned offset) const
    {
        return partitionManager->innerSet(offset);
    }

    inline const Region<DIM>& getVolatileKernel() const
    {
        return partitionManager->getVolatileKernel();
    }

    inline void resetValidGhostZoneWidth()
    {
        validGhostZoneWidth = ghostZoneWidth();
    }

    inline void saveRim(std::size_t nanoStep)
    {
        rimBuffer.pushRequest(nanoStep);
        rimBuffer.put(*oldGrid, rim(), nanoStep);
    }

    inline void restoreRim(bool remove)
    {
        rimBuffer.get(&*oldGrid, rim(), globalNanoStep(), remove);
    }

    inline void saveKernel()
    {
        kernelBuffer.pushRequest(globalNanoStep());
        kernelBuffer.put(*oldGrid,
                         innerSet(ghostZoneWidth()),
                         globalNanoStep());
    }

    inline void restoreKernel()
    {
        kernelBuffer.get(
            &*oldGrid,
            getVolatileKernel(),
            globalNanoStep(),
            true);
    }
};

}
}

#endif
