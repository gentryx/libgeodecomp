#ifndef LIBGEODECOMP_PARALLELIZATION_CUDASIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_CUDASIMULATOR_H

#include <libgeodecomp/geometry/fixedcoord.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/cudautil.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>
#include <libgeodecomp/storage/cudaupdatefunctor.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/proxygrid.h>

namespace LibGeoDecomp {

// fixme: rename Cuda to CUDA
namespace CudaSimulatorHelpers {

/**
 * Simple neighborhood object, optimized for GPUs (actually: NVCC as
 * we're prefering "int*" over "int" -- the switch from int* to int in
 * LibFlatArray's soa_accessor crilpled performance in our CUDA
 * performance tests, hence the creation of the soa_accessor_light...)
 * and AoS storage.
 */
template<class CELL_TYPE>
class HoodType
{
public:
    __device__
    HoodType(int *index,
             CELL_TYPE *grid,
             int *offsetY,
             int *offsetZ,
             int *addWest,
             int *addEast,
             int *addTop,
             int *addBottom,
             int *addSouth,
             int *addNorth) :
        index(index),
        grid(grid),
        offsetY(offsetY),
        offsetZ(offsetZ),
        addWest(addWest),
        addEast(addEast),
        addTop(addTop),
        addBottom(addBottom),
        addSouth(addSouth),
        addNorth(addNorth)
    {}

    template<int X, int Y, int Z>
    __device__
    const CELL_TYPE& operator[](FixedCoord<X, Y, Z> coord) const
    {
        return grid[
            *index +
            ((X < 0) ? *addWest   + X : 0) +
            ((X > 0) ? *addEast   + X : 0) +
            ((Y < 0) ? *addTop    + Y * *offsetY : 0) +
            ((Y > 0) ? *addBottom + Y * *offsetY : 0) +
            ((Z < 0) ? *addSouth  + Z * *offsetZ : 0) +
            ((Z > 0) ? *addNorth  + Z * *offsetZ : 0)];
    }

private:
    int *index;
    CELL_TYPE *grid;
    int *offsetY;
    int *offsetZ;
    int *addWest;
    int *addEast;
    int *addTop;
    int *addBottom;
    int *addSouth;
    int *addNorth;
};

/**
 * see CudaStepper::nanoStep() for a documentation of the parameters.
 */
template<bool WRAP_X_AXIS, bool WRAP_Y_AXIS, bool WRAP_Z_AXIS, class CELL_TYPE>
__global__
void kernel2D(CELL_TYPE *gridOld, CELL_TYPE *gridNew, int nanoStep, dim3 gridOffset, dim3 logicalGridDim, dim3 axisWrapOffset, int offsetY, int offsetZ, int wavefrontLength)
{
    // we need to distinguish logical coordinates and padded
    // coordinates: padded coords will be used to compute addresses
    // within the actual grid while logical coords correspond to a
    // cells position within the topology.
    int logicalX = blockIdx.x * blockDim.x + threadIdx.x;
    int logicalY = blockIdx.y * blockDim.y + threadIdx.y;

    int paddedX = logicalX - gridOffset.x;
    int paddedMinY = (logicalY + 0) * wavefrontLength - gridOffset.y;
    int paddedMaxY = (logicalY + 1) * wavefrontLength - gridOffset.y;
    int paddedMaxY2 = logicalGridDim.y - gridOffset.y;
    if (paddedMaxY2 < paddedMaxY) {
        paddedMaxY = paddedMaxY2;
    }

    int addWest   = WRAP_X_AXIS && (logicalX == 0                     ) ?  axisWrapOffset.x : 0;
    int addEast   = WRAP_X_AXIS && (logicalX == (logicalGridDim.x - 1)) ? -axisWrapOffset.x : 0;
    int addTop    =  0;
    int addBottom =  0;
    int addNorth  =  0;
    int addSouth  =  0;

    int index =
        paddedMinY * offsetY +
        paddedX;

    if (logicalX >= logicalGridDim.x) {
        return;
    }

    HoodType<CELL_TYPE> hood(
        &index,
        gridOld,
        &offsetY,
        &offsetZ,
        &addWest,
        &addEast,
        &addTop,
        &addBottom,
        &addSouth,
        &addNorth);

    if (WRAP_Y_AXIS && (paddedMinY == 0)) {
        addTop    = axisWrapOffset.y;
        addBottom = 0;

        CudaUpdateFunctor<CELL_TYPE>()(gridNew, index, offsetY, hood, nanoStep);
        paddedMinY += 1;
    }

    addTop    = 0;
    addBottom = 0;

    if (WRAP_Y_AXIS && (paddedMaxY == logicalGridDim.y)) {
#pragma unroll
        for (int myY = paddedMinY; myY < (paddedMaxY - 1); ++myY) {
            CudaUpdateFunctor<CELL_TYPE>()(gridNew, index, offsetY, hood, nanoStep);
        }

        addTop = 0;
        addBottom = -axisWrapOffset.y;
        gridNew[index].update(hood, nanoStep);

    } else {
#pragma unroll
        for (int myY = paddedMinY; myY < paddedMaxY; ++myY) {
            CudaUpdateFunctor<CELL_TYPE>()(gridNew, index, offsetY, hood, nanoStep);
        }
    }
}

/**
 * see CudaStepper::nanoStep() for a documentation of the parameters.
 */
template<bool WRAP_X_AXIS, bool WRAP_Y_AXIS, bool WRAP_Z_AXIS, class CELL_TYPE>
__global__
void kernel3D(CELL_TYPE *gridOld, CELL_TYPE *gridNew, int nanoStep, dim3 gridOffset, dim3 logicalGridDim, dim3 axisWrapOffset, int offsetY, int offsetZ, int wavefrontLength)
{
    // we need to distinguish logical coordinates and padded
    // coordinates: padded coords will be used to compute addresses
    // within the actual grid while logical coords correspond to a
    // cells position within the topology.
    int logicalX = blockIdx.x * blockDim.x + threadIdx.x;
    int logicalY = blockIdx.y * blockDim.y + threadIdx.y;
    int logicalZ = blockIdx.z * blockDim.z + threadIdx.z;

    int paddedX = logicalX - gridOffset.x;
    int paddedY = logicalY - gridOffset.y;
    int paddedMinZ = (logicalZ + 0) * wavefrontLength - gridOffset.z;
    int paddedMaxZ = (logicalZ + 1) * wavefrontLength - gridOffset.z;
    int paddedMaxZ2 = logicalGridDim.z - gridOffset.z;
    if (paddedMaxZ2 < paddedMaxZ) {
        paddedMaxZ = paddedMaxZ2;
    }

    int addWest   = WRAP_X_AXIS && (logicalX == 0                     ) ?  axisWrapOffset.x : 0;
    int addEast   = WRAP_X_AXIS && (logicalX == (logicalGridDim.x - 1)) ? -axisWrapOffset.x : 0;
    int addTop    = WRAP_Y_AXIS && (logicalY == 0                     ) ?  axisWrapOffset.y : 0;
    int addBottom = WRAP_Y_AXIS && (logicalY == (logicalGridDim.y - 1)) ? -axisWrapOffset.y : 0;
    int addSouth =  0;
    int addNorth =  0;

    int index =
        paddedMinZ * offsetZ +
        paddedY    * offsetY +
        paddedX;

    if ((logicalX >= logicalGridDim.x) || (logicalY >= logicalGridDim.y)) {
        return;
    }

    HoodType<CELL_TYPE> hood(
        &index,
        gridOld,
        &offsetY,
        &offsetZ,
        &addWest,
        &addEast,
        &addTop,
        &addBottom,
        &addSouth,
        &addNorth);

    if (WRAP_Z_AXIS && (paddedMinZ == 0)) {
        addSouth = axisWrapOffset.z;
        addNorth = 0;

        CudaUpdateFunctor<CELL_TYPE>()(gridNew, index, offsetZ, hood, nanoStep);
        paddedMinZ += 1;
    }

    addSouth = 0;
    addNorth = 0;

    if (WRAP_Z_AXIS && (paddedMaxZ == logicalGridDim.z)) {
#pragma unroll
        for (int myZ = paddedMinZ; myZ < (paddedMaxZ - 1); ++myZ) {
            CudaUpdateFunctor<CELL_TYPE>()(gridNew, index, offsetZ, hood, nanoStep);
        }

        addSouth = 0;
        addNorth = -axisWrapOffset.z;
        CudaUpdateFunctor<CELL_TYPE>()(gridNew, index, offsetZ, hood, nanoStep);

    } else {
#pragma unroll
        for (int myZ = paddedMinZ; myZ < paddedMaxZ; ++myZ) {
            CudaUpdateFunctor<CELL_TYPE>()(gridNew, index, offsetZ, hood, nanoStep);
        }
    }
}

/**
 * Type gate for kernel selection based on the model's dimensionality.
 * Weird: placing the kernels as static methods directly into the
 * classes led to all sorts of "interesting" compiler errors (e.g.
 * "inline hint illegal").
 */
template<int DIM, bool WRAP_X_AXIS, bool WRAP_Y_AXIS, bool WRAP_Z_AXIS>
class KernelWrapper;

/**
 * See above
 */
template<bool WRAP_X_AXIS, bool WRAP_Y_AXIS, bool WRAP_Z_AXIS>
class KernelWrapper<1, WRAP_X_AXIS, WRAP_Y_AXIS, WRAP_Z_AXIS>
{
public:
    template<class CELL_TYPE>
    void operator()(dim3 cudaDimGrid, dim3 cudaDimBlock, CELL_TYPE *devGridOld, CELL_TYPE *devGridNew, int nanoStep, dim3 gridOffset, dim3 logicalGridDim, dim3 axisWrapOffset, int offsetY, int offsetZ, int wavefrontLength) const
    {
        kernel2D<WRAP_X_AXIS, WRAP_Y_AXIS, WRAP_Z_AXIS><<<cudaDimGrid, cudaDimBlock>>>(
            devGridOld, devGridNew, nanoStep, gridOffset, logicalGridDim, axisWrapOffset, offsetY, offsetZ, wavefrontLength);
    }
};

/**
 * See above
 */
template<bool WRAP_X_AXIS, bool WRAP_Y_AXIS, bool WRAP_Z_AXIS>
class KernelWrapper<2, WRAP_X_AXIS, WRAP_Y_AXIS, WRAP_Z_AXIS>
{
public:
    template<class CELL_TYPE>
    void operator()(dim3 cudaDimGrid, dim3 cudaDimBlock, CELL_TYPE *devGridOld, CELL_TYPE *devGridNew, int nanoStep, dim3 gridOffset, dim3 logicalGridDim, dim3 axisWrapOffset, int offsetY, int offsetZ, int wavefrontLength) const
    {
        kernel2D<WRAP_X_AXIS, WRAP_Y_AXIS, WRAP_Z_AXIS><<<cudaDimGrid, cudaDimBlock>>>(
            devGridOld, devGridNew, nanoStep, gridOffset, logicalGridDim, axisWrapOffset, offsetY, offsetZ, wavefrontLength);
    }
};

/**
 * See above
 */
template<bool WRAP_X_AXIS, bool WRAP_Y_AXIS, bool WRAP_Z_AXIS>
class KernelWrapper<3, WRAP_X_AXIS, WRAP_Y_AXIS, WRAP_Z_AXIS>
{
public:
    template<class CELL_TYPE>
    void operator()(dim3 cudaDimGrid, dim3 cudaDimBlock, CELL_TYPE *devGridOld, CELL_TYPE *devGridNew, int nanoStep, dim3 gridOffset, dim3 logicalGridDim, dim3 axisWrapOffset, int offsetY, int offsetZ, int wavefrontLength) const
    {
        kernel3D<WRAP_X_AXIS, WRAP_Y_AXIS, WRAP_Z_AXIS><<<cudaDimGrid, cudaDimBlock>>>(
            devGridOld, devGridNew, nanoStep, gridOffset, logicalGridDim, axisWrapOffset, offsetY, offsetZ, wavefrontLength);
    }
};

}

/**
 * CudaSimulator delegates all the work to the current Nvidia GPU.
 */
template<typename CELL_TYPE>
class CudaSimulator : public MonolithicSimulator<CELL_TYPE>
{
public:
    friend class CudaSimulatorTest;

    typedef typename MonolithicSimulator<CELL_TYPE>::Topology Topology;
    typedef DisplacedGrid<CELL_TYPE, Topology> GridType;

    static const int DIM = Topology::DIM;

    /**
     * creates a CudaSimulator with the given initializer. The
     * blockSize will heavily influence the performance and should be
     * chosen on a per GPU basis -- GPU architectures vary greatly.
     */
    explicit CudaSimulator(
        Initializer<CELL_TYPE> *initializer,
        // fixme: blockSize should be driven by an auto-tuner or at
        // least come with a better heuristic.
        Coord<3> blockSize = Coord<3>(128, (DIM < 3) ? 1 : 4, 1)) :
        MonolithicSimulator<CELL_TYPE>(initializer),
        blockSize(blockSize),
        ioGrid(&grid, CoordBox<DIM>()),
        hasCurrentGridOnHost(true)
    {
        stepNum = initializer->startStep();

        // to avoid conditionals within the kernel when accessing
        // neighboring cells at the grid's boundary, we'll simply pad
        // the grid on those faces where we don't use periodic
        // boundary conditions:
        Coord<DIM> offset;
        Coord<DIM> dim = initializer->gridBox().dimensions;
        for (int d = 0; d < DIM; ++d) {
            if (!Topology::wrapsAxis(d)) {
                offset[d] = -1;
                dim[d] += 2;
            }
        }

        grid = GridType(CoordBox<DIM>(offset, dim));
        byteSize = dim.prod() * sizeof(CELL_TYPE);
        cudaMalloc(&devGridOld, byteSize);
        cudaMalloc(&devGridNew, byteSize);

        CoordBox<DIM> box = grid.boundingBox();
        simArea << box;

        ioGrid = ProxyGrid<CELL_TYPE, DIM> (&grid, initializer->gridBox());
    }

    virtual ~CudaSimulator()
    {
        cudaFree(devGridOld);
        cudaFree(devGridNew);
    }

    /**
     * performs a single simulation step.
     */
    virtual void step()
    {
        // fixme: test steerer application, ensure grid gets copied to host and back to device
        // notify all registered Steerers
        for(unsigned i = 0; i < steerers.size(); ++i) {
            if (stepNum % steerers[i]->getPeriod() == 0) {
                steerers[i]->nextStep(&ioGrid, simArea, simArea.boundingBox().dimensions, stepNum, STEERER_NEXT_STEP, 0, true, 0);
            }
        }

        for (unsigned i = 0; i < APITraits::SelectNanoSteps<CELL_TYPE>::VALUE; ++i) {
            nanoStep(i);
            std::swap(devGridOld, devGridNew);
        }

        ++stepNum;

        bool lastStep = (stepNum == initializer->maxSteps());
        WriterEvent event = lastStep ? WRITER_ALL_DONE : WRITER_STEP_FINISHED;

        // call back all registered Writers
        for(unsigned i = 0; i < writers.size(); ++i) {
            bool writerRequestsWakeup = (stepNum % writers[i]->getPeriod() == 0);

            if (writerRequestsWakeup || lastStep) {
                writers[i]->stepFinished(
                    *getGrid(),
                    getStep(),
                    event);
            }
        }
    }

    const typename Simulator<CELL_TYPE>::GridType *getGrid()
    {
        if (!hasCurrentGridOnHost) {
            cudaMemcpy(grid.baseAddress(), devGridOld, byteSize, cudaMemcpyDeviceToHost);
            hasCurrentGridOnHost = true;
        }
        return &ioGrid;
    }

    /**
     * continue simulating until the maximum number of steps is reached.
     */
    virtual void run()
    {
        initializer->grid(&ioGrid);

        // pad boundaries, see c-tor
        Region<DIM> padding;
        padding << grid.boundingBox();
        padding >> initializer->gridBox();
        for (typename Region<DIM>::Iterator i = padding.begin(); i != padding.end(); ++i) {
            grid.set(*i, grid.getEdge());
        }

        cudaMemcpy(devGridOld, grid.baseAddress(), byteSize, cudaMemcpyHostToDevice);
        cudaMemcpy(devGridNew, grid.baseAddress(), byteSize, cudaMemcpyHostToDevice);
        stepNum = initializer->startStep();
        CUDAUtil::checkForError();

        for(unsigned i = 0; i < writers.size(); ++i) {
            writers[i]->stepFinished(
                ioGrid,
                getStep(),
                WRITER_INITIALIZED);
        }

        for (; stepNum < initializer->maxSteps();) {
            step();
        }
    }

private:
    Coord<3> blockSize;
    GridType grid;
    ProxyGrid<CELL_TYPE, DIM> ioGrid;
    CELL_TYPE *devGridOld;
    CELL_TYPE *devGridNew;
    int baseAddress;
    int byteSize;
    Region<DIM> simArea;
    bool hasCurrentGridOnHost;

    using MonolithicSimulator<CELL_TYPE>::initializer;
    using MonolithicSimulator<CELL_TYPE>::steerers;
    using MonolithicSimulator<CELL_TYPE>::stepNum;
    using MonolithicSimulator<CELL_TYPE>::writers;
    using MonolithicSimulator<CELL_TYPE>::getStep;

    void nanoStep(const unsigned nanoStep)
    {
        nanoStepImpl(nanoStep, typename APITraits::SelectSoA<CELL_TYPE>::Value());
        hasCurrentGridOnHost = false;
    }

    /**
     * We'll need a couple of parameters to describe the topology of
     * the problem space to the kernel. Here is a quick scetch. As an
     * example I'll use a 2D model with constant boundary conditions.
     * The grid will be padded on its boundary. Some parameters will
     * be used to calculate addresses within the memory allocated for
     * the grid:
     *
     * - offsetY/Z: how many cells separate two adjecent lines (y) or
     *   frames (z)?
     *
     * - axisWrapOffset: the same, but calculated for opposing sides
     *   of the grid (hence each is a multiple of the logicalGridDim
     *   and the corresponding offsetY/Z. "offsetX" is trivially
     *   assumed to be 1).
     *
     * - wavefrontLength will be used to determine how many cells each
     *   thread is tasked to traverse along the higest dimension of
     *   the simspace (Y axis for 2D, Z axis for 3D).
     *
     *                 gridOffset.x
     *                 -
     *  gridOffset.y | XXXXXXXXXXXXXX -
     *                 X$$$$$$$$$$$$X |      -
     *                 X$$$$$$$$$$$$X |      |
     *                 X$$$$$$$$$$$$X |      |
     *                 X$$$$$$$$$$$$X |      |
     *                 X$$$$$$$$$$$$X |      - logicalGridDim.y
     *                 XXXXXXXXXXXXXX - paddedGrid.y
     *
     *                    paddedGridDim.x
     *                 <------------>
     *
     *                    logicalGridDim.x
     *                  <---------->
     *
     *
     *
     * Legend
     * ------
     *
     * X: Boundary
     * $: Active grid content (i.e. cells which are going to be updated)
     */
    void nanoStepImpl(const unsigned nanoStep, APITraits::FalseType /* has no SoA */)
    {
        // fixme: measure time for this preprocessing via chronometer
        Coord<DIM> initGridDim = initializer->gridDimensions();

        Coord<DIM> cudaGridDim;
        // hack: treat 1D as special case of 2D, we don't need
        // wavefront traversal in this case:
        const int lastDim = (DIM == 1) ? 1 : DIM - 1;

        for (int d = 0; d < lastDim; ++d) {
            cudaGridDim[d] = ceil(1.0 * initGridDim[d] / blockSize[d]);
        }

        if (lastDim < DIM) {
            cudaGridDim[lastDim] = 1;
        }

        dim3 logicalGridDim = initGridDim;
        dim3 cudaDimBlock = blockSize;
        dim3 cudaDimGrid = cudaGridDim;

        Coord<3> rawOffset;
        for (int d = 0; d < DIM; ++d) {
            rawOffset[d] = grid.boundingBox().origin[d];
        }
        for (int d = DIM; d < 3; ++d) {
            rawOffset[d] = 0;
        }
        dim3 gridOffset = rawOffset;
        dim3 paddedGridDim = grid.boundingBox().dimensions;
        int offsetY = paddedGridDim.x;
        int offsetZ = paddedGridDim.x * paddedGridDim.y;

        dim3 axisWrapOffset;
        axisWrapOffset.x = logicalGridDim.x * 1;
        axisWrapOffset.y = logicalGridDim.y * offsetY;
        axisWrapOffset.z = logicalGridDim.z * offsetZ;

        int wavefrontLength = lastDim < DIM ? ceil(1.0 * initGridDim[lastDim] / blockSize[lastDim]) : 1;
        if (wavefrontLength == 0) {
            wavefrontLength = 1;
        }

        LOG(DBG, "CudaSimulator running kernel on grid size " << initGridDim
            << " with cudaDimGrid " << cudaDimGrid
            << " and cudaDimBlock " << cudaDimBlock
            << " and gridOffset " << gridOffset
            << " and logicalGridDim " << logicalGridDim
            << " and wavefrontLength " << wavefrontLength);

        CudaSimulatorHelpers::KernelWrapper<
            DIM,
            Topology::template WrapsAxis<0>::VALUE,
            Topology::template WrapsAxis<1>::VALUE,
            Topology::template WrapsAxis<2>::VALUE>()(
                cudaDimGrid,
                cudaDimBlock,
                devGridOld,
                devGridNew,
                nanoStep,
                gridOffset,
                logicalGridDim,
                axisWrapOffset,
                offsetY,
                offsetZ,
                wavefrontLength);
    }

    void nanoStepImpl(const unsigned nanoStep, APITraits::TrueType /* has SoA */)
    {
    }

};

}

#endif
