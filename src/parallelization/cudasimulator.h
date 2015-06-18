#ifndef LIBGEODECOMP_PARALLELIZATION_CUDASIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_CUDASIMULATOR_H

#include <libgeodecomp/geometry/fixedcoord.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/cudautil.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/proxygrid.h>

namespace LibGeoDecomp {

namespace CudaSimulatorHelpers {

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

        gridNew[index].update(hood, nanoStep);
        paddedMinY += 1;
        index += offsetY;
    }

    addTop    = 0;
    addBottom = 0;

    if (WRAP_Y_AXIS && (paddedMaxY == logicalGridDim.y)) {
#pragma unroll
        for (int myY = paddedMinY; myY < (paddedMaxY - 1); ++myY) {
            gridNew[index].update(hood, nanoStep);
            index += offsetY;
        }

        addTop = 0;
        addBottom = -axisWrapOffset.y;
        gridNew[index].update(hood, nanoStep);

    } else {
#pragma unroll
        for (int myY = paddedMinY; myY < paddedMaxY; ++myY) {
            gridNew[index].update(hood, nanoStep);
            index += offsetY;
        }
    }
}

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

        gridNew[index].update(hood, nanoStep);
        paddedMinZ += 1;
        index += offsetZ;
    }

    addSouth = 0;
    addNorth = 0;

    if (WRAP_Z_AXIS && (paddedMaxZ == logicalGridDim.z)) {
#pragma unroll
        for (int myZ = paddedMinZ; myZ < (paddedMaxZ - 1); ++myZ) {
            gridNew[index].update(hood, nanoStep);
            index += offsetZ;
        }

        addSouth = 0;
        addNorth = -axisWrapOffset.z;
        gridNew[index].update(hood, nanoStep);

    } else {
#pragma unroll
        for (int myZ = paddedMinZ; myZ < paddedMaxZ; ++myZ) {
            gridNew[index].update(hood, nanoStep);
            index += offsetZ;
        }
    }
}

template<int DIM, bool WRAP_X_AXIS, bool WRAP_Y_AXIS, bool WRAP_Z_AXIS>
class KernelWrapper;

template<bool WRAP_X_AXIS, bool WRAP_Y_AXIS, bool WRAP_Z_AXIS>
class KernelWrapper<2, WRAP_X_AXIS, WRAP_Y_AXIS, WRAP_Z_AXIS>
{
public:
    template<class CELL_TYPE>
    void operator()(dim3 dimGrid, dim3 dimBlock, CELL_TYPE *devGridOld, CELL_TYPE *devGridNew, int nanoStep, dim3 gridOffset, dim3 logicalGridDim, dim3 axisWrapOffset, int offsetY, int offsetZ, int wavefrontLength) const
    {
        kernel2D<WRAP_X_AXIS, WRAP_Y_AXIS, WRAP_Z_AXIS><<<dimGrid, dimBlock>>>(
            devGridOld, devGridNew, nanoStep, gridOffset, logicalGridDim, axisWrapOffset, offsetY, offsetZ, wavefrontLength);
    }
};

template<bool WRAP_X_AXIS, bool WRAP_Y_AXIS, bool WRAP_Z_AXIS>
class KernelWrapper<3, WRAP_X_AXIS, WRAP_Y_AXIS, WRAP_Z_AXIS>
{
public:
    template<class CELL_TYPE>
    void operator()(dim3 dimGrid, dim3 dimBlock, CELL_TYPE *devGridOld, CELL_TYPE *devGridNew, int nanoStep, dim3 gridOffset, dim3 logicalGridDim, dim3 axisWrapOffset, int offsetY, int offsetZ, int wavefrontLength) const
    {
        kernel3D<WRAP_X_AXIS, WRAP_Y_AXIS, WRAP_Z_AXIS><<<dimGrid, dimBlock>>>(
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
    CudaSimulator(
        Initializer<CELL_TYPE> *initializer,
        // blockSize depends on DIM as we want 1D blocks for our
        // wavefront algorithm. This is still very stupid and should
        // be driven by an auto-tuner or at least come with a better
        // heuristic.
        Coord<3> blockSize = Coord<3>(128, (DIM * 3 - 5), 1)) :
        MonolithicSimulator<CELL_TYPE>(initializer),
        blockSize(blockSize),
        ioGrid(&grid, CoordBox<DIM>())
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

    ~CudaSimulator()
    {
        cudaFree(devGridOld);
        cudaFree(devGridNew);
    }

    /**
     * performs a single simulation step.
     */
    virtual void step()
    {
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

        cudaMemcpy(grid.baseAddress(), devGridOld, byteSize, cudaMemcpyDeviceToHost);
        CUDAUtil::checkForError();

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
        // fixme: only copy back if required by writers
        cudaMemcpy(grid.baseAddress(), devGridOld, byteSize, cudaMemcpyDeviceToHost);
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

    using MonolithicSimulator<CELL_TYPE>::initializer;
    using MonolithicSimulator<CELL_TYPE>::steerers;
    using MonolithicSimulator<CELL_TYPE>::stepNum;
    using MonolithicSimulator<CELL_TYPE>::writers;
    using MonolithicSimulator<CELL_TYPE>::getStep;

    void nanoStep(const unsigned nanoStep)
    {
        // fixme: measure time for this preprocessing via chronometer
        // fixme: rename vars:
        Coord<DIM> initGridDim = initializer->gridDimensions();

        Coord<DIM> cudaGridDim;
        for (int d = 0; d < (DIM - 1); ++d) {
            cudaGridDim[d] = ceil(1.0 * initGridDim[d] / blockSize[d]);
        }
        // fixme: make the number of wavefronts configurable
        cudaGridDim[DIM - 1] = 1;

        dim3 logicalGridDim = initGridDim;
        dim3 dimBlock = blockSize;
        dim3 dimGrid = cudaGridDim;

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

        int wavefrontLength = initGridDim[DIM - 1] / cudaGridDim[DIM - 1];
        if (wavefrontLength == 0) {
            wavefrontLength = 1;
        }

        LOG(DBG, "CudaSimulator running kernel on grid size " << initGridDim
            << " with dimGrid " << dimGrid
            << " and dimBlock " << dimBlock
            << " and gridOffset " << gridOffset
            << " and logicalGridDim " << logicalGridDim
            << " and wavefrontLength " << wavefrontLength);

        // fixme: check case when dimZ is smaller than effective grid size in z direction (i.e. two wavefronts traverse the grid)

        CudaSimulatorHelpers::KernelWrapper<
            DIM,
            Topology::template WrapsAxis<0>::VALUE,
            Topology::template WrapsAxis<1>::VALUE,
            Topology::template WrapsAxis<2>::VALUE>()(
                dimGrid, dimBlock, devGridOld, devGridNew, nanoStep, gridOffset, logicalGridDim, axisWrapOffset, offsetY, offsetZ, wavefrontLength);
    }
};

}

#endif
