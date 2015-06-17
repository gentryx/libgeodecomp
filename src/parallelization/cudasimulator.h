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

// fixme: needs test, move to dedicated namespace?
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

template<class CELL_TYPE, bool WRAP_X_AXIS, bool WRAP_Y_AXIS, bool WRAP_Z_AXIS>
__global__
void kernel(CELL_TYPE *gridOld, CELL_TYPE *gridNew, int nanoStep, dim3 offset, dim3 logicalGridDim, dim3 paddedGridDim, int dimZ)
{
    // we need to distinguish logical coordinates and padded
    // coordinates: padded coords will be used to compute addresses
    // within the actual grid while logical coords correspond to a
    // cells position within the topology.
    int logicalX = blockIdx.x * blockDim.x + threadIdx.x;
    int logicalY = blockIdx.y * blockDim.y + threadIdx.y;
    int logicalZ = blockIdx.z * blockDim.z + threadIdx.z;

    int paddedX = logicalX - offset.x;
    int paddedY = logicalY - offset.y;
    int paddedMinZ = (logicalZ + 0) * dimZ - offset.z;
    int paddedMaxZ = (logicalZ + 1) * dimZ - offset.z;
    int paddedMaxZ2 = logicalGridDim.z - offset.z;
    if (paddedMaxZ2 < paddedMaxZ) {
        paddedMaxZ = paddedMaxZ2;
    }

    int offsetY = paddedGridDim.x;
    int offsetZ = paddedGridDim.x * paddedGridDim.y;

    int addWest   = WRAP_X_AXIS && (logicalX == 0                     ) ?  logicalGridDim.x : 0;
    int addEast   = WRAP_X_AXIS && (logicalX == (logicalGridDim.x - 1)) ? -logicalGridDim.x : 0;
    int addTop    = WRAP_Y_AXIS && (logicalY == 0                     ) ?  (logicalGridDim.y * offsetY) : 0;
    int addBottom = WRAP_Y_AXIS && (logicalY == (logicalGridDim.y - 1)) ? -(logicalGridDim.y * offsetY) : 0;
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
        addSouth = logicalGridDim.z * offsetZ;
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
        addNorth = -logicalGridDim.z * offsetZ;
        gridNew[index].update(hood, nanoStep);
    } else {
#pragma unroll
        for (int myZ = paddedMinZ; myZ < paddedMaxZ; ++myZ) {
            gridNew[index].update(hood, nanoStep);
            index += offsetZ;
        }
    }
}

/**
 * CudaSimulator delegates all the work to the current Nvidia GPU.
 */
template<typename CELL_TYPE>
class CudaSimulator : public MonolithicSimulator<CELL_TYPE>
{
public:
    friend class CudaSimulatorTest;
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    typedef DisplacedGrid<CELL_TYPE, Topology> GridType;
    static const int DIM = Topology::DIM;

    /**
     * creates a CudaSimulator with the given initializer. The
     * blockSize will heavily influence the performance and should be
     * chosen on a per GPU basis -- GPU architectures vary greatly.
     */
    CudaSimulator(
        Initializer<CELL_TYPE> *initializer,
        Coord<3> blockSize = Coord<3>(128, 4, 1)) :
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
            // fixme
            // std::cout << "nanoStep(" << stepNum << ", " << i << ") " << devGridOld << " -> " << devGridNew << "\n";
            nanoStep(i);
            std::swap(devGridOld, devGridNew);
            // fixme
            // {
            //     cudaMemcpy(grid.baseAddress(), devGridOld, byteSize, cudaMemcpyDeviceToHost);
            //     std::cout << "i: " << i << "\n";
            //     std::cout << "  ioGrid[0, 0,  0] cycle: " << ioGrid.get(Coord<DIM>(0, 0,  0)).cycleCounter <<  ", valid: " << ioGrid.get(Coord<DIM>(0, 0,  0)).isValid << ", testValue: " << ioGrid.get(Coord<DIM>(0, 0,  0)).testValue << "\n";
            //     std::cout << "  ioGrid[1, 1,  1] cycle: " << ioGrid.get(Coord<DIM>(1, 1,  1)).cycleCounter <<  ", valid: " << ioGrid.get(Coord<DIM>(1, 1,  1)).isValid << ", testValue: " << ioGrid.get(Coord<DIM>(1, 1,  1)).testValue << "\n";
            //     std::cout << "  ioGrid[1, 1,  8] cycle: " << ioGrid.get(Coord<DIM>(1, 1,  8)).cycleCounter <<  ", valid: " << ioGrid.get(Coord<DIM>(1, 1,  8)).isValid << ", testValue: " << ioGrid.get(Coord<DIM>(1, 1,  8)).testValue << "\n";
            //     std::cout << "  ioGrid[1, 1,  9] cycle: " << ioGrid.get(Coord<DIM>(1, 1,  9)).cycleCounter <<  ", valid: " << ioGrid.get(Coord<DIM>(1, 1,  9)).isValid << ", testValue: " << ioGrid.get(Coord<DIM>(1, 1,  9)).testValue << "\n";
            //     std::cout << "  ioGrid[1, 1, 10] cycle: " << ioGrid.get(Coord<DIM>(1, 1, 10)).cycleCounter <<  ", valid: " << ioGrid.get(Coord<DIM>(1, 1, 10)).isValid << ", testValue: " << ioGrid.get(Coord<DIM>(1, 1, 10)).testValue << "\n";
            // }
        }

        ++stepNum;

        cudaMemcpy(grid.baseAddress(), devGridOld, byteSize, cudaMemcpyDeviceToHost);

        // call back all registered Writers
        for(unsigned i = 0; i < writers.size(); ++i) {
            if (stepNum % writers[i]->getPeriod() == 0) {
                writers[i]->stepFinished(
                    *getGrid(),
                    getStep(),
                    WRITER_STEP_FINISHED);
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

        // fixme
        // std::cout << "ioGrid[0, 0,  0] " << ioGrid.get(Coord<DIM>(0, 0,  0)).cycleCounter << ", valid: " << ioGrid.get(Coord<DIM>(0, 0,  0)).isValid << ", edge: " << ioGrid.get(Coord<DIM>(0, 0,  0)).isEdgeCell << "\n";
        // std::cout << "ioGrid[1, 1,  1] " << ioGrid.get(Coord<DIM>(1, 1,  1)).cycleCounter << ", valid: " << ioGrid.get(Coord<DIM>(1, 1,  1)).isValid << ", edge: " << ioGrid.get(Coord<DIM>(0, 0,  1)).isEdgeCell << "\n";
        // std::cout << "ioGrid[1, 1,  9] " << ioGrid.get(Coord<DIM>(1, 1,  9)).cycleCounter << ", valid: " << ioGrid.get(Coord<DIM>(1, 1,  9)).isValid << ", edge: " << ioGrid.get(Coord<DIM>(0, 0,  9)).isEdgeCell << "\n";
        // std::cout << "ioGrid[1, 1, 10] " << ioGrid.get(Coord<DIM>(1, 1, 10)).cycleCounter << ", valid: " << ioGrid.get(Coord<DIM>(1, 1, 10)).isValid << ", edge: " << ioGrid.get(Coord<DIM>(0, 0, 10)).isEdgeCell << "\n";

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
            CUDAUtil::checkForError();
        }

        for(unsigned i = 0; i < writers.size(); ++i) {
            writers[i]->stepFinished(
                ioGrid,
                getStep(),
                WRITER_ALL_DONE);
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
        // fixme: rename vars:
        Coord<DIM> d = initializer->gridDimensions();
        dim3 dim(d.x(), d.y(), d.z());
        dim3 dimBlock(blockSize.x(), blockSize.y(), blockSize.z());
        dim3 dimGrid(
            ceil(1.0 * dim.x / dimBlock.x),
            ceil(1.0 * dim.y / dimBlock.y),
            1);
        int dimZ = dim.z / dimGrid.z;
        dim3 offset = grid.boundingBox().origin;
        dim3 paddedGridDim = grid.boundingBox().dimensions;

        LOG(DBG, "CudaSimulator running kernel on grid size " << d
            << " with dimGrid " << dimGrid
            << " and dimBlock " << dimBlock
            << " and offset " << offset
            << " and dim " << dim
            << " and dimZ " << dimZ);

        // fixme: check case when dimZ is smaller than effective grid size in z direction (i.e. two wavefronts traverse the grid)
        kernel<
            CELL_TYPE,
            Topology::template WrapsAxis<0>::VALUE,
            Topology::template WrapsAxis<1>::VALUE,
            Topology::template WrapsAxis<2>::VALUE><<<dimGrid, dimBlock>>>(devGridOld, devGridNew, nanoStep, offset, dim, paddedGridDim, dimZ);
    }
};

}

#endif
