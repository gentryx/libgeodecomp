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

template<class CELL_TYPE>
__global__
void kernel(CELL_TYPE *gridOld, CELL_TYPE *gridNew, dim3 gridDim, int dimZ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetY = gridDim.x;
    int offsetZ = gridDim.x * gridDim.y;

    int addWest = 0;
    int addEast = 0;
    int addTop = 0;
    int addBottom = 0;
    int addSouth = 0;
    int addNorth = 0;

    int minZ = (z + 0) * dimZ;
    int maxZ = (z + 1) * dimZ;
    int index =
        z * gridDim.y * gridDim.x +
        y * gridDim.y +
        x;

    if ((x > gridDim.x) || (y > gridDim.y)) {
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

    for (int myZ = (minZ + 1); myZ < (maxZ - 1); ++myZ) {
        index += gridDim.x * gridDim.y;
        gridNew[index].update(hood, 0);
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
            nanoStep(i);
            std::swap(devGridOld, devGridNew);
        }

        ++stepNum;

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
            CUDAUtil::checkForError();
        }

        for(unsigned i = 0; i < writers.size(); ++i) {
            writers[i]->stepFinished(
                ioGrid,
                getStep(),
                WRITER_ALL_DONE);
        }
    }

    virtual const GridType *getGrid()
    {
        cudaMemcpy(grid.baseAddress(), devGridOld, byteSize, cudaMemcpyDeviceToHost);
        return &grid;
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
        Coord<DIM> d = initializer->gridDimensions();
        dim3 dim(d.x(), d.y(), d.z());
        dim3 dimBlock(blockSize.x(), blockSize.y(), blockSize.z());
        dim3 dimGrid(
            ceil(1.0 * dim.x / dimBlock.x),
            ceil(1.0 * dim.y / dimBlock.y),
            1);
        int dimZ = dim.z / dimGrid.z;

        LOG(DBG, "CudaSimulator running kernel on grid size " << d
            << " with dimGrid " << dimGrid
            << " and dimBlock " << dimBlock);

        kernel<CELL_TYPE><<<dimGrid, dimBlock>>>(devGridOld, devGridNew, dim, dimZ);
    }
};

}

#endif
