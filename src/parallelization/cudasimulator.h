#ifndef LIBGEODECOMP_PARALLELIZATION_CUDASIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_CUDASIMULATOR_H

#include <libgeodecomp/misc/cudautil.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/fixedcoord.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>

namespace LibGeoDecomp {

template<class CELL_TYPE>
class MyHood
{
public:
    __device__
    MyHood(int *index, dim3 *gridDim, CELL_TYPE *grid) :
        index(index),
        gridDim(gridDim),
        grid(grid)
    {}

    template<int X, int Y, int Z>
    __device__
    const CELL_TYPE& operator[](FixedCoord<X, Y, Z> coord) const
    {
        return grid[*index + (Z * gridDim->x * gridDim->y) + (Y * gridDim->x) + X];
    }
    
private:
    int *index;
    dim3 *gridDim;
    CELL_TYPE *grid;
};

template<class CELL_TYPE>
__global__
void kernel(CELL_TYPE *gridOld, CELL_TYPE *gridNew, dim3 gridDim, int dimZ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int minZ = (z + 0) * dimZ;
    int maxZ = (z + 1) * dimZ;
    int index = 
        z * gridDim.y * gridDim.x +
        y * gridDim.y + 
        x;

    if ((x > gridDim.x) || (y > gridDim.y)) {
        return;
    }

    MyHood<CELL_TYPE> hood(&index, &gridDim, gridOld);

    for (int myZ = (minZ + 1); myZ < (maxZ - 1); ++myZ) {
        index += gridDim.x * gridDim.y;
        gridNew[index].update(hood, 0);
        // gridNew[index] = gridOld[index];
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
    typedef typename CELL_TYPE::Topology Topology;
    typedef Grid<CELL_TYPE, Topology> GridType;
    static const int DIM = Topology::DIMENSIONS;

    /**
     * creates a CudaSimulator with the given @a initializer.
     */
    CudaSimulator(Initializer<CELL_TYPE> *initializer) : 
        MonolithicSimulator<CELL_TYPE>(initializer)
    {
        stepNum = initializer->startStep();
        Coord<DIM> dim = initializer->gridBox().dimensions;
        grid = GridType(dim);
        byteSize = dim.prod() * sizeof(CELL_TYPE);
        cudaMalloc(&devGridOld, byteSize);
        cudaMalloc(&devGridNew, byteSize);

        CoordBox<DIM> box = grid.boundingBox();
        unsigned endX = box.dimensions.x();
        box.dimensions.x() = 1;
        for(typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            simArea << Streak<DIM>(*i, endX);
        }
    }

    /**
     * performs a single simulation step.
     */
    virtual void step()
    {
        // notify all registered Steerers
        for(unsigned i = 0; i < steerers.size(); ++i) {
            if (stepNum % steerers[i]->getPeriod() == 0) {
                steerers[i]->nextStep(&grid, simArea, stepNum);
            }
        }

        for (unsigned i = 0; i < CELL_TYPE::nanoSteps(); ++i) {
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
        initializer->grid(&grid);
        cudaMemcpy(devGridOld, grid.baseAddress(), byteSize, cudaMemcpyHostToDevice);
        cudaMemcpy(devGridNew, grid.baseAddress(), byteSize, cudaMemcpyHostToDevice);
        stepNum = initializer->startStep();
        CUDAUtil::checkForError();

        for(unsigned i = 0; i < writers.size(); ++i) {
            writers[i]->stepFinished(
                *getGrid(),
                getStep(),
                WRITER_INITIALIZED);
        }

        for (; stepNum < initializer->maxSteps();) {
            step();
        }

        CUDAUtil::checkForError();
        for(unsigned i = 0; i < writers.size(); ++i) {
            writers[i]->stepFinished(
                *getGrid(),
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
    GridType grid;
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

    void nanoStep(const unsigned& nanoStep)
    {
        Coord<DIM> d = initializer->gridDimensions();
        dim3 dim(d.x(), d.y(), d.z());
        dim3 dimBlock(128, 4, 1);
        dim3 dimGrid(dim.x / dimBlock.x, dim.y / dimBlock.y, 1);
        int dimZ = dim.z / dimGrid.z;
        kernel<CELL_TYPE><<<dimGrid, dimBlock>>>(devGridOld, devGridNew, dim, dimZ);
    }
};

}

#endif
