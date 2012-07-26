#ifndef _libgeodecomp_parallelization_hiparsimulator_cudastepper_fixme_h_
#define _libgeodecomp_parallelization_hiparsimulator_cudastepper_fixme_h_

#include <cuda.h>
 
#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchbufferfixed.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/testbed/reversetimemigration/config.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

__global__ void update(double *gridOld, double *gridNew)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    for (int z = 0; z < DIM_Z; ++z) {
        int offset = z * DIM_X * DIM_Y + y * DIM_X + x;
        gridNew[offset] = gridOld[offset];
    }
}

template<typename CELL_TYPE>
class CUDAStepper
// class CUDAStepper : public Stepper<CELL_TYPE>
{
public:
    typedef typename Stepper<CELL_TYPE>::GridType GridType;

    inline CUDAStepper(// const GridType& sourceGrid
                       )
    {
        std::cout << "allocing\n";
        long bytesize = long(sizeof(double)) * DIM_X * DIM_Y * DIM_Z;
        cudaMalloc(&devGridOld, bytesize);
        cudaMalloc(&devGridNew, bytesize);
        checkForCUDAError();

        sleep(10);
    }
    
    inline void step()
    {
        dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
        dim3 dimGrid(GRID_DIM_X, GRID_DIM_Y);
        update<<<dimGrid, dimBlock>>>(devGridOld, devGridNew);
    }

    ~CUDAStepper()
    {
        std::cout << "freeing\n";
        cudaFree(devGridOld);
        cudaFree(devGridNew);
    }

private:
    double *devGridOld;
    double *devGridNew;

    // fixme: move to utility class
    void checkForCUDAError()
    {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "ERROR: " << cudaGetErrorString(error) << "\n";
            throw std::runtime_error("CUDA error");
        }
    }
};

}
}

#endif
