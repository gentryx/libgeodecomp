#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_CUDASTEPPER_FIXME_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_CUDASTEPPER_FIXME_H

#include <cuda.h>

#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchbufferfixed.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/testbed/reversetimemigration/config.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

#define OFFSET(X, Y, Z) (Z * DIM_X * DIM_Y + Y * DIM_X + X)

__global__ void update(double *gridOld, double *gridNew)
{
    int x = blockIdx.x * BLOCK_DIM_X + threadIdx.x + 2;
    int y = (blockIdx.y * BLOCK_DIM_Y + threadIdx.y) * 2 + 2;
    int offset = y * DIM_X + x;

    __shared__ double buf[BLOCK_DIM_Y][BLOCK_DIM_X];

    double lineA0 = gridOld[offset + OFFSET(0, 0, 0)];
    double lineA1 = gridOld[offset + OFFSET(0, 0, 1)];
    double lineA2 = gridOld[offset + OFFSET(0, 0, 2)];
    double lineA3 = gridOld[offset + OFFSET(0, 0, 3)];
    double lineB0 = gridOld[offset + OFFSET(0, 1, 0)];
    double lineB1 = gridOld[offset + OFFSET(0, 1, 1)];
    double lineB2 = gridOld[offset + OFFSET(0, 1, 2)];
    double lineB3 = gridOld[offset + OFFSET(0, 1, 3)];

    for (int z = 2; z < (DIM_Z - 2); ++z) {
        double lineA4 = gridOld[offset + OFFSET(0, 0, 2)];
        double lineB4 = gridOld[offset + OFFSET(0, 1, 2)];

        double bottom = gridOld[offset + OFFSET(0,  2, 0)];
        double top    = gridOld[offset + OFFSET(0, -1, 0)];

        gridNew[offset + OFFSET(0, 0, 0)] =
            0.01 * lineA0 +
            0.02 * lineA1 +
            0.03 * lineA2 +
            0.04 * lineA3 +
            0.05 * lineA4 +
            0.06 * top +
            0.07 * gridOld[offset + OFFSET(0, -2, 0)] +
            0.08 * lineB2 +
            0.09 * bottom +
            0.10 * gridOld[offset + OFFSET(-1, 0, 0)] +
            0.11 * gridOld[offset + OFFSET(-2, 0, 0)] +
            0.12 * gridOld[offset + OFFSET( 1, 0, 0)] +
            0.13 * gridOld[offset + OFFSET( 2, 0, 0)];
        gridNew[offset + OFFSET(0, 1, 0)] =
            0.01 * lineB0 +
            0.02 * lineB1 +
            0.03 * lineB2 +
            0.04 * lineB3 +
            0.05 * lineB4 +
            0.06 * lineA2 +
            0.07 * top +
            0.08 * bottom +
            0.09 * gridOld[offset + OFFSET(0, 3, 0)] +
            0.10 * gridOld[offset + OFFSET(-1, 1, 0)] +
            0.11 * gridOld[offset + OFFSET(-2, 1, 0)] +
            0.12 * gridOld[offset + OFFSET( 1, 1, 0)] +
            0.13 * gridOld[offset + OFFSET( 2, 1, 0)];


        offset += DIM_X * DIM_Y;
        lineA0 = lineA1;
        lineA1 = lineA2;
        lineA2 = lineA3;
        lineA3 = lineA4;
        lineB0 = lineB1;
        lineB1 = lineB2;
        lineB2 = lineB3;
        lineB3 = lineB4;
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
    }

    inline void step()
    {
        dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
        dim3 dimGrid(GRID_DIM_X, GRID_DIM_Y);
        update<<<dimGrid, dimBlock>>>(devGridOld, devGridNew);
        std::swap(devGridOld, devGridNew);
    }

    ~CUDAStepper()
    {
        std::cout << "freeing\n";
        cudaFree(devGridOld);
        cudaFree(devGridNew);
    }

    // fixme: move to utility class
    static void checkForCUDAError()
    {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "ERROR: " << cudaGetErrorString(error) << "\n";
            throw std::runtime_error("CUDA error");
        }
    }

private:
    double *devGridOld;
    double *devGridNew;
};

}
}

#endif
