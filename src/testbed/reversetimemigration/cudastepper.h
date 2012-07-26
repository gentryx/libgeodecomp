#ifndef _libgeodecomp_parallelization_hiparsimulator_cudastepper_fixme_h_
#define _libgeodecomp_parallelization_hiparsimulator_cudastepper_fixme_h_

#include <cuda.h>

#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchbufferfixed.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<typename CELL_TYPE>
class CUDAStepper
// class CUDAStepper : public Stepper<CELL_TYPE>
{
public:
    inline CUDAStepper()
    {
        std::cout << "allocing\n";
        long bytesize = long(sizeof(double)) * 256 * 512 * 1024;
        cudaMalloc(&devGridOld, bytesize);
        cudaMalloc(&devGridNew, bytesize);
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
};

}
}

#endif
