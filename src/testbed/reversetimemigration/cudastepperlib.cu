#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/testbed/reversetimemigration/cudastepperlib.h>
#include <libgeodecomp/testbed/reversetimemigration/cudastepper.h>

using namespace LibGeoDecomp;
using namespace HiParSimulator;

class RTMCell
{
public:
    class API : public APITraits::HasCubeTopology<3>
    {};

    static int flops()
    {
        return 50;
    }

};

void CudaStepperLib::doit(const int& deviceID)
{
    std::cout << "setting device " << deviceID << "\n";
    cudaSetDevice(deviceID);
    CUDAStepper<RTMCell> stepper;

    long long timeStart = ScopedTimer::timeUSec();

    int repeats = 100;
    for (int i = 0; i < repeats; ++i) {
        stepper.step();
    }

    cudaThreadSynchronize();
    long long timeEnd = ScopedTimer::timeUSec();
    double updates = 1.0 * repeats * (DIM_Z - 4)* GRID_DIM_X * GRID_DIM_Y * BLOCK_DIM_X * BLOCK_DIM_Y;
    double time = (timeEnd - timeStart) * 0.0000001;
    double glups = updates / time * 0.0000000001;
    double gflops = glups * RTMCell::flops();

    std::cout << "GLUPS:  " << glups << "\n";
    std::cout << "GFLOPS: " << gflops << "\n";

    CUDAStepper<RTMCell>::checkForCUDAError();


}
