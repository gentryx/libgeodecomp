#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/testbed/reversetimemigration/cudastepperlib.h>
#include <libgeodecomp/testbed/reversetimemigration/cudastepper.h>

using namespace LibGeoDecomp;

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

    double seconds = 0;
    int repeats = 100;

    {
        ScopedTimer t(&seconds);

        for (int i = 0; i < repeats; ++i) {
            stepper.step();
        }

        cudaThreadSynchronize();
    }

    double updates = 1.0 * repeats * (DIM_Z - 4)* GRID_DIM_X * GRID_DIM_Y * BLOCK_DIM_X * BLOCK_DIM_Y;
    double glups = updates / seconds * 0.0000000001;
    double gflops = glups * RTMCell::flops();

    std::cout << "GLUPS:  " << glups << "\n";
    std::cout << "GFLOPS: " << gflops << "\n";

    CUDAStepper<RTMCell>::checkForCUDAError();


}
