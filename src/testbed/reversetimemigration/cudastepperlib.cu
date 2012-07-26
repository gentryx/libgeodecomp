#include <libgeodecomp/testbed/reversetimemigration/cudastepperlib.h>
#include <libgeodecomp/testbed/reversetimemigration/cudastepper.h>

using namespace LibGeoDecomp;
using namespace HiParSimulator;

void CudaStepperLib::doit(const int& deviceID)
{
    cudaSetDevice(deviceID);
    CUDAStepper<double> stepper;
}
