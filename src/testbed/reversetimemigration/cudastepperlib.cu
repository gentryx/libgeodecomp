#include <libgeodecomp/testbed/reversetimemigration/cudastepperlib.h>
#include <libgeodecomp/testbed/reversetimemigration/cudastepper.h>

using namespace LibGeoDecomp;
using namespace HiParSimulator;

class RTMCell
{
public:
    typedef Topologies::Cube<3>::Topology Topology;

};

void CudaStepperLib::doit(const int& deviceID)
{
    std::cout << "setting device " << deviceID << "\n";
    cudaSetDevice(deviceID);
    CUDAStepper<RTMCell> stepper;
}
