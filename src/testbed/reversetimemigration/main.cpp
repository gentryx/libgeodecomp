#include <emmintrin.h>
#include <mpi.h>
#include <libgeodecomp/misc/supermap.h>
#include <libgeodecomp/parallelization/hiparsimulator.h>
#include <libgeodecomp/testbed/reversetimemigration/cudastepperlib.h>

using namespace LibGeoDecomp;
using namespace HiParSimulator;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    Typemaps::initializeMaps();
    MPILayer layer;

    int length;
    char hostname[MPI::MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &length);

    std::cout << "hit it!\n"
              << hostname << "\n";

    char *names = new char[layer.size() * MPI::MAX_PROCESSOR_NAME];
    layer.allGather(hostname, names, MPI::MAX_PROCESSOR_NAME);
    SuperMap<std::string, int> hostCount;
    SuperMap<int, int> cudaIDs;

    if (layer.rank() == 0) {
        for (int i = 0; i < layer.size(); ++i) {
            std::string name(names + i * MPI::MAX_PROCESSOR_NAME);

            int id = hostCount[name];
            hostCount[name]++;
            cudaIDs[i] = id;

            std::cout << "names[" << i << "] = " << name << " ID = " << id << "\n";
        }
    }

    int myDeviceID = cudaIDs[layer.rank()];
    // fixme:
    myDeviceID = 2;

    CudaStepperLib l;
    l.doit(myDeviceID);

    delete names;
    MPI_Finalize();
    return 0;
}
