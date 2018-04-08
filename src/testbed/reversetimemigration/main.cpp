#include <libgeodecomp/parallelization/hiparsimulator.h>
#include <map>
#include <mpi.h>

#include "config.h"

using namespace LibGeoDecomp;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPILayer layer;

    int length;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &length);

    std::cout << "hit it!\n"
              << hostname << "\n";

    char *names = new char[layer.size() * MPI_MAX_PROCESSOR_NAME];
    layer.allGather(hostname, names, MPI_MAX_PROCESSOR_NAME);
    std::map<std::string, int> hostCount;
    std::map<int, int> cudaIDs;

    for (int i = 0; i < layer.size(); ++i) {
        std::string name(names + i * MPI_MAX_PROCESSOR_NAME);

        int id = hostCount[name];
        hostCount[name]++;
        cudaIDs[i] = id;

        if (layer.rank() == 0) {
            std::cout << "names[" << i << "] = " << name << " ID = " << id << "\n";
        }
    }

    int myDeviceID = cudaIDs[layer.rank()];
    std::cout << "DIM_X = " << DIM_X << "\n";

    // fixme
    delete names;
    MPI_Finalize();
    return 0;
}

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
