#include <mpi.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/testbed/performancetests/benchmark.h>
#include <libgeodecomp/testbed/performancetests/cpubenchmark.h>
#include <libgeodecomp/testbed/performancetests/evaluate.h>

using namespace LibGeoDecomp;

std::string revision;

class CollectingWriterGridCollection : public CPUBenchmark
{
public:
    std::string family()
    {
        return "CollectingWriterGridCollection";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(const Coord<3>& dim)
    {
        // fixme: implement me
        return 0;
    }

    std::string unit()
    {
        return "s";
    }
};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    if (MPILayer().size() != 2) {
        std::cerr << "Please run with two MPI processes\n";
        return 1;
    }

    if ((argc < 3) || (argc > 4)) {
        std::cerr << "usage: " << argv[0] << "[-q,--quick] REVISION CUDA_DEVICE\n";
        return 1;
    }

    bool quick = false;
    int argumentIndex = 1;
    if (argc == 4) {
        if ((std::string(argv[1]) == "-q") ||
            (std::string(argv[1]) == "--quick")) {
            quick = true;
        }
        argumentIndex = 2;
    }
    revision = argv[argumentIndex];

    std::cout << "#rev              ; date                 ; host            ; device                                          ; order   ; family          ; species ; dimensions              ; perf        ; unit\n";

    evaluate(CollectingWriterGridCollection(), Coord<3>( 128,  128,  128));

    MPI_Finalize();
    return 0;
}
