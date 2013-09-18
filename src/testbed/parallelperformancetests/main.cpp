#include <mpi.h>
#include <libgeodecomp/io/collectingwriter.h>
#include <libgeodecomp/io/memorywriter.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchlink.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/testbed/performancetests/cpubenchmark.h>
#include <libgeodecomp/testbed/performancetests/cpubenchmark.h>
#include <libgeodecomp/testbed/performancetests/evaluate.h>
#include <libgeodecomp/testbed/parallelperformancetests/mysimplecell.h>

using namespace LibGeoDecomp;

std::string revision;

template<typename CELL_TYPE>
class CollectingWriterPerfTest : public CPUBenchmark
{
public:
    CollectingWriterPerfTest(const std::string& modelName) :
        modelName(modelName)
    {}

    std::string family()
    {
        return "CollectingWriter<" + modelName + ">";
    }
    std::string species()
    {
        return "gold";
    }

    double performance(const Coord<3>& dim)
    {
        MPILayer mpiLayer;

        Writer<CELL_TYPE> *cargoWriter = 0;
        if (mpiLayer.rank() == 0) {
            cargoWriter = new MemoryWriter<CELL_TYPE>(1);
        }
        CollectingWriter<CELL_TYPE> writer(cargoWriter, 1, 0);

        typedef typename CollectingWriter<CELL_TYPE>::SimulatorGridType SimulatorGridType;
        typedef typename CollectingWriter<CELL_TYPE>::StorageGridType StorageGridType;

        StorageGridType grid(CoordBox<3>(Coord<3>(), dim));

        // determine regions so that rank 0 gets the first half of the
        // grid and rank 1 the second half:
        CoordBox<3> regionBox(Coord<3>(), dim);
        int zOffsetStart = (mpiLayer.rank() + 0) * dim.z() / 2;
        int zOffsetEnd =   (mpiLayer.rank() + 1) * dim.z() / 2;
        int zDim = zOffsetEnd - zOffsetStart;
        regionBox.origin.z() = zOffsetStart;
        regionBox.dimensions.z() = zDim;

        Region<3> region;
        region << regionBox;

        long long tStart = Chronometer::timeUSec();
        for (int i = 0; i < repeats(); ++i) {
            writer.stepFinished(grid, region, dim, 0, WRITER_INITIALIZED, mpiLayer.rank(), true);
        }
        long long tEnd = Chronometer::timeUSec();


        // fixme: add test for cell with SoA
        return gigaBytesPerSecond(dim, seconds(tStart, tEnd));
    }

    std::string unit()
    {
        return "GB/s";
    }

private:
    std::string modelName;

    double gigaBytesPerSecond(const Coord<3>& dim, double seconds)
    {
        // multiply by 2 because all parts of the grid get sent AND received
        return 2.0 * dim.prod() * repeats() * sizeof(CELL_TYPE) * 1e-9 / seconds;
    }

    int repeats()
    {
        return 10;
    }
};

template<typename CELL_TYPE>
class PatchLinkPerfTest : public CPUBenchmark
{
public:
    PatchLinkPerfTest(const std::string& modelName) :
        modelName(modelName)
    {}

    std::string family()
    {
        return "PatchLink<" + modelName + ">";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(const Coord<3>& dim)
    {
        MPILayer mpiLayer;

        typedef typename HiParSimulator::Stepper<CELL_TYPE>::GridType GridType;

        CoordBox<3> gridBox(Coord<3>(), dim);
        GridType grid(gridBox, CELL_TYPE(), CELL_TYPE(), dim);
        Coord<3> offset(10, 10, 10);
        CoordBox<3> transmissionBox(offset, dim - offset * 2);
        Region<3> transmissionRegion;
        transmissionRegion << transmissionBox;
        Region<3> wholeGridRegion;
        wholeGridRegion << gridBox;
        int repeats = 0;
        long long tStart = 0;
        long long tEnd = 0;
        int maxNanoStep = 201234;

        if (mpiLayer.rank() == 0) {
            typename HiParSimulator::PatchLink<GridType>::Provider provider(
                transmissionRegion,
                1,
                666,
                Typemaps::lookup<CELL_TYPE>());
            provider.charge(1234, maxNanoStep, 1000);

            tStart = Chronometer::timeUSec();

            for (int i = 1234; i <= maxNanoStep; i += 1000) {
                provider.get(&grid, wholeGridRegion, i, true);
                ++repeats;
            }

            tEnd = Chronometer::timeUSec();

        } else {
            typename HiParSimulator::PatchLink<GridType>::Accepter accepter(
                transmissionRegion,
                0,
                666,
                Typemaps::lookup<CELL_TYPE>());
            accepter.charge(1234, maxNanoStep, 1000);

            for (int i = 1234; i <= maxNanoStep; i += 1000) {
                accepter.put(grid, wholeGridRegion, i);
            }
        }

        // fixme: add test for cell with SoA
        return gigaBytesPerSecond(transmissionBox.dimensions, repeats, seconds(tStart, tEnd));
    }

    std::string unit()
    {
        return "GB/s";
    }

private:
    std::string modelName;

    double gigaBytesPerSecond(const Coord<3>& dim, int repeats, double seconds)
    {
        // multiply by 2 because the whole transmissionBox is read and written
        return 2.0 * dim.prod() * repeats * sizeof(CELL_TYPE) * 1e-9 / seconds;
    }

};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    Typemaps::initializeMaps();

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

    Evaluate eval;

    bool output = MPILayer().rank() == 0;
    if (output) {
        eval.printHeader();
    }

    eval(CollectingWriterPerfTest<MySimpleCell>("MySimpleCell"), Coord<3>::diagonal(256), output);
    eval(CollectingWriterPerfTest<TestCell<3> >("TestCell<3> "), Coord<3>::diagonal(64), output);
    eval(PatchLinkPerfTest<MySimpleCell>("MySimpleCell"), Coord<3>::diagonal(200), output);
    eval(PatchLinkPerfTest<TestCell<3> >("TestCell<3> "), Coord<3>::diagonal(64), output);

    MPI_Finalize();
    return 0;
}
