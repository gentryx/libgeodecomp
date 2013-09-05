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

using namespace LibGeoDecomp;

std::string revision;

class MySimpleCell
{
public:
    class API :
        public APITraits::HasCubeTopology<3>,
        public APITraits::HasStencil<Stencils::Moore<3, 1> >
    {};

    double temp;
};

class CollectingWriterStepFinished1 : public CPUBenchmark
{
public:
    std::string family()
    {
        return "CollectingWriterStepFinished1";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(const Coord<3>& dim)
    {
        MPILayer mpiLayer;

        Writer<MySimpleCell> *cargoWriter = 0;
        if (mpiLayer.rank() == 0) {
            cargoWriter = new MemoryWriter<MySimpleCell>(1);
        }
        CollectingWriter<MySimpleCell> writer(cargoWriter, 1, 0, MPI_COMM_WORLD, MPI_DOUBLE);

        typedef CollectingWriter<MySimpleCell>::SimulatorGridType SimulatorGridType;
        typedef CollectingWriter<MySimpleCell>::StorageGridType StorageGridType;

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
        for (int i = 0; i < 10; ++i) {
            writer.stepFinished(grid, region, dim, 0, WRITER_INITIALIZED, mpiLayer.rank(), true);
        }
        long long tEnd = Chronometer::timeUSec();


        // fixme: implement me
        // fixme: add test for cell with SoA
        return seconds(tStart, tEnd);
    }

    // fixme: benchmark mpi functions

    std::string unit()
    {
        return "s";
    }
};

class PatchLink1 : public CPUBenchmark
{
public:
    std::string family()
    {
        return "PatchLink1";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(const Coord<3>& dim)
    {
        MPILayer mpiLayer;

        typedef HiParSimulator::Stepper<MySimpleCell>::GridType GridType;

        CoordBox<3> gridBox(Coord<3>(), dim);
        GridType grid(gridBox, MySimpleCell(), MySimpleCell(), dim);
        Coord<3> offset(10, 10, 10);
        CoordBox<3> transmissionBox(offset, dim - offset * 2);
        Region<3> transmissionRegion;
        transmissionRegion << transmissionBox;
        Region<3> wholeGridRegion;
        wholeGridRegion << gridBox;

        long long tStart = 0;
        long long tEnd = 0;

        if (mpiLayer.rank() == 0) {
            HiParSimulator::PatchLink<GridType>::Provider provider(
                transmissionRegion,
                1,
                666,
                MPI_DOUBLE);
            provider.charge(1234, 201234, 1000);

            tStart = Chronometer::timeUSec();

            for (int i = 1234; i <= 201234; i += 1000) {
                provider.get(&grid, wholeGridRegion, i, true);
            }

            tEnd = Chronometer::timeUSec();

        } else {
            HiParSimulator::PatchLink<GridType>::Accepter accepter(
                transmissionRegion,
                0,
                666,
                MPI_DOUBLE);
            accepter.charge(1234, 201234, 1000);

            for (int i = 1234; i <= 201234; i += 1000) {
                accepter.put(grid, wholeGridRegion, i);
            }
        }

        // // fixme: repeats? sizes?
        // // fixme: implement me
        // // fixme: add test for cell with SoA
        return seconds(tStart, tEnd);
    }

    // fixme: benchmark mpi functions

    std::string unit()
    {
        return "s";
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

    eval(CollectingWriterStepFinished1(), Coord<3>::diagonal(256), output);
    eval(PatchLink1(), Coord<3>::diagonal(256), output);

    MPI_Finalize();
    return 0;
}
