#include <mpi.h>
#include <libgeodecomp.h>
#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/communication/patchlink.h>
#include <libgeodecomp/geometry/partitions/hilbertpartition.h>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/geometry/partitionmanager.h>
#include <libgeodecomp/io/collectingwriter.h>
#include <libgeodecomp/io/memorywriter.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/misc/cpubenchmark.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/parallelization/nesting/stepper.h>
#include <libflatarray/testbed/cpu_benchmark.hpp>
#include <libflatarray/testbed/evaluate.hpp>

#include "mysimplecell.h"

using namespace LibGeoDecomp;

int cudaDevice;

template<typename CELL_TYPE>
class CollectingWriterPerfTest : public CPUBenchmark
{
public:
    explicit CollectingWriterPerfTest(const std::string& modelName, const std::string& speciesName) :
        modelName(modelName),
        speciesName(speciesName)
    {}

    std::string family()
    {
        return "CollectingWriter<" + modelName + ">";
    }
    std::string species()
    {
        return speciesName;
    }

    double performance(std::vector<int> rawDim)
    {
        MPILayer mpiLayer;
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);

        Writer<CELL_TYPE> *cargoWriter = 0;
        if (mpiLayer.rank() == 0) {
            cargoWriter = new MemoryWriter<CELL_TYPE>(1);
        }
        CollectingWriter<CELL_TYPE> writer(cargoWriter, 0);

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

        double seconds = 0;

        {
            ScopedTimer t(&seconds);
            for (int i = 0; i < repeats(); ++i) {
                writer.stepFinished(grid, region, dim, 0, WRITER_INITIALIZED, mpiLayer.rank(), true);
            }
        }

        return gigaBytesPerSecond(dim, seconds);
    }

    std::string unit()
    {
        return "GB/s";
    }

private:
    std::string modelName;
    std::string speciesName;

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
    explicit PatchLinkPerfTest(const std::string& modelName, const std::string& speciesName) :
        modelName(modelName),
        speciesName(speciesName)
    {}

    std::string family()
    {
        return "PatchLink<" + modelName + ">";
    }

    std::string species()
    {
        return speciesName;
    }

    double performance(std::vector<int> rawDim)
    {
        MPILayer mpiLayer;
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);

        typedef typename Stepper<CELL_TYPE>::GridType GridType;

        CoordBox<3> gridBox(Coord<3>(), dim);
        GridType grid(gridBox, CELL_TYPE(), CELL_TYPE(), dim);
        Coord<3> offset(10, 10, 10);
        CoordBox<3> transmissionBox(offset, dim - offset * 2);
        Region<3> transmissionRegion;
        transmissionRegion << transmissionBox;
        Region<3> wholeGridRegion;
        wholeGridRegion << gridBox;
        int repeats = 0;
        int maxNanoStep = 201234;
        double seconds = 0;

        if (mpiLayer.rank() == 0) {
            typename PatchLink<GridType>::Provider provider(
                transmissionRegion,
                1,
                666,
                SerializationBuffer<CELL_TYPE>::cellMPIDataType());
            provider.charge(1234, maxNanoStep, 1000);

            {
                ScopedTimer t(&seconds);

                for (int i = 1234; i <= maxNanoStep; i += 1000) {
                    provider.get(&grid, wholeGridRegion, dim, i, 0, true);
                    ++repeats;
                }
            }
        } else {
            typename PatchLink<GridType>::Accepter accepter(
                transmissionRegion,
                0,
                666,
                SerializationBuffer<CELL_TYPE>::cellMPIDataType());
            accepter.charge(1234, maxNanoStep, 1000);

            for (int i = 1234; i <= maxNanoStep; i += 1000) {
                accepter.put(grid, wholeGridRegion, dim, i, 0);
            }
        }

        return gigaBytesPerSecond(transmissionBox.dimensions, repeats, seconds);
    }

    std::string unit()
    {
        return "GB/s";
    }

private:
    std::string modelName;
    std::string speciesName;

    double gigaBytesPerSecond(const Coord<3>& dim, int repeats, double seconds)
    {
        // multiply by 2 because the whole transmissionBox is read and written
        return 2.0 * dim.prod() * repeats * sizeof(CELL_TYPE) * 1e-9 / seconds;
    }

};

template<typename PARTITION>
class PartitionManagerBig3DPerfTest : public CPUBenchmark
{
public:
    explicit PartitionManagerBig3DPerfTest(const std::string& partitionName) :
        partitionName(partitionName)
    {}

    std::string family()
    {
        return "PartMngr3D<" + partitionName + ">";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(std::vector<int> rawDim)
    {
        MPILayer mpiLayer;
        Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            int ghostZoneWidth = 3;
            CoordBox<3> box(Coord<3>(), Coord<3>(2 * dim.x(), dim.y(), dim.z()));
            std::vector<std::size_t> weights;
            weights << dim.prod()
                    << dim.prod();

            typename SharedPtr<PARTITION>::Type partition(new PARTITION(Coord<3>(), box.dimensions, 0, weights));

            PartitionManager<Topologies::Torus<3>::Topology> myPartitionManager;
            typename SharedPtr<AdjacencyManufacturer<3> >::Type dummyAdjacencyManufacturer(new DummyAdjacencyManufacturer<3>);

            myPartitionManager.resetRegions(
                dummyAdjacencyManufacturer,
                box,
                partition,
                0,
                ghostZoneWidth);

            std::vector<CoordBox<3> > boundingBoxes;
            std::vector<CoordBox<3> > expandedBoundingBoxes;

            for (int i = 0; i < 2; ++i) {
                boundingBoxes << myPartitionManager.getRegion(i, 0).boundingBox();
            }

            for (int i = 0; i < 2; ++i) {
                expandedBoundingBoxes << myPartitionManager.getRegion(i, ghostZoneWidth).boundingBox();
            }

            myPartitionManager.resetGhostZones(boundingBoxes, expandedBoundingBoxes);

            for (int i = 0; i < 2; ++i) {
                if (myPartitionManager.getRegion(i, 0).boundingBox() == CoordBox<3>()) {
                    throw std::runtime_error("test failed: empty bounding box!");
                }
            }
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }

private:
    std::string partitionName;
};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    if ((argc < 3) || (argc == 4) || (argc > 5)) {
        std::cerr << "usage: " << argv[0] << " [-n,--name SUBSTRING] REVISION CUDA_DEVICE \n"
                  << "  - optional: only run tests whose name contains a SUBSTRING,\n"
                  << "  - REVISION is purely for output reasons,\n"
                  << "  - CUDA_DEVICE causes CUDA tests to run on the device with the given ID.\n";
        return 1;
    }
    std::string name = "";
    int argumentIndex = 1;
    if (argc == 5) {
        if ((std::string(argv[1]) == "-n") ||
            (std::string(argv[1]) == "--name")) {
            name = std::string(argv[2]);
        }
        argumentIndex = 3;
    }
    std::string revision = argv[argumentIndex + 0];

    std::stringstream s;
    s << argv[argumentIndex + 1];
    int cudaDevice;
    s >> cudaDevice;

    LibFlatArray::evaluate eval(name, revision);

    bool output = MPILayer().rank() == 0;
    if (output) {
        eval.print_header();
    }

    std::vector<int> diag64  = toVector(Coord<3>::diagonal(64));
    std::vector<int> diag100 = toVector(Coord<3>::diagonal(100));
    std::vector<int> diag200 = toVector(Coord<3>::diagonal(200));
    std::vector<int> diag256 = toVector(Coord<3>::diagonal(256));

    eval(CollectingWriterPerfTest<MySimpleCell>("MySimpleCell", "gold"),                       diag256, output);
    eval(CollectingWriterPerfTest<MySimpleCellSoA>("MySimpleCell", "platinum"),                diag256, output);

    eval(CollectingWriterPerfTest<TestCell<3> >("TestCell<3> ", "gold"),                       diag64,  output);
    eval(CollectingWriterPerfTest<TestCellSoA>( "TestCell<3> ", "platinum"),                   diag64,  output);

    eval(PatchLinkPerfTest<MySimpleCell>("MySimpleCell", "gold"),                              diag200, output);
    eval(PatchLinkPerfTest<MySimpleCellSoA>("MySimpleCell", "platinum"),                       diag200, output);

    eval(PatchLinkPerfTest<TestCell<3> >("TestCell<3> ", "gold"),                              diag64,  output);
    eval(PatchLinkPerfTest<TestCellSoA>( "TestCell<3> ", "platinum"),                          diag64,  output);

    eval(PartitionManagerBig3DPerfTest<RecursiveBisectionPartition<3> >("RecursiveBisection"), diag100, output);
    eval(PartitionManagerBig3DPerfTest<ZCurvePartition<3> >("ZCurve"),                         diag100, output);

    MPI_Finalize();
    return 0;
}
