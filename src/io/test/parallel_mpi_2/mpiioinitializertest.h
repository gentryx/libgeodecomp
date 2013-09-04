#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <unistd.h>
#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/mpiioinitializer.h>
#include <libgeodecomp/io/mpiiowriter.h>
#include <libgeodecomp/io/parallelmpiiowriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/misc/tempfile.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MPIIOInitializerTest : public CxxTest::TestSuite
{
public:
    SuperVector<std::string> files;
    int rank;

    void setUp()
    {
        rank = MPILayer().rank();
    }

    void tearDown()
    {
        if (rank == 0) {
            for (std::size_t i = 0; i < files.size(); ++i) {
                boost::filesystem::remove(files[i]);
            }
        }
    }

    void testBasic()
    {
        files << "testmpiioinitializer1_00000.mpiio"
              << "testmpiioinitializer1_00004.mpiio"
              << "testmpiioinitializer1_00008.mpiio"
              << "testmpiioinitializer1_00012.mpiio"
              << "testmpiioinitializer1_00016.mpiio"
              << "testmpiioinitializer1_00020.mpiio"
              << "testmpiioinitializer1_00021.mpiio";
        files << "testmpiioinitializer2_00008.mpiio"
              << "testmpiioinitializer2_00012.mpiio"
              << "testmpiioinitializer2_00016.mpiio"
              << "testmpiioinitializer2_00020.mpiio"
              << "testmpiioinitializer2_00021.mpiio";

        std::string snapshotFile = "testmpiioinitializer1_00008.mpiio";
        Coord<3> dimensions;

        if (rank == 0) {
            TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >();
            dimensions = init->gridDimensions();
            SerialSimulator<TestCell<3> > referenceSim(init);
            referenceSim.addWriter(
                new MPIIOWriter<TestCell<3> >(
                    "testmpiioinitializer1_",
                    4,
                    init->maxSteps(),
                    MPI::COMM_SELF));

            referenceSim.run();
        }

        MPILayer().barrier();

        LoadBalancer *balancer = MPILayer().rank()? 0 : new RandomBalancer;
        MPIIOInitializer<TestCell<3> > *init =
            new MPIIOInitializer<TestCell<3> >(snapshotFile);
        StripingSimulator<TestCell<3> > sim(init, balancer);
        sim.addWriter(
            new ParallelMPIIOWriter<TestCell<3> >(
                "testmpiioinitializer2_",
                4,
                init->maxSteps(),
                MPI::COMM_SELF));

        sim.run();
        MPILayer().barrier();

        if (rank == 0) {
            typedef APITraits::SelectTopology<TestCell<3> >::Value Topology;
            Grid<TestCell<3>, Topology> expected(dimensions);
            Grid<TestCell<3>, Topology> actual(dimensions);

            Region<3> region;
            region << CoordBox<3>(Coord<3>(), dimensions);
            MPIIO<TestCell<3> >::readRegion(
                &expected, "testmpiioinitializer1_00021.mpiio", region, MPI::COMM_SELF);
            MPIIO<TestCell<3> >::readRegion(
                &actual,   "testmpiioinitializer2_00021.mpiio", region, MPI::COMM_SELF);

            TS_ASSERT_EQUALS(expected, actual);
        }
    }
};

}
