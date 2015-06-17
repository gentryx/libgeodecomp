#include <cuda.h>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/testwriter.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/cudasimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CudaSimulatorTest : public CxxTest::TestSuite
{
public:
    typedef TestCell<3, Stencils::VonNeumann<3, 1>, Topologies::Cube<3>::Topology,
                     TestCellHelpers::EmptyAPI, TestCellHelpers::NoOutput> TestCell3dCube;
    typedef TestInitializer<TestCell3dCube> TestInitializer3dCube;

    void test3DCube()
    {
        std::cout << "----------------------------------------------\n";
        Coord<3> dim(50, 20, 10);
        int numSteps = 5;
        CudaSimulator<TestCell3dCube> sim(new TestInitializer3dCube(dim, numSteps));

        std::vector<int> expectedSteps;
        std::vector<WriterEvent> expectedEvents;

        expectedSteps << 0
                      << 1
                      << 2
                      << 3
                      << 4
                      << 5
                      << 5;
        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;
        sim.addWriter(new TestWriter<TestCell3dCube>(1, expectedSteps, expectedEvents));
        std::cout << "----------------------------------------------\n";
        sim.run();
        std::cout << "----------------------------------------------\n";
    }
};

}
