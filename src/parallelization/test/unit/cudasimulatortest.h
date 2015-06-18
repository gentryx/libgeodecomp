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
    typedef TestCell<2, Stencils::VonNeumann<2, 1>, Topologies::Cube<2>::Topology,
                     TestCellHelpers::EmptyAPI, TestCellHelpers::NoOutput> TestCell2dCube;
    typedef TestCell<2, Stencils::Moore<2, 1>, Topologies::Torus<2>::Topology,
                     TestCellHelpers::EmptyAPI, TestCellHelpers::NoOutput> TestCell2dTorus;

    typedef TestCell<3, Stencils::VonNeumann<3, 1>, Topologies::Cube<3>::Topology,
                     TestCellHelpers::EmptyAPI, TestCellHelpers::NoOutput> TestCell3dCube;
    typedef TestCell<3, Stencils::Moore<3, 1>, Topologies::Torus<3>::Topology,
                     TestCellHelpers::EmptyAPI, TestCellHelpers::NoOutput> TestCell3dTorus;

    typedef TestInitializer<TestCell2dCube> TestInitializer2dCube;
    typedef TestInitializer<TestCell2dTorus> TestInitializer2dTorus;

    typedef TestInitializer<TestCell3dCube> TestInitializer3dCube;
    typedef TestInitializer<TestCell3dTorus> TestInitializer3dTorus;

    void test2dCube()
    {
        Coord<2> dim(421, 351);
        int numSteps = 21;
        CudaSimulator<TestCell2dCube> sim(new TestInitializer2dCube(dim, numSteps));

        sim.run();
    }

    void test3dCube()
    {
        Coord<3> dim(50, 20, 10);
        int numSteps = 5;
        CudaSimulator<TestCell3dCube> sim(new TestInitializer3dCube(dim, numSteps));

        TestWriter<TestCell3dCube> *writer = new TestWriter<TestCell3dCube>(1, 0, numSteps);
        sim.addWriter(writer);

        sim.run();
        TS_ASSERT(writer->allEventsDone());
    }

    void test3DTorus()
    {
        Coord<3> dim(50, 20, 10);
        int numSteps = 5;
        CudaSimulator<TestCell3dTorus> sim(new TestInitializer3dTorus(dim, numSteps));

        TestWriter<TestCell3dTorus> *writer = new TestWriter<TestCell3dTorus>(1, 0, numSteps);
        sim.addWriter(writer);

        sim.run();
        TS_ASSERT(writer->allEventsDone());
    }
};

}
