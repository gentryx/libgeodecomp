#include <cuda.h>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/testwriter.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/cudasimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

typedef TestCell<1, Stencils::VonNeumann<1, 1>, Topologies::Cube<1>::Topology,
                 TestCellHelpers::EmptyAPI, TestCellHelpers::NoOutput> TestCell1dCube;
typedef TestCell<1, Stencils::Moore<1, 1>, Topologies::Torus<1>::Topology,
                 TestCellHelpers::EmptyAPI, TestCellHelpers::NoOutput> TestCell1dTorus;

typedef TestCell<2, Stencils::VonNeumann<2, 1>, Topologies::Cube<2>::Topology,
                 TestCellHelpers::EmptyAPI, TestCellHelpers::NoOutput> TestCell2dCube;
typedef TestCell<2, Stencils::Moore<2, 1>, Topologies::Torus<2>::Topology,
                 TestCellHelpers::EmptyAPI, TestCellHelpers::NoOutput> TestCell2dTorus;

typedef TestCell<3, Stencils::VonNeumann<3, 1>, Topologies::Cube<3>::Topology,
                 TestCellHelpers::EmptyAPI, TestCellHelpers::NoOutput> TestCell3dCube;
typedef TestCell<3, Stencils::Moore<3, 1>, Topologies::Torus<3>::Topology,
                 TestCellHelpers::EmptyAPI, TestCellHelpers::NoOutput> TestCell3dTorus;

typedef TestCell<3, Stencils::Moore<3, 1>, Topologies::Cube<3>::Topology,
                 TestCellHelpers::SoAAPI, TestCellHelpers::NoOutput> TestCellSoA3dCube;
typedef TestCell<3, Stencils::Moore<3, 1>, Topologies::Torus<3>::Topology,
                 TestCellHelpers::SoAAPI, TestCellHelpers::NoOutput> TestCellSoA3dTorus;

typedef TestInitializer<TestCell1dCube> TestInitializer1dCube;
typedef TestInitializer<TestCell1dTorus> TestInitializer1dTorus;

typedef TestInitializer<TestCell2dCube> TestInitializer2dCube;
typedef TestInitializer<TestCell2dTorus> TestInitializer2dTorus;

typedef TestInitializer<TestCell3dCube> TestInitializer3dCube;
typedef TestInitializer<TestCell3dTorus> TestInitializer3dTorus;

typedef TestInitializer<TestCellSoA3dCube> TestInitializerSoA3dCube;
typedef TestInitializer<TestCellSoA3dTorus> TestInitializerSoA3dTorus;

}

LIBFLATARRAY_REGISTER_SOA(
    LibGeoDecomp::TestCellSoA3dCube,
    ((LibGeoDecomp::Coord<3>)(pos))
    ((LibGeoDecomp::CoordBox<3>)(dimensions))
    ((unsigned)(cycleCounter))
    ((bool)(isEdgeCell))
    ((bool)(isValid))((double)(testValue)))

LIBFLATARRAY_REGISTER_SOA(
    LibGeoDecomp::TestCellSoA3dTorus,
    ((LibGeoDecomp::Coord<3>)(pos))
    ((LibGeoDecomp::CoordBox<3>)(dimensions))
    ((unsigned)(cycleCounter))
    ((bool)(isEdgeCell))
    ((bool)(isValid))((double)(testValue)))


namespace LibGeoDecomp {

class CUDASimulatorTest : public CxxTest::TestSuite
{
public:

    void test1dCube()
    {
        Coord<1> dim(777);
        unsigned numSteps = 33;
        CUDASimulator<TestCell1dCube> sim(new TestInitializer1dCube(dim, numSteps));

        TestWriter<TestCell1dCube> *writer = new TestWriter<TestCell1dCube>(1, 0, numSteps);
        sim.addWriter(writer);

        sim.run();
        TS_ASSERT(writer->allEventsDone());
    }

    void test1dTorus()
    {
        Coord<1> dim(666);
        unsigned startStep = 40;
        unsigned endStep = 70;
        CUDASimulator<TestCell1dTorus> sim(new TestInitializer1dTorus(dim, endStep, startStep));

        TestWriter<TestCell1dTorus> *writer = new TestWriter<TestCell1dTorus>(3, startStep, endStep);
        sim.addWriter(writer);

        sim.run();
        TS_ASSERT(writer->allEventsDone());
    }

    void test2dCube()
    {
        Coord<2> dim(121, 151);
        unsigned numSteps = 21;
        CUDASimulator<TestCell2dCube> sim(new TestInitializer2dCube(dim, numSteps));

        TestWriter<TestCell2dCube> *writer = new TestWriter<TestCell2dCube>(8, 0, numSteps);
        sim.addWriter(writer);

        sim.run();
        TS_ASSERT(writer->allEventsDone());
    }

    void test2dTorus()
    {
        Coord<2> dim(141, 131);
        unsigned startStep = 35;
        unsigned endStep = 60;
        CUDASimulator<TestCell2dTorus> sim(new TestInitializer2dTorus(dim, endStep, startStep));

        TestWriter<TestCell2dTorus> *writer = new TestWriter<TestCell2dTorus>(6, startStep, endStep);
        sim.addWriter(writer);

        sim.run();
        TS_ASSERT(writer->allEventsDone());
    }

    void test3dCube()
    {
        Coord<3> dim(50, 20, 10);
        unsigned numSteps = 5;
        CUDASimulator<TestCell3dCube> sim(new TestInitializer3dCube(dim, numSteps));

        TestWriter<TestCell3dCube> *writer = new TestWriter<TestCell3dCube>(1, 0, numSteps);
        sim.addWriter(writer);

        sim.run();
        TS_ASSERT(writer->allEventsDone());
    }

    void test3DTorus()
    {
        Coord<3> dim(50, 20, 10);
        unsigned startStep = 28;
        unsigned endStep = 38;
        unsigned ioPeriod = 3;
        CUDASimulator<TestCell3dTorus> sim(new TestInitializer3dTorus(dim, endStep, startStep));

        TestWriter<TestCell3dTorus> *writer = new TestWriter<TestCell3dTorus>(ioPeriod, startStep, endStep);
        sim.addWriter(writer);

        sim.run();
        TS_ASSERT(writer->allEventsDone());
    }

    void testMultipleWavefronts2DCube()
    {
        Coord<2> dim(135, 127);
        unsigned numSteps = 21;
        CUDASimulator<TestCell2dCube> sim(
            new TestInitializer2dCube(dim, numSteps),
            Coord<3>(128, 5, 1));

        TestWriter<TestCell2dCube> *writer = new TestWriter<TestCell2dCube>(5, 0, numSteps);
        sim.addWriter(writer);

        sim.run();
        TS_ASSERT(writer->allEventsDone());
    }

    void testMultipleWavefronts2DTorus()
    {
        Coord<2> dim(251, 91);
        unsigned numSteps = 21;
        CUDASimulator<TestCell2dTorus> sim(
            new TestInitializer2dTorus(dim, numSteps),
            Coord<3>(128, 5, 1));

        TestWriter<TestCell2dTorus> *writer = new TestWriter<TestCell2dTorus>(6, 0, numSteps);
        sim.addWriter(writer);

        sim.run();
        TS_ASSERT(writer->allEventsDone());
    }

    void testMultipleWavefronts3DCube()
    {
        Coord<3> dim(52, 20, 14);
        unsigned numSteps = 6;
        CUDASimulator<TestCell3dCube> sim(
            new TestInitializer3dCube(dim, numSteps),
            Coord<3>(128, 2, 3));

        TestWriter<TestCell3dCube> *writer = new TestWriter<TestCell3dCube>(1, 0, numSteps);
        sim.addWriter(writer);

        sim.run();
        TS_ASSERT(writer->allEventsDone());
    }

    void testMultipleWavefronts3DTorus()
    {
        Coord<3> dim(54, 20, 13);
        unsigned numSteps = 7;
        CUDASimulator<TestCell3dTorus> sim(
            new TestInitializer3dTorus(dim, numSteps),
            Coord<3>(64, 2, 3));

        TestWriter<TestCell3dTorus> *writer = new TestWriter<TestCell3dTorus>(1, 0, numSteps);
        sim.addWriter(writer);

        sim.run();
        TS_ASSERT(writer->allEventsDone());
    }

    void testSoA2dTorus()
    {
        // fixme
    }

    void testSoA2dCube()
    {
        // fixme
    }

    void testSoA3dTorus()
    {
        // fixme
    }

    void testSoA3dCube()
    {
        Coord<3> dim(54, 20, 13);
        unsigned numSteps = 7;
        CUDASimulator<TestCellSoA3dTorus> sim(
            new TestInitializerSoA3dTorus(dim, numSteps),
            Coord<3>(64, 2, 3));

        TestWriter<TestCellSoA3dTorus> *writer = new TestWriter<TestCellSoA3dTorus>(1, 0, numSteps);
        // fixme
        // sim.addWriter(writer);

        sim.run();
        // TS_ASSERT(writer->allEventsDone());
    }
};

}
