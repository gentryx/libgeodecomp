#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <libgeodecomp/geometry/partitions/unstructuredstripingpartition.h>
#include <libgeodecomp/io/paralleltestwriter.h>
#include <libgeodecomp/io/unstructuredtestinitializer.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/parallelization/hpxsimulator.h>

using namespace LibGeoDecomp;

typedef UnstructuredTestCell<> TestCell4;
LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(TestCell4)

namespace LibGeoDecomp {

class HpxSimulator5Test : public CxxTest::TestSuite
{
public:
    void testUnstructuredSoA1()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA1 TestCellType;
        int startStep = 7;
        int endStep = 20;

        std::vector<double> updateGroupSpeeds(1, 1.0);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;

        HpxSimulator<TestCellType, UnstructuredStripingPartition> sim(
            new UnstructuredTestInitializer<TestCellType>(614, endStep, startStep),
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulator5Test/testUnstructuredSoA1");

        std::vector<unsigned> expectedSteps;
        std::vector<WriterEvent> expectedEvents;
        expectedSteps << 7
                      << 10
                      << 15
                      << 20;
        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;
        sim.addWriter(new ParallelTestWriter<TestCellType>(5, expectedSteps, expectedEvents));

        sim.run();
#endif
    }

    void testUnstructuredSoA2()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA2 TestCellType;
        int startStep = 7;
        int endStep = 15;

        std::vector<double> updateGroupSpeeds(1, 1.0);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;

        HpxSimulator<TestCellType, UnstructuredStripingPartition> sim(
            new UnstructuredTestInitializer<TestCellType>(632, endStep, startStep),
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulator5Test/testUnstructuredSoA2");

        std::vector<unsigned> expectedSteps;
        std::vector<WriterEvent> expectedEvents;
        expectedSteps << 7
                      << 8
                      << 10
                      << 12
                      << 14
                      << 15;
        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;
        sim.addWriter(new ParallelTestWriter<TestCellType>(2, expectedSteps, expectedEvents));

        sim.run();
#endif
    }

    void testUnstructuredSoA3()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA3 TestCellType;
        int startStep = 7;
        int endStep = 19;

        std::vector<double> updateGroupSpeeds(1, 1.0);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;

        HpxSimulator<TestCellType, UnstructuredStripingPartition> sim(
            new UnstructuredTestInitializer<TestCellType>(655, endStep, startStep),
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulator5Test/testUnstructuredSoA3");

        std::vector<unsigned> expectedSteps;
        std::vector<WriterEvent> expectedEvents;
        expectedSteps << 7
                      << 9
                      << 12
                      << 15
                      << 18
                      << 19;
        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;
        sim.addWriter(new ParallelTestWriter<TestCellType>(3, expectedSteps, expectedEvents));

        sim.run();
#endif
    }

    void testUnstructuredSoA4()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        typedef UnstructuredTestCellSoA1 TestCellType;
        int startStep = 5;
        int endStep = 24;

        std::vector<double> updateGroupSpeeds(1, 1.0);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;

        HpxSimulator<TestCellType, UnstructuredStripingPartition> sim(
            new UnstructuredTestInitializer<TestCellType>(444, endStep, startStep),
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulator5Test/testUnstructuredSoA4");

        std::vector<unsigned> expectedSteps;
        std::vector<WriterEvent> expectedEvents;
        expectedSteps << 5
                      << 6
                      << 9
                      << 12
                      << 15
                      << 18
                      << 21
                      << 24;
        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;
        sim.addWriter(new ParallelTestWriter<TestCellType>(3, expectedSteps, expectedEvents));

        sim.run();
#endif
    }

    void testUnstructured()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        std::vector<double> updateGroupSpeeds(1, 1.0);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;

        int startStep = 7;
        int endStep = 20;

        typedef HpxSimulator<TestCell4, UnstructuredStripingPartition> SimulatorType;

        SimulatorType sim(
            new UnstructuredTestInitializer<TestCell4>(614, endStep, startStep),
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulator5Test/testUnstructured");

        std::vector<unsigned> expectedSteps;
        std::vector<WriterEvent> expectedEvents;
        expectedSteps << 7
                      << 8
                      << 12
                      << 16
                      << 20;
        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;
        sim.addWriter(new ParallelTestWriter<TestCell4>(4, expectedSteps, expectedEvents));

        sim.run();
#endif
    }

};

}
