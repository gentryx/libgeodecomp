#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/io/mocksteerer.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/io/paralleltestwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/teststeerer.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/parallelization/hpxsimulator.h>

using namespace LibGeoDecomp;

typedef TestCell<3> TestCell3;
LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(TestCell3)

namespace LibGeoDecomp {

class HpxSimulator3Test : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        outputFrequency = 1;
        maxTimeSteps = 10;
        rank = hpx::get_locality_id();
        localities = hpx::find_all_localities();
        events.reset(new MockWriter<>::EventsStore);
    }

    void tearDown()
    {
        int i;
        for (i = 0; i < maxTimeSteps; i += outputFrequency) {
            removeFiles(i);
        }

        if (i > maxTimeSteps) {
            i = maxTimeSteps;
        }
        removeFiles(i);
    }

    void testWithTestCell3DHeterogeneous()
    {
        typedef HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        outputFrequency = 5;
        maxTimeSteps = 9;
        Coord<3> dim(50, 40, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testWithTestCell3DHeterogeneous");

        MockWriter<TestCell<3>> *writer = new MockWriter<TestCell<3>>(events, outputFrequency);
        sim.addWriter(writer);

        sim.run();

        MockWriter<>::EventsStore expectedEvents;
        int startStep = init->startStep();
        std::size_t startRank = (rank + 0) * (rank + 1) / 2;
        std::size_t endRank   = (rank + 1) * (rank + 2) / 2;

        for (std::size_t groupRank = startRank; groupRank < endRank; ++groupRank) {
            expectedEvents << MockWriter<>::Event(startStep, WRITER_INITIALIZED, groupRank, false);
            expectedEvents << MockWriter<>::Event(startStep, WRITER_INITIALIZED, groupRank, true);

            for (unsigned i = startStep + outputFrequency; i < init->maxSteps(); i += outputFrequency) {
                expectedEvents << MockWriter<>::Event(i, WRITER_STEP_FINISHED, groupRank, false);
                expectedEvents << MockWriter<>::Event(i, WRITER_STEP_FINISHED, groupRank, true);
            }

            expectedEvents << MockWriter<>::Event(init->maxSteps(), WRITER_ALL_DONE, groupRank, false);
            expectedEvents << MockWriter<>::Event(init->maxSteps(), WRITER_ALL_DONE, groupRank, true);
        }

        TS_ASSERT_EQUALS(expectedEvents.size(), events->size());
        TS_ASSERT_EQUALS(expectedEvents,       *events);
    }

    void testSteererFunctionality3DWithGhostZoneWidth1()
    {
        typedef HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testSteererFunctionality3DWithGhostZoneWidth1");

        sim.addSteerer(new TestSteerer<3>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality3DWithGhostZoneWidth2()
    {
        typedef HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 2;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testSteererFunctionality3DWithGhostZoneWidth2");

        sim.addSteerer(new TestSteerer<3>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality3DWithGhostZoneWidth3()
    {
        typedef HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 3;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testSteererFunctionality3DWithGhostZoneWidth3");

        sim.addSteerer(new TestSteerer<3>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality3DWithGhostZoneWidth4()
    {
        typedef HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 4;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testSteererFunctionality3DWithGhostZoneWidth4");

        sim.addSteerer(new TestSteerer<3>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality3DWithGhostZoneWidth5()
    {
        typedef HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 5;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testSteererFunctionality3DWithGhostZoneWidth5");

        sim.addSteerer(new TestSteerer<3>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality3DWithGhostZoneWidth6()
    {
        typedef HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 6;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testSteererFunctionality3DWithGhostZoneWidth6");

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addSteerer(new TestSteerer<3>(5, 25, 4711 * 27));
        sim.run();
    }

    void testWriterFunctionality3DWithGhostZoneWidth1()
    {
        typedef HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testWriterFunctionality3DWithGhostZoneWidth1");

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<3> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality3DWithGhostZoneWidth2()
    {
        typedef HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 110;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 2;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testWriterFunctionality3DWithGhostZoneWidth2");

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<   0
                            <<   5
                            <<  10
                            <<  15
                            <<  20
                            <<  25
                            <<  30
                            <<  35
                            <<  40
                            <<  45
                            <<  50
                            <<  55
                            <<  60
                            <<  65
                            <<  70
                            <<  75
                            <<  80
                            <<  85
                            <<  90
                            <<  95
                            << 100
                            << 105
                            << 110;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<3> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality3DWithGhostZoneWidth3()
    {
        typedef HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 3;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testWriterFunctionality3DWithGhostZoneWidth3");

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<3> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality3DWithGhostZoneWidth4()
    {
        typedef HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 4;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testWriterFunctionality3DWithGhostZoneWidth4");

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<3> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality3DWithGhostZoneWidth5()
    {
        typedef HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 5;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testWriterFunctionality3DWithGhostZoneWidth5");

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<3> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterCallback3DWithGhostZoneWidth6()
    {
        typedef HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<3> dim(20, 25, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 6;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testWriterCallback3DWithGhostZoneWidth6");

        std::vector<unsigned> expectedWriterSteps;
        std::vector<WriterEvent> expectedWriterEvents;

        expectedWriterSteps <<  0
                            <<  5
                            << 10
                            << 15
                            << 20
                            << 25
                            << 30
                            << 35
                            << 40
                            << 45
                            << 50
                            << 55
                            << 60
                            << 65
                            << 70
                            << 75
                            << 80
                            << 85
                            << 90;

        expectedWriterEvents << WRITER_INITIALIZED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_STEP_FINISHED
                             << WRITER_ALL_DONE;

        sim.addWriter(new ParallelTestWriter<TestCell<3> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testStepAndGetStep()
    {
        typedef HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        outputFrequency = 5;
        int startStep = 4;
        maxTimeSteps = 29;
        Coord<3> dim(80, 40, 30);

        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >(dim, maxTimeSteps, startStep);
        std::vector<double> updateGroupSpeeds(4, 1.0);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testStepAndGetStep");

        TS_ASSERT_EQUALS(startStep + 0, sim.getStep());

        sim.step();
        TS_ASSERT_EQUALS(startStep + 1, sim.getStep());

        sim.step();
        TS_ASSERT_EQUALS(startStep + 2, sim.getStep());
    }

    void removeFile(std::string name)
    {
        remove(name.c_str());
    }

    void removeFiles(int timestep)
    {
        std::stringstream buf;
        buf << "game." << std::setfill('0') << std::setw(5) << timestep;
        removeFile(buf.str() + ".bov");
        removeFile(buf.str() + ".data");
    }

private:
    std::size_t rank;
    std::vector<hpx::id_type> localities;
    int outputFrequency;
    int maxTimeSteps;
    SharedPtr<MockWriter<>::EventsStore>::Type events;
};

}
