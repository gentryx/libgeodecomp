#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/geometry/partitions/unstructuredstripingpartition.h>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/io/mocksteerer.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/io/paralleltestwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/teststeerer.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/parallelization/hpxsimulator.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

using namespace LibGeoDecomp;

typedef TestCell<2> TestCell2;

LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(TestCell2)

namespace LibGeoDecomp {

class HpxSimulatorTest : public CxxTest::TestSuite
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

    void testWithTestCell2D()
    {
        typedef HpxSimulator<TestCell<2>, RecursiveBisectionPartition<2> > SimulatorType;
        std::vector<double> updateGroupSpeeds(1, 1.0);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;
        Coord<2> dim(100, 50);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);

        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testWithTestCell2D");

        MockWriter<TestCell<2> > *writer = new MockWriter<TestCell<2> >(events, outputFrequency);
        sim.addWriter(writer);

        sim.run();

        MockWriter<>::EventsStore expectedEvents;
        int startStep = init->startStep();
        expectedEvents << MockWriter<>::Event(startStep, WRITER_INITIALIZED, rank, false);
        expectedEvents << MockWriter<>::Event(startStep, WRITER_INITIALIZED, rank, true);

        for (unsigned i = startStep + outputFrequency; i < init->maxSteps(); i += outputFrequency) {
            expectedEvents << MockWriter<>::Event(i, WRITER_STEP_FINISHED, rank, false);
            expectedEvents << MockWriter<>::Event(i, WRITER_STEP_FINISHED, rank, true);
        }

        expectedEvents << MockWriter<>::Event(init->maxSteps(), WRITER_ALL_DONE, rank, false);
        expectedEvents << MockWriter<>::Event(init->maxSteps(), WRITER_ALL_DONE, rank, true);

        TS_ASSERT_EQUALS(expectedEvents.size(), events->size());
        TS_ASSERT_EQUALS(expectedEvents,       *events);
    }

    void testSteererFunctionality2DWithGhostZoneWidth1()
    {
        typedef HpxSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
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
            "/HpxSimulatorTest/testSteererFunctionality2DWithGhostZoneWidth1");

        sim.addSteerer(new TestSteerer<2>(5, 25, 4711 * 27));
        sim.run();
    }


    void testSteererFunctionality2DWithGhostZoneWidth2()
    {
        typedef HpxSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
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
            "/HpxSimulatorTest/testSteererFunctionality2DWithGhostZoneWidth2");

        sim.addSteerer(new TestSteerer<2>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality2DWithGhostZoneWidth3()
    {
        typedef HpxSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
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
            "/HpxSimulatorTest/testSteererFunctionality2DWithGhostZoneWidth3");

        sim.addSteerer(new TestSteerer<2>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality2DWithGhostZoneWidth4()
    {
        typedef HpxSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
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
            "/HpxSimulatorTest/testSteererFunctionality2DWithGhostZoneWidth4");

        sim.addSteerer(new TestSteerer<2>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality2DWithGhostZoneWidth5()
    {
        typedef HpxSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 5;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            "/HpxSimulatorTest/testSteererFunctionality2DWithGhostZoneWidth5");

        sim.addSteerer(new TestSteerer<2>(5, 25, 4711 * 27));
        sim.run();
    }

    void testSteererFunctionality2DWithGhostZoneWidth6()
    {
        typedef HpxSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
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
            "/HpxSimulatorTest/testSteererFunctionality2DWithGhostZoneWidth6");

        sim.addSteerer(new TestSteerer<2>(5, 25, 4711 * 27));
        sim.run();
    }

    void testWriterFunctionality2DWithGhostZoneWidth1()
    {
        typedef HpxSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
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
            "/HpxSimulatorTest/testWriterFunctionality2DWithGhostZoneWidth1");

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

        sim.addWriter(new ParallelTestWriter<TestCell<2> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality2DWithGhostZoneWidth2()
    {
        typedef HpxSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
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
            "/HpxSimulatorTest/testWriterFunctionality2DWithGhostZoneWidth2");

        sim.addSteerer(new TestSteerer<2>(5, 25, 4711 * 27));
        sim.run();
    }

    void testWriterFunctionality2DWithGhostZoneWidth3()
    {
        typedef HpxSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
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
            "/HpxSimulatorTest/testWriterFunctionality2DWithGhostZoneWidth3");

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

        sim.addWriter(new ParallelTestWriter<TestCell<2> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality2DWithGhostZoneWidth4()
    {
        typedef HpxSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
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
            "/HpxSimulatorTest/testWriterFunctionality2DWithGhostZoneWidth4");

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

        sim.addWriter(new ParallelTestWriter<TestCell<2> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality2DWithGhostZoneWidth5()
    {
        typedef HpxSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
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
            "/HpxSimulatorTest/testWriterFunctionality2DWithGhostZoneWidth5");

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

        sim.addWriter(new ParallelTestWriter<TestCell<2> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
    }

    void testWriterFunctionality2DWithGhostZoneWidth6()
    {
        typedef HpxSimulator<TestCell<2>, ZCurvePartition<2> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        maxTimeSteps = 90;
        Coord<2> dim(20, 25);

        TestInitializer<TestCell<2> > *init = new TestInitializer<TestCell<2> >(dim, maxTimeSteps);
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
            "/HpxSimulatorTest/testWriterFunctionality2DWithGhostZoneWidth6");

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

        sim.addWriter(new ParallelTestWriter<TestCell<2> >(5, expectedWriterSteps, expectedWriterEvents));
        sim.run();
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
