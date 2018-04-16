#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/io/paralleltestwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/parallelization/hpxsimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class HpxSimulator4Test : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        outputFrequency = 1;
        maxTimeSteps = 10;
        rank = hpx::get_locality_id();
        localities = hpx::find_all_localities();
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

    void testSoAWithGhostZoneWidth1()
    {
        int ghostZoneWidth = 3;
        int startStep = 0;
        int endStep = 25;
        Coord<3> dim(33, 40, 50);

        typedef HpxSimulator<TestCellSoA, ZCurvePartition<3> > SimulatorType;
        SimulatorType sim(
            new TestInitializer<TestCellSoA>(dim, endStep, startStep),
            std::vector<double>(1, 1.0),
            new TracingBalancer(new OozeBalancer()),
            10000,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testSoAWithGhostZoneWidth1");

        std::vector<unsigned> expectedSteps;
        std::vector<WriterEvent> expectedEvents;

        expectedSteps << 0
                      << 4
                      << 8
                      << 12
                      << 16
                      << 20
                      << 24
                      << 25;
        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;
        sim.addWriter(new ParallelTestWriter<TestCellSoA>(4, expectedSteps, expectedEvents));

        sim.run();
    }

    void testSoAWithGhostZoneWidth3()
    {
        int ghostZoneWidth = 3;
        int startStep = 3;
        int endStep = 30;
        Coord<3> dim(26, 25, 24);

        typedef HpxSimulator<TestCellSoA, ZCurvePartition<3> > SimulatorType;
        SimulatorType sim(
            new TestInitializer<TestCellSoA>(dim, endStep, startStep),
            std::vector<double>(1, 1.0),
            new TracingBalancer(new OozeBalancer()),
            10000,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testSoAWithGhostZoneWidth3");

        std::vector<unsigned> expectedSteps;
        std::vector<WriterEvent> expectedEvents;

        expectedSteps << 3
                      << 4
                      << 8
                      << 12
                      << 16
                      << 20
                      << 24
                      << 28
                      << 30;
        expectedEvents << WRITER_INITIALIZED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_STEP_FINISHED
                       << WRITER_ALL_DONE;
        sim.addWriter(new ParallelTestWriter<TestCellSoA>(4, expectedSteps, expectedEvents));

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
};

}
