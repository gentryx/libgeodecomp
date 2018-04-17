#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/io/collectingwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/testwriter.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/hiparsimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class HiParSimulator2Test : public CxxTest::TestSuite
{
public:

    void setUp()
    {
        rank = MPILayer().rank();
    }

    void testSoAWithGhostZoneWidth1()
    {
        int ghostZoneWidth = 1;
        int startStep = 0;
        int endStep = 21;
        Coord<3> dim(15, 10, 13);

        HiParSimulator<TestCellSoA, ZCurvePartition<3> > sim(
            new TestInitializer<TestCellSoA>(dim, endStep, startStep),
            rank? 0 : new NoOpBalancer(),
            100000,
            ghostZoneWidth);

        Writer<TestCellSoA> *writer = 0;
        if (rank == 0) {
            writer = new TestWriter<TestCellSoA>(3, startStep, endStep);
        }
        sim.addWriter(new CollectingWriter<TestCellSoA>(writer));

        sim.run();
    }

    void testSoAWithGhostZoneWidth3()
    {
        int ghostZoneWidth = 3;
        int startStep = 3;
        int endStep = 26;
        Coord<3> dim(16, 12, 14);

        HiParSimulator<TestCellSoA, ZCurvePartition<3> > sim(
            new TestInitializer<TestCellSoA>(dim, endStep, startStep),
            rank? 0 : new NoOpBalancer(),
            100000,
            ghostZoneWidth);

        Writer<TestCellSoA> *writer = 0;
        if (rank == 0) {
            writer = new TestWriter<TestCellSoA>(3, startStep, endStep);
        }
        sim.addWriter(new CollectingWriter<TestCellSoA>(writer));

        sim.run();
    }

    std::size_t rank;

};

}
