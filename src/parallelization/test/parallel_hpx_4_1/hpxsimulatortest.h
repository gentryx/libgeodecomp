#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <boost/assign/std/deque.hpp>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/parallelization/hpxsimulator.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/mockpatchaccepter.h>

using namespace LibGeoDecomp;
using namespace boost::assign;

class ConwayCell
{
public:
    ConwayCell(bool alive = false) :
        alive(alive)
    {}

    int countLivingNeighbors(const CoordMap<ConwayCell>& neighborhood)
    {
        int ret = 0;
        for (int y = -1; y < 2; ++y) {
            for (int x = -1; x < 2; ++x) {
                ret += neighborhood[Coord<2>(x, y)].alive;
            }
        }
        ret -= neighborhood[Coord<2>(0, 0)].alive;
        return ret;
    }

    void update(const CoordMap<ConwayCell>& neighborhood, const unsigned&)
    {
        int livingNeighbors = countLivingNeighbors(neighborhood);
        alive = neighborhood[Coord<2>(0, 0)].alive;
        if (alive) {
            alive = (2 <= livingNeighbors) && (livingNeighbors <= 3);
        } else {
            alive = (livingNeighbors == 3);
        }
    }

    bool alive;

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & alive;
    }
};

typedef TestCell<2> TestCell2;
typedef TestCell<3> TestCell3;

LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(ConwayCell)
LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(TestCell2)
LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(TestCell3)

class CellInitializer : public SimpleInitializer<ConwayCell>
{
public:
    CellInitializer(int maxTimeSteps = 100) :
        SimpleInitializer<ConwayCell>(Coord<2>(160, 90), maxTimeSteps)
    {}

    virtual void grid(GridBase<ConwayCell, 2> *ret)
    {
        CoordBox<2> rect = ret->boundingBox();
        std::vector<Coord<2> > startCells;
        // start with a single glider...
        //          x
        //           x
        //         xxx
        startCells +=
            Coord<2>(11, 10),
            Coord<2>(12, 11),
            Coord<2>(10, 12), Coord<2>(11, 12), Coord<2>(12, 12);


        // ...add a Diehard pattern...
        //                x
        //          xx
        //           x   xxx
        startCells +=
            Coord<2>(55, 70), Coord<2>(56, 70), Coord<2>(56, 71),
            Coord<2>(60, 71), Coord<2>(61, 71), Coord<2>(62, 71),
            Coord<2>(61, 69);

        // ...and an Acorn pattern:
        //        x
        //          x
        //       xx  xxx
        startCells +=
            Coord<2>(111, 30),
            Coord<2>(113, 31),
            Coord<2>(110, 32), Coord<2>(111, 32),
            Coord<2>(113, 32), Coord<2>(114, 32), Coord<2>(115, 32);


        for (std::vector<Coord<2> >::iterator i = startCells.begin();
             i != startCells.end();
             ++i) {
            if (rect.inBounds(*i)) {
                ret->set(*i, ConwayCell(true));
            }
        }
    }

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & hpx::serialization::base_object<SimpleInitializer<ConwayCell> >(*this);
    }
};

typedef HpxSimulator::HpxSimulator<ConwayCell, RecursiveBisectionPartition<2> > SimulatorType;

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
        events.reset(new MockWriter<>::EventVec);
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

    void testBasic()
    {
        CellInitializer *init = new CellInitializer(maxTimeSteps);

        std::vector<double> updateGroupSpeeds(1, 1.0);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            "/HpxSimulatorTest/testBasic");

        MockWriter<ConwayCell> *writer = new MockWriter<ConwayCell>(events, outputFrequency);
        sim.addWriter(writer);

        sim.run();

        MockWriter<>::EventVec expectedEvents;
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

    void testHeterogeneous()
    {
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();
        outputFrequency = 5;
        maxTimeSteps = 66;
        CellInitializer *init = new CellInitializer(maxTimeSteps);
        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            "/HpxSimulatorTest/testHeterogeneous");

        MockWriter<ConwayCell> *writer = new MockWriter<ConwayCell>(events, outputFrequency);
        sim.addWriter(writer);

        sim.run();

        MockWriter<>::EventVec expectedEvents;
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

    void testWithTestCell2D()
    {
        typedef HpxSimulator::HpxSimulator<TestCell<2>, RecursiveBisectionPartition<2> > SimulatorType;
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
            "/HpxSimulatorTest/testWithTestCell2D");

        MockWriter<TestCell<2> > *writer = new MockWriter<TestCell<2> >(events, outputFrequency);
        sim.addWriter(writer);

        sim.run();

        MockWriter<>::EventVec expectedEvents;
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

    void testWithTestCell3DHeterogeneous()
    {
        typedef HpxSimulator::HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
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
            "/HpxSimulatorTest/testWithTestCell3DHeterogeneous");

        MockWriter<TestCell<3>> *writer = new MockWriter<TestCell<3>>(events, outputFrequency);
        sim.addWriter(writer);

        sim.run();

        MockWriter<>::EventVec expectedEvents;
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

    void testStepAndGetStep()
    {
        typedef HpxSimulator::HpxSimulator<TestCell<3>, ZCurvePartition<3> > SimulatorType;
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
    boost::shared_ptr<MockWriter<>::EventVec> events;
};

}
