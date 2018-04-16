#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <libgeodecomp/parallelization/hpxsimulator.h>
#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/io/mocksteerer.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>

using namespace LibGeoDecomp;

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

    void update(const CoordMap<ConwayCell>& neighborhood, unsigned)
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

LIBGEODECOMP_REGISTER_HPX_COMM_TYPE(ConwayCell)

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
        startCells << Coord<2>(11, 10)
                   << Coord<2>(12, 11)
                   << Coord<2>(10, 12) << Coord<2>(11, 12) << Coord<2>(12, 12);


        // ...add a Diehard pattern...
        //                x
        //          xx
        //           x   xxx
        startCells << Coord<2>(61, 69)
                   << Coord<2>(55, 70) << Coord<2>(56, 70)
                   << Coord<2>(56, 71)
                   << Coord<2>(60, 71) << Coord<2>(61, 71) << Coord<2>(62, 71);

        // ...and an Acorn pattern:
        //        x
        //          x
        //       xx  xxx
        startCells << Coord<2>(111, 30)
                   << Coord<2>(113, 31)
                   << Coord<2>(110, 32) << Coord<2>(111, 32)
                   << Coord<2>(113, 32) << Coord<2>(114, 32) << Coord<2>(115, 32);


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

typedef HpxSimulator<ConwayCell, RecursiveBisectionPartition<2> > SimulatorType;

namespace LibGeoDecomp {

class HpxSimulator2Test : public CxxTest::TestSuite
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
            false,
            "/HpxSimulatorTest/testBasic");

        MockWriter<ConwayCell> *writer = new MockWriter<ConwayCell>(events, outputFrequency);
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

    void testSteererCallback()
    {
        CellInitializer *init = new CellInitializer(maxTimeSteps);

        std::vector<double> updateGroupSpeeds(1, 1.0);
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 1;
        int steeringPeriod = 7;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testSteererCallback");

        SharedPtr<MockSteerer<ConwayCell>::EventsStore>::Type events(new MockSteerer<ConwayCell>::EventsStore);
        sim.addSteerer(new MockSteerer<ConwayCell>(steeringPeriod, events));

        sim.run();

        MockSteerer<ConwayCell>::EventsStore expectedEvents;
        int startStep = init->startStep();
        expectedEvents << MockSteerer<ConwayCell>::Event(startStep, STEERER_INITIALIZED, rank, false);
        expectedEvents << MockSteerer<ConwayCell>::Event(startStep, STEERER_INITIALIZED, rank, true);

        for (unsigned i = startStep + steeringPeriod; i < init->maxSteps(); i += steeringPeriod) {
            expectedEvents << MockSteerer<ConwayCell>::Event(i, STEERER_NEXT_STEP, rank, false);
            expectedEvents << MockSteerer<ConwayCell>::Event(i, STEERER_NEXT_STEP, rank, true);
        }

        expectedEvents << MockSteerer<ConwayCell>::Event(init->maxSteps(), STEERER_ALL_DONE, rank, false);
        expectedEvents << MockSteerer<ConwayCell>::Event(init->maxSteps(), STEERER_ALL_DONE, rank, true);

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
        int steeringPeriod = 3;
        SimulatorType sim(
            init,
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulatorTest/testHeterogeneous");

        MockWriter<ConwayCell> *writer = new MockWriter<ConwayCell>(events, outputFrequency);
        sim.addWriter(writer);

        SharedPtr<MockSteerer<ConwayCell>::EventsStore>::Type steererEvents(new MockSteerer<ConwayCell>::EventsStore);
        sim.addSteerer(new MockSteerer<ConwayCell>(steeringPeriod, steererEvents));

        sim.run();

        // check writer interaction:
        MockWriter<ConwayCell>::EventsStore expectedWriterEvents;
        int startStep = init->startStep();

        std::size_t startRank = (rank + 0) * (rank + 1) / 2;
        std::size_t endRank   = (rank + 1) * (rank + 2) / 2;

        for (std::size_t groupRank = startRank; groupRank < endRank; ++groupRank) {
            expectedWriterEvents << MockWriter<ConwayCell>::Event(startStep, WRITER_INITIALIZED, groupRank, false);
            expectedWriterEvents << MockWriter<ConwayCell>::Event(startStep, WRITER_INITIALIZED, groupRank, true);

            for (unsigned i = startStep + outputFrequency; i < init->maxSteps(); i += outputFrequency) {
                expectedWriterEvents << MockWriter<ConwayCell>::Event(i, WRITER_STEP_FINISHED, groupRank, false);
                expectedWriterEvents << MockWriter<ConwayCell>::Event(i, WRITER_STEP_FINISHED, groupRank, true);
            }

            expectedWriterEvents << MockWriter<ConwayCell>::Event(init->maxSteps(), WRITER_ALL_DONE, groupRank, false);
            expectedWriterEvents << MockWriter<ConwayCell>::Event(init->maxSteps(), WRITER_ALL_DONE, groupRank, true);
        }

        TS_ASSERT_EQUALS(expectedWriterEvents.size(), events->size());
        TS_ASSERT_EQUALS(expectedWriterEvents,       *events);

        // check steerer interaction:
        MockSteerer<ConwayCell>::EventsStore expectedSteererEvents;
        startStep = init->startStep();

        for (std::size_t groupRank = startRank; groupRank < endRank; ++groupRank) {
            expectedSteererEvents << MockSteerer<ConwayCell>::Event(startStep, STEERER_INITIALIZED, groupRank, false);
            expectedSteererEvents << MockSteerer<ConwayCell>::Event(startStep, STEERER_INITIALIZED, groupRank, true);

            for (unsigned i = startStep + steeringPeriod; i < init->maxSteps(); i += steeringPeriod) {
                expectedSteererEvents << MockSteerer<ConwayCell>::Event(i, STEERER_NEXT_STEP, groupRank, false);
                expectedSteererEvents << MockSteerer<ConwayCell>::Event(i, STEERER_NEXT_STEP, groupRank, true);
            }

            expectedSteererEvents << MockSteerer<ConwayCell>::Event(init->maxSteps(), STEERER_ALL_DONE, groupRank, false);
            expectedSteererEvents << MockSteerer<ConwayCell>::Event(init->maxSteps(), STEERER_ALL_DONE, groupRank, true);
        }

        TS_ASSERT_EQUALS(expectedSteererEvents.size(), steererEvents->size());
        TS_ASSERT_EQUALS(expectedSteererEvents,       *steererEvents);
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
