#include <cxxtest/TestSuite.h>
#include <iostream>
#include <libflatarray/short_vec.hpp>
#include <libgeodecomp/io/memorywriter.h>
#include <libgeodecomp/io/mockinitializer.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/io/mocksteerer.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/teststeerer.h>
#include <libgeodecomp/io/testwriter.h>
#include <libgeodecomp/parallelization/openmpsimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class OpenMPSimulatorTest : public CxxTest::TestSuite
{
public:
    static const int NANO_STEPS_2D = APITraits::SelectNanoSteps<TestCell<2> >::VALUE;
    static const int NANO_STEPS_3D = APITraits::SelectNanoSteps<TestCell<3> >::VALUE;
    typedef MockSteerer<TestCell<2> > MockSteererType;
    typedef GridBase<TestCell<2>, 2> GridBaseType;

    void setUp()
    {
        dim = Coord<2>(17, 12);
        maxSteps = 21;
        startStep = 13;

        init.reset(createInitializer());
        simulator.reset(new OpenMPSimulator<TestCell<2> >(createInitializer()));
        events.reset(new MockWriter<>::EventsStore);
    }

    void tearDown()
    {
        simulator.reset();
        init.reset();
    }

    void testInitialization()
    {
        TS_ASSERT_EQUALS(simulator->getGrid()->dimensions().x(), 17);
        TS_ASSERT_EQUALS(simulator->getGrid()->dimensions().y(), 12);
        TS_ASSERT_TEST_GRID(GridBaseType, *simulator->getGrid(), startStep * NANO_STEPS_2D);
    }

    void testStep()
    {
        TS_ASSERT_EQUALS(startStep, simulator->getStep());

        simulator->step();
        const GridBase<TestCell<2>, 2> *grid = simulator->getGrid();
        TS_ASSERT_TEST_GRID(GridBaseType, *grid,
                            (startStep + 1) * NANO_STEPS_2D);
        TS_ASSERT_EQUALS(startStep + 1, simulator->getStep());
    }

    void testRun()
    {
        simulator->run();
        TS_ASSERT_EQUALS(init->maxSteps(), simulator->getStep());
        TS_ASSERT_TEST_GRID(
            GridBaseType,
            *simulator->getGrid(),
            init->maxSteps() * NANO_STEPS_2D);
    }

    void testWriterInvocation()
    {
        unsigned period = 4;
        simulator->addWriter(new TestWriter<>(period, 13, 21));
        simulator->run();
    }

    void testDeleteInitializer()
    {
        MockInitializer::events = "";
        {
            OpenMPSimulator<TestCell<2> > foo(new MockInitializer);
        }
        TS_ASSERT_EQUALS(
            MockInitializer::events,
            "created, configString: ''\ndeleted\n");
    }

    void testRegisterWriter()
    {
        MockWriter<> *w = new MockWriter<>(events);
        simulator->addWriter(w);
        OpenMPSimulator<TestCell<2> >::WriterVector writers = simulator->writers;
        TS_ASSERT_EQUALS(std::size_t(1), writers.size());
        TS_ASSERT_EQUALS(w, writers[0].get());
    }

    void testOpenMPSimulatorShouldCallBackWriter()
    {
        MockWriter<> *w = new MockWriter<>(events, 3);
        simulator->addWriter(w);
        simulator->run();

        MockWriter<>::EventsStore expectedEvents;
        expectedEvents << MockWriter<>::Event(startStep, WRITER_INITIALIZED, 0, true);

        for (unsigned i = startStep + 2; i <= init->maxSteps(); i += 3) {
            expectedEvents << MockWriter<>::Event(i, WRITER_STEP_FINISHED, 0, true);
        }

        expectedEvents << MockWriter<>::Event(init->maxSteps(), WRITER_ALL_DONE, 0, true);

        TS_ASSERT_EQUALS(expectedEvents, *events);
    }

    void testRunMustResetGridPriorToSimulation()
    {
        MockWriter<> *eventWriter1 = new MockWriter<>(events);
        MemoryWriter<TestCell<2> > *gridWriter1 =
            new MemoryWriter<TestCell<2> >();
        simulator->addWriter(eventWriter1);
        simulator->addWriter(gridWriter1);

        simulator->run();
        MockWriter<>::EventsStore events1 = *events;
        std::vector<Grid<TestCell<2> > > grids1 = gridWriter1->getGrids();


        SharedPtr<MockWriter<>::EventsStore>::Type events2(new MockWriter<>::EventsStore);
        events->clear();
        MockWriter<> *eventWriter2 = new MockWriter<>(events2);
        MemoryWriter<TestCell<2> > *gridWriter2 =
            new MemoryWriter<TestCell<2> >();
        simulator->addWriter(eventWriter2);
        simulator->addWriter(gridWriter2);
        simulator->run();
        std::vector<Grid<TestCell<2> > > grids2 = gridWriter2->getGrids();

        TS_ASSERT_EQUALS(events1.size(), events2->size());
        TS_ASSERT_EQUALS(events1, *events2);
        TS_ASSERT_EQUALS(grids1, grids2);
    }

    typedef APITraits::SelectTopology<TestCell<3> >::Value Topology;
    typedef Grid<TestCell<3>, Topology> Grid3D;
    typedef GridBase<TestCell<3>, 3> GridBase3D;

    void test3D()
    {
        OpenMPSimulator<TestCell<3> > sim(new TestInitializer<TestCell<3> >());
        TS_ASSERT_TEST_GRID(GridBase3D, *sim.getGrid(), 0);

        sim.step();
        TS_ASSERT_TEST_GRID(GridBase3D, *sim.getGrid(), NANO_STEPS_3D);

        sim.nanoStep(0);
        TS_ASSERT_TEST_GRID(GridBase3D, *sim.getGrid(), NANO_STEPS_3D + 1);

        sim.run();
        TS_ASSERT_TEST_GRID(GridBase3D, *sim.getGrid(), 21 * NANO_STEPS_3D);
    }

    void testSteererCallback()
    {
        SharedPtr<MockSteererType::EventsStore>::Type events(new MockSteererType::EventsStore);
        simulator->addSteerer(new MockSteererType(5, events));

        MockSteererType::EventsStore expectedEvents;
        expectedEvents << MockSteererType::Event(13, STEERER_INITIALIZED, 0, true);
        for (std::size_t t = startStep; t < maxSteps; t += 1) {
            if ((t % 5) == 0) {
                expectedEvents << MockSteererType::Event(t, STEERER_NEXT_STEP, 0, true);
            }
        }

        expectedEvents << MockSteererType::Event(21, STEERER_ALL_DONE,  0, true)
                       << MockSteererType::Event(-1, STEERER_ALL_DONE, -1, true);

        simulator->run();
        simulator.reset();

        TS_ASSERT_EQUALS(*events, expectedEvents);
    }

    void testSteererCanTerminateSimulation()
    {
        unsigned eventStep = 15;
        unsigned endStep = 19;
        unsigned jumpSteps = 2;
        simulator->addSteerer(new TestSteerer<2>(1, eventStep, NANO_STEPS_2D * jumpSteps, endStep));
        simulator->run();

        // simulators are allowed to run past the end signal of the
        // steerer b/c of issues with updating ghost zones first.
        TS_ASSERT_TEST_GRID(
            GridBaseType,
            *simulator->getGrid(),
            (endStep + 1 + jumpSteps) * NANO_STEPS_2D);
    }

    void testSoA()
    {
        typedef GridBase<TestCellSoA, 3> GridBaseType;
        OpenMPSimulator<TestCellSoA> sim(new TestInitializer<TestCellSoA>());
        TS_ASSERT_TEST_GRID(GridBaseType, *sim.getGrid(), 0);

        sim.step();
        TS_ASSERT_TEST_GRID(GridBaseType, *sim.getGrid(), NANO_STEPS_3D);

        sim.nanoStep(0);
        TS_ASSERT_TEST_GRID(GridBaseType, *sim.getGrid(), NANO_STEPS_3D + 1);

        sim.run();
        TS_ASSERT_TEST_GRID(GridBaseType, *sim.getGrid(), 21 * NANO_STEPS_3D);
    }

private:
    SharedPtr<MockWriter<>::EventsStore>::Type events;
    SharedPtr<OpenMPSimulator<TestCell<2> > >::Type simulator;
    SharedPtr<Initializer<TestCell<2> > >::Type init;
    unsigned maxSteps;
    unsigned startStep;
    Coord<2> dim;

    Initializer<TestCell<2> > *createInitializer()
    {
        return new TestInitializer<TestCell<2> >(dim, maxSteps, startStep);
    }
};

}
