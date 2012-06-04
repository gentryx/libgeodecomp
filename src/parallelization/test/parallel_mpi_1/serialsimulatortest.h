#include <cxxtest/TestSuite.h>
#include <sstream>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/io/memorywriter.h>
#include <libgeodecomp/io/mockinitializer.h>
#include <libgeodecomp/io/mockwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class SerialSimulatorTest : public CxxTest::TestSuite 
{
private:
    SerialSimulator<TestCell<2> > *simulator;
    Initializer<TestCell<2> > *init;

public:

    void setUp() 
    {
        init = new TestInitializer<2>();
        simulator = new SerialSimulator<TestCell<2> >(new TestInitializer<2>());
    }
    
    void tearDown()
    {
        delete simulator;
        delete init;
    }

    void testInitialization()
    {
        TS_ASSERT_EQUALS(simulator->getGrid()->getDimensions().x(),  
                         (unsigned)17);
        TS_ASSERT_EQUALS(simulator->getGrid()->getDimensions().y(), 
                         (unsigned)12);
        TS_ASSERT_TEST_GRID(Grid<TestCell<2> >, *simulator->getGrid(), 0);
    }

    void testStep()
    {
        TS_ASSERT_EQUALS(0, (int)simulator->getStep());

        simulator->step();
        const Grid<TestCell<2> > *grid = simulator->getGrid();
        TS_ASSERT_TEST_GRID(Grid<TestCell<2> >, *grid, 
                            TestCell<2>::nanoSteps());
        TS_ASSERT_EQUALS(1, (int)simulator->getStep());
    }
    
    void testRun()
    {
        simulator->run();
        TS_ASSERT_EQUALS(init->maxSteps(), simulator->getStep());
        TS_ASSERT_TEST_GRID(
            Grid<TestCell<2> >, 
            *simulator->getGrid(), 
            init->maxSteps() * TestCell<2>::nanoSteps());
    }

    //fixme: move those tests converning just the abstract base class
    //Simulator to unitsimulator.h
    void testDeleteInitializer()
    {
        MockInitializer::events = "";
        {
            SerialSimulator<TestCell<2> > foo(new MockInitializer);
        }
        TS_ASSERT_EQUALS(
            MockInitializer::events, 
            "created, configString: ''\ndeleted\n");
    }
    
    void testRegisterWriter()
    {
        MockWriter *w = new MockWriter(simulator);
        SerialSimulator<TestCell<2> >::WriterVector writers = simulator->writers;
        TS_ASSERT_EQUALS((unsigned)1, writers.size());
        TS_ASSERT_EQUALS(w, writers[0].get());
    }

    void testSerialSimulatorShouldCallBackWriter()
    {
        MockWriter *w = new MockWriter(simulator);
        simulator->run();
        
        std::string expectedEvents = "initialized()\n";
        for (unsigned i = 1; i <= init->maxSteps(); i++) 
            expectedEvents += 
                "stepFinished(step=" + StringConv::itoa(i) + ")\n";        
        expectedEvents += "allDone()\n";

        TS_ASSERT_EQUALS(expectedEvents, w->events());
    }

    void testRunMustResetGridPriorToSimulation()
    {
        MockWriter *eventWriter1 = new MockWriter(simulator);
        MemoryWriter<TestCell<2> > *gridWriter1 = 
            new MemoryWriter<TestCell<2> >(simulator);
        simulator->run();
        std::string events1 = eventWriter1->events();
        std::vector<Grid<TestCell<2> > > grids1 = gridWriter1->getGrids();

        MockWriter *eventWriter2 = new MockWriter(simulator);
        MemoryWriter<TestCell<2> > *gridWriter2 = 
            new MemoryWriter<TestCell<2> >(simulator);
        simulator->run();
        std::string events2 = eventWriter2->events();
        std::vector<Grid<TestCell<2> > > grids2 = gridWriter2->getGrids();

        TS_ASSERT_EQUALS(events1, events2);
        TS_ASSERT_EQUALS(grids1, grids2);
    }

    typedef Grid<TestCell<3>, TestCell<3>::Topology> Grid3D;

    void test3D()
    {
        SerialSimulator<TestCell<3> > sim(new TestInitializer<3>());
        TS_ASSERT_TEST_GRID(Grid3D, *sim.getGrid(), 0);

        sim.step();
        TS_ASSERT_TEST_GRID(Grid3D, *sim.getGrid(), 
                            TestCell<3>::nanoSteps());

        sim.nanoStep(0);
        TS_ASSERT_TEST_GRID(Grid3D, *sim.getGrid(), 
                            TestCell<3>::nanoSteps() + 1);

        sim.run();
        TS_ASSERT_TEST_GRID(Grid3D, *sim.getGrid(), 
                            21 * TestCell<3>::nanoSteps());
    }
};

};
