#include <boost/shared_ptr.hpp>
#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/io/teststeerer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class TestSteererTest : public CxxTest::TestSuite 
{
public:
    typedef TestSteerer<2> SteererType;

    void setUp() 
    {
        simulator.reset(
            new SerialSimulator<TestCell<2> >(
                new TestInitializer<2>(
                    Coord<2>(10, 20),
                    34,
                    10)));
        simulator->addSteerer(new SteererType(5, 15, 4 * 27));
    }
    
    void tearDown()
    {
        simulator.reset();
    }

    void testCycleJump()
    {
        simulator->run();

        TS_ASSERT_TEST_GRID(SerialSimulator<TestCell<2> >::GridType, 
                            *simulator->getGrid(),
                            (34 + 4) * 27);
                            
    }

private:
    boost::shared_ptr<SerialSimulator<TestCell<2> > > simulator;
};

}
