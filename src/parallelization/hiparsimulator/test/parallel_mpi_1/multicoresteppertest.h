#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator/mockpatchaccepter.h>
#include <libgeodecomp/parallelization/hiparsimulator/multicorestepper.h>

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class MulticoreStepperTest : public CxxTest::TestSuite
{
public:

    typedef DisplacedGrid<TestCell<3>, TestCell<3>::Topology, true> GridType;
    typedef MulticoreStepper<TestCell<3> > MyStepper;

    void setUp()
    {
        init.reset(new TestInitializer<TestCell<2> >(Coord<2>(17, 12)));
        CoordBox<2> rect = init->gridBox();

        patchAccepter.reset(new MockPatchAccepter<GridType>());
        patchAccepter->pushRequest(2);
        patchAccepter->pushRequest(10);
        patchAccepter->pushRequest(13);
   
        partitionManager.reset(new PartitionManager<2>(rect));
        // stepper.reset(
        //     new MyStepper(partitionManager, init));

        // stepper->addPatchAccepter(patchAccepter, MyStepper::GHOST);
    }

    void testFoo()
    {
        std::cout << "fixme\n";
    }

private:
    boost::shared_ptr<TestInitializer<TestCell<2> > > init;
    boost::shared_ptr<PartitionManager<2> > partitionManager;
    boost::shared_ptr<MyStepper> stepper;
    boost::shared_ptr<MockPatchAccepter<GridType> > patchAccepter;
};

}
}
