#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator/mockpatchaccepter.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>

using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

class VanillaStepperBasicTest : public CxxTest::TestSuite
{
public:
    typedef DisplacedGrid<TestCell<2>, Topologies::Cube<2>::Topology, true> GridType;
    typedef VanillaStepper<TestCell<2> > StepperType;

    void setUp()
    {
        init.reset(new TestInitializer<TestCell<2> >(Coord<2>(17, 12)));
        CoordBox<2> rect = init->gridBox();

        patchAccepter.reset(new MockPatchAccepter<GridType>());
        patchAccepter->pushRequest(2);
        patchAccepter->pushRequest(10);
        patchAccepter->pushRequest(13);

        partitionManager.reset(new PartitionManager<2>(rect));
        stepper.reset(
            new StepperType(partitionManager, &*init));

        stepper->addPatchAccepter(patchAccepter, StepperType::GHOST);
    }

    void testUpdate()
    {
        TS_ASSERT_TEST_GRID(GridType, stepper->grid(), 0);
        stepper->update();
        TS_ASSERT_TEST_GRID(GridType, stepper->grid(), 1);
    }

    void testUpdateMultiple()
    {
        stepper->update(8);
        TS_ASSERT_TEST_GRID(GridType, stepper->grid(), 8);
        stepper->update(30);
        TS_ASSERT_TEST_GRID(GridType, stepper->grid(), 38);
    }

    void testPutPatch()
    {
        stepper->update(9);
        TS_ASSERT_EQUALS(2, patchAccepter->offeredNanoSteps.size());

        stepper->update(4);
        TS_ASSERT_EQUALS(3, patchAccepter->offeredNanoSteps.size());
    }

private:
    boost::shared_ptr<TestInitializer<TestCell<2> > > init;
    boost::shared_ptr<PartitionManager<2> > partitionManager;
    boost::shared_ptr<StepperType> stepper;
    boost::shared_ptr<MockPatchAccepter<GridType> > patchAccepter;
};

}
}
