#include <cxxtest/TestSuite.h>

#include <libgeodecomp.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator/vanillastepper.h>
#include <libgeodecomp/storage/mockpatchaccepter.h>

using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

class VanillaStepperBasicTest : public CxxTest::TestSuite
{
public:
    typedef DisplacedGrid<TestCell<2>, Topologies::Cube<2>::Topology, true> GridType;
    typedef VanillaStepper<TestCell<2>, UpdateFunctorHelpers::ConcurrencyNoP> StepperType;

    void setUp()
    {
        init.reset(new TestInitializer<TestCell<2> >(Coord<2>(17, 12)));
        CoordBox<2> rect = init->gridBox();

        patchAccepter.reset(new MockPatchAccepter<GridType>());
        patchAccepter->pushRequest(2);
        patchAccepter->pushRequest(10);
        patchAccepter->pushRequest(13);

        partitionManager.reset(new PartitionManager<Topologies::Cube<2>::Topology>(rect));
        stepper.reset(
            new StepperType(partitionManager, init));

        stepper->addPatchAccepter(patchAccepter, StepperType::GHOST);
    }

    void testUpdate1()
    {
        TS_ASSERT_TEST_GRID(GridType, stepper->grid(), 0);
        stepper->update1();
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
        TS_ASSERT_EQUALS(std::size_t(2), patchAccepter->getOfferedNanoSteps().size());

        stepper->update(4);
        TS_ASSERT_EQUALS(std::size_t(3), patchAccepter->getOfferedNanoSteps().size());
    }

private:
    boost::shared_ptr<TestInitializer<TestCell<2> > > init;
    boost::shared_ptr<PartitionManager<Topologies::Cube<2>::Topology> > partitionManager;
    boost::shared_ptr<StepperType> stepper;
    boost::shared_ptr<MockPatchAccepter<GridType> > patchAccepter;
};

}
}
