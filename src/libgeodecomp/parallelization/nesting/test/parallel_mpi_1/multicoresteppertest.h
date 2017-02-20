#include <cxxtest/TestSuite.h>

#include <libgeodecomp.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/nesting/multicorestepper.h>
#include <libgeodecomp/storage/mockpatchaccepter.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MulticoreStepperTest : public CxxTest::TestSuite
{
public:
    typedef APITraits::SelectTopology<TestCell<2> >::Value Topology;
    typedef DisplacedGrid<TestCell<2>, Topology, true> GridType;
#ifdef LIBGEODECOMP_WITH_THREADS
    typedef MultiCoreStepper<TestCell<2> > StepperType;
#endif

    void setUp()
    {
        init.reset(new TestInitializer<TestCell<2> >(Coord<2>(17, 12)));
        CoordBox<2> rect = init->gridBox();

        patchAccepter.reset(new MockPatchAccepter<GridType>());
        patchAccepter->pushRequest(2);
        patchAccepter->pushRequest(10);
        patchAccepter->pushRequest(13);

        partitionManager.reset(new PartitionManager<Topology>(rect));
        // stepper.reset(
        //     new StepperType(partitionManager, init));

        // stepper->addPatchAccepter(patchAccepter, StepperType::GHOST);
    }

    void testFoo()
    {
#ifdef LIBGEODECOMP_WITH_THREADS
        std::cout << "fixme\n";
#endif
    }

private:
    SharedPtr<TestInitializer<TestCell<2> > >::Type init;
    SharedPtr<PartitionManager<Topology> >::Type partitionManager;
#ifdef LIBGEODECOMP_WITH_THREADS
    SharedPtr<StepperType>::Type stepper;
#endif
    SharedPtr<MockPatchAccepter<GridType> >::Type patchAccepter;
};

}
