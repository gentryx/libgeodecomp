#include <cxxtest/TestSuite.h>

#include <libgeodecomp.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator/multicorestepper.h>
#include <libgeodecomp/storage/mockpatchaccepter.h>

using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

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
    boost::shared_ptr<TestInitializer<TestCell<2> > > init;
    boost::shared_ptr<PartitionManager<Topology> > partitionManager;
#ifdef LIBGEODECOMP_WITH_THREADS
    boost::shared_ptr<StepperType> stepper;
#endif
    boost::shared_ptr<MockPatchAccepter<GridType> > patchAccepter;
};

}
}
