#include <deque>
#include <fstream>
#include <cerrno>

#include <libgeodecomp.h>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/nesting/mpiupdategroup.h>
#include <libgeodecomp/storage/mockpatchaccepter.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MPIUpdateGroupTest : public CxxTest::TestSuite
{
public:
    typedef ZCurvePartition<3> PartitionType;
    typedef VanillaStepper<TestCell<3>, UpdateFunctorHelpers::ConcurrencyEnableOpenMP> StepperType;
    typedef MPIUpdateGroup<TestCell<3> > UpdateGroupType;
    typedef StepperType::GridType GridType;

    void setUp()
    {
        rank = MPILayer().rank();
        dimensions = Coord<3>(28, 50, 32);
        weights.clear();
        weights << dimensions.prod();
        partition.reset(new PartitionType(Coord<3>(), dimensions, 0, weights));
        ghostZoneWidth = 10;
        init.reset(new TestInitializer<TestCell<3> >(dimensions));
        updateGroup.reset(
            new UpdateGroupType(
                partition,
                CoordBox<3>(Coord<3>(), dimensions),
                ghostZoneWidth,
                init,
                reinterpret_cast<StepperType*>(0)));
        mockPatchAccepter.reset(new MockPatchAccepter<GridType>());
        mockPatchAccepter->pushRequest(5);
        mockPatchAccepter->pushRequest(7);
        mockPatchAccepter->pushRequest(8);
        updateGroup->addPatchAccepter(mockPatchAccepter, StepperType::INNER_SET);
    }

    void tearDown()
    {
        updateGroup.reset();
    }

    void testBasic()
    {
        updateGroup->update(10);
        std::deque<std::size_t> expected;
        expected.push_back(5);
        expected.push_back(7);
        expected.push_back(8);
        std::deque<std::size_t> actual = mockPatchAccepter->getOfferedNanoSteps();
        TS_ASSERT_EQUALS(actual, expected);
        TS_ASSERT_EQUALS(static_cast<unsigned>(10), updateGroup->grid()[Coord<3>(1, 2, 3)].cycleCounter);
    }

private:
    unsigned rank;
    Coord<3> dimensions;
    std::vector<std::size_t> weights;
    unsigned ghostZoneWidth;
    SharedPtr<PartitionType>::Type partition;
    SharedPtr<Initializer<TestCell<3> > >::Type init;
    SharedPtr<MPIUpdateGroup<TestCell<3> > >::Type updateGroup;
    SharedPtr<MockPatchAccepter<GridType> >::Type mockPatchAccepter;
};

}
