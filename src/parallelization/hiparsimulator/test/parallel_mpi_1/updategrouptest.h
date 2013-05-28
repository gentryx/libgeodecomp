#include <deque>
#include <fstream>
#include <cerrno>
#include <boost/assign/std/vector.hpp>

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/hiparsimulator/mockpatchaccepter.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/zcurvepartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/updategroup.h>

using namespace boost::assign;
using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

class UpdateGroupTest : public CxxTest::TestSuite
{
public:
    typedef ZCurvePartition<3> PartitionType;
    typedef VanillaStepper<TestCell<3> > StepperType;
    typedef UpdateGroup<TestCell<3>, StepperType> UpdateGroupType;
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
                init));
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
        std::deque<long> expected;
        expected.push_back(5);
        expected.push_back(7);
        expected.push_back(8);
        std::deque<long> actual = mockPatchAccepter->getOfferedNanoSteps();
        TS_ASSERT_EQUALS(actual, expected);
        TS_ASSERT_EQUALS(10, updateGroup->grid()[Coord<3>(1, 2, 3)].cycleCounter);
    }

private:
    unsigned rank;
    Coord<3> dimensions;
    SuperVector<long> weights;
    unsigned ghostZoneWidth;
    boost::shared_ptr<PartitionType> partition;
    boost::shared_ptr<Initializer<TestCell<3> > > init;
    boost::shared_ptr<UpdateGroup<TestCell<3> > > updateGroup;
    boost::shared_ptr<MockPatchAccepter<GridType> > mockPatchAccepter;
};

}
}
