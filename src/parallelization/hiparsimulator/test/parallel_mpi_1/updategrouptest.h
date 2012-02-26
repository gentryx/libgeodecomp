#include <deque>
#include <fstream>
#include <cerrno>
#include <boost/assign/std/vector.hpp>

#include "../../../../io/testinitializer.h"
#include "../../../../misc/testcell.h"
#include "../../mockpatchaccepter.h"
#include "../../partitions/zcurvepartition.h"
#include "../../updategroup.h"

using namespace boost::assign;
using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class UpdateGroupTest : public CxxTest::TestSuite
{
public:
    typedef ZCurvePartition<3> Partition;
    typedef VanillaStepper<TestCell<3> > MyStepper;
    typedef UpdateGroup<TestCell<3>, Partition, MyStepper> MyUpdateGroup;
    typedef MyStepper::GridType GridType;

    void setUp()
    {
        rank = MPILayer().rank();
        dimensions = Coord<3>(28, 50, 32);
        partition = Partition(Coord<3>(), dimensions);
        weights.clear();
        weights << dimensions.prod();
        ghostZoneWidth = 10;
        init = new TestInitializer<3>(dimensions);
        updateGroup.reset(
            new MyUpdateGroup(
                partition,
                weights,
                0,
                CoordBox<3>(Coord<3>(), dimensions),
                ghostZoneWidth,
                init));
        mockPatchAccepter.reset(new MockPatchAccepter<GridType>());
        mockPatchAccepter->pushRequest(5);
        mockPatchAccepter->pushRequest(7);
        mockPatchAccepter->pushRequest(8);
        updateGroup->addPatchAccepter(mockPatchAccepter, MyStepper::INNER_SET);
    }
                              
    void tearDown()
    {
        delete init;
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
    Partition partition;
    unsigned ghostZoneWidth;
    Initializer<TestCell<3> > *init;
    boost::shared_ptr<UpdateGroup<TestCell<3>, Partition > > updateGroup;
    boost::shared_ptr<MockPatchAccepter<GridType> > mockPatchAccepter;
};

}
}
