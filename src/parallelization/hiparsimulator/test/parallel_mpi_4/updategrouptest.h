#include <fstream>
#include <iostream>
#include <boost/assign/std/deque.hpp>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/hiparsimulator/mockpatchaccepter.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/zcurvepartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/updategroup.h>

using namespace LibGeoDecomp;
using namespace HiParSimulator;
using namespace boost::assign;

namespace LibGeoDecomp {
namespace HiParSimulator {

class UpdateGroupTest : public CxxTest::TestSuite
{
public:
    typedef ZCurvePartition<2> PartitionType;
    typedef VanillaStepper<TestCell<2> > StepperType;
    typedef UpdateGroup<TestCell<2>, StepperType> UpdateGroupType;
    typedef StepperType::GridType GridType;

    void setUp()
    {
        rank = MPILayer().rank();
        dimensions = Coord<2>(231, 350);
        weights = genWeights(dimensions.x(), dimensions.y(), MPILayer().size());
        partition.reset(new PartitionType(Coord<2>(), dimensions, 0, weights));
        ghostZoneWidth = 9;
        init.reset(new TestInitializer<TestCell<2> >(dimensions));
        updateGroup.reset(
            new UpdateGroupType(
                partition,
                CoordBox<2>(Coord<2>(), dimensions),
                ghostZoneWidth,
                init));
        expectedNanoSteps.clear();
        expectedNanoSteps += 5, 7, 8, 33, 55;
        mockPatchAccepter.reset(new MockPatchAccepter<GridType>());
        for (std::deque<long>::iterator i = expectedNanoSteps.begin();
             i != expectedNanoSteps.end();
             ++i) {
            mockPatchAccepter->pushRequest(*i);
        }
        updateGroup->addPatchAccepter(mockPatchAccepter, StepperType::INNER_SET);
    }

    void tearDown()
    {
        updateGroup.reset();
    }

    void testBench()
    {
        updateGroup->update(100);
        std::deque<long> actualNanoSteps = mockPatchAccepter->getOfferedNanoSteps();
        TS_ASSERT_EQUALS(actualNanoSteps, expectedNanoSteps);
    }

private:
    std::deque<long> expectedNanoSteps;
    unsigned rank;
    Coord<2> dimensions;
    SuperVector<long> weights;
    unsigned ghostZoneWidth;
    boost::shared_ptr<PartitionType> partition;
    boost::shared_ptr<Initializer<TestCell<2> > > init;
    boost::shared_ptr<UpdateGroup<TestCell<2> > > updateGroup;
    boost::shared_ptr<MockPatchAccepter<GridType> > mockPatchAccepter;

    SuperVector<long> genWeights(
        const unsigned& width,
        const unsigned& height,
        const unsigned& size)
    {
        SuperVector<long> ret(size);
        unsigned totalSize = width * height;
        for (int i = 0; i < ret.size(); ++i)
            ret[i] = pos(i+1, ret.size(), totalSize) - pos(i, ret.size(), totalSize);
        return ret;
    }

    long pos(const unsigned& i, const unsigned& size, const unsigned& totalSize)
    {
        return i * totalSize / size;
    }
};

}
}
