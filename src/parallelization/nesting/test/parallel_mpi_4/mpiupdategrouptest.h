#include <fstream>
#include <iostream>
#include <cxxtest/TestSuite.h>

#include <libgeodecomp.h>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/nesting/mpiupdategroup.h>
#include <libgeodecomp/storage/mockpatchaccepter.h>

#include <boost/assign/std/deque.hpp>

using namespace LibGeoDecomp;
using namespace boost::assign;

namespace LibGeoDecomp {

class MPIUpdateGroupTest : public CxxTest::TestSuite
{
public:
    typedef ZCurvePartition<2> PartitionType;
    typedef VanillaStepper<TestCell<2>, UpdateFunctorHelpers::ConcurrencyNoP> StepperType;
    typedef MPIUpdateGroup<TestCell<2> > UpdateGroupType;
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
                init,
                reinterpret_cast<StepperType*>(0)));
        expectedNanoSteps.clear();
        expectedNanoSteps += 5, 7, 8, 33, 55;
        mockPatchAccepter.reset(new MockPatchAccepter<GridType>());
        for (std::deque<std::size_t>::iterator i = expectedNanoSteps.begin();
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

    void testUpdate()
    {
        updateGroup->update(100);
        std::deque<std::size_t> actualNanoSteps = mockPatchAccepter->getOfferedNanoSteps();
        TS_ASSERT_EQUALS(actualNanoSteps, expectedNanoSteps);
    }

private:
    std::deque<std::size_t> expectedNanoSteps;
    unsigned rank;
    Coord<2> dimensions;
    std::vector<std::size_t> weights;
    unsigned ghostZoneWidth;
    boost::shared_ptr<PartitionType> partition;
    boost::shared_ptr<Initializer<TestCell<2> > > init;
    boost::shared_ptr<MPIUpdateGroup<TestCell<2> > > updateGroup;
    boost::shared_ptr<MockPatchAccepter<GridType> > mockPatchAccepter;

    std::vector<std::size_t> genWeights(
        unsigned width,
        unsigned height,
        unsigned size)
    {
        std::vector<std::size_t> ret(size);
        unsigned totalSize = width * height;
        for (std::size_t i = 0; i < ret.size(); ++i) {
            ret[i] = pos(i+1, ret.size(), totalSize) - pos(i, ret.size(), totalSize);
        }
        return ret;
    }

    long pos(unsigned i, unsigned size, unsigned totalSize)
    {
        return i * totalSize / size;
    }
};

}
