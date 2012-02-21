#include <fstream>
#include <iostream>

#include "../../../../io/testinitializer.h"
#include "../../../../misc/testcell.h"
#include "../../mockpatchaccepter.h"
#include "../../partitions/zcurvepartition.h"
#include "../../updategroup.h"

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class UpdateGroupTest : public CxxTest::TestSuite
{
public:
    typedef ZCurvePartition<2> Partition;
    typedef VanillaStepper<TestCell<2> > MyStepper;
    typedef UpdateGroup<TestCell<2>, Partition, MyStepper> MyUpdateGroup;
    typedef MyStepper::GridType GridType;

    void setUp()
    {
        rank = MPILayer().rank();
        dimensions = Coord<2>(231, 350);
        partition = Partition(Coord<2>(), dimensions);
        weights = genWeights(dimensions.x(), dimensions.y(), MPILayer().size());
        ghostZoneWidth = 10;
        init.reset(new TestInitializer<2>(dimensions));
        updateGroup.reset(
            new MyUpdateGroup(
                Partition(Coord<2>(), dimensions),
                weights,
                0,
                CoordBox<2>(Coord<2>(), dimensions),
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
        init.reset();
        updateGroup.reset();
    }

    void testBench()
    {
        updateGroup->update(9);
    }
    
private:
    unsigned rank;
    Coord<2> dimensions;
    SuperVector<unsigned> weights;
    Partition partition;
    unsigned ghostZoneWidth;
    boost::shared_ptr<Initializer<TestCell<2> > > init;
    boost::shared_ptr<UpdateGroup<TestCell<2>, Partition > > updateGroup;
    boost::shared_ptr<MockPatchAccepter<GridType> > mockPatchAccepter;

    SuperVector<unsigned> genWeights(
        const unsigned& width, 
        const unsigned& height, 
        const unsigned& size)
    {
        SuperVector<unsigned> ret(size);
        unsigned totalSize = width * height;
        for (int i = 0; i < ret.size(); ++i) 
            ret[i] = pos(i+1, ret.size(), totalSize) - pos(i, ret.size(), totalSize);
        return ret;            
    }

    unsigned pos(const unsigned& i, const unsigned& size, const unsigned& totalSize)
    {
        return i * totalSize / size;
    }
};

}
}
