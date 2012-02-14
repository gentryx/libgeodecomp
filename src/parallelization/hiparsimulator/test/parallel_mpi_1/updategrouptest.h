#include <deque>
#include <fstream>
#include <cerrno>
#include <boost/assign/std/vector.hpp>
#include "../../../../io/image.h"
#include "../../../../io/ioexception.h"
#include "../../../../io/testinitializer.h"
#include "../../../../misc/testcell.h"
#include "../../partitions/zcurvepartition.h"
#include "../../updategroup.h"
#include "../../../serialsimulator.h"

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

    void setUp()
    {
        rank = MPILayer().rank();
        dimensions = Coord<3>(280, 50, 32);
        partition = Partition(Coord<3>(), dimensions);
        weights.clear();
        weights << 0;
        weights << dimensions.prod();
        ghostZoneWidth = 10;
        init.reset(new TestInitializer<3>(dimensions));
        updateGroup.reset(
            new MyUpdateGroup(
                Partition(Coord<3>(), dimensions),
                weights,
                0,
                CoordBox<3>(Coord<3>(), dimensions),
                ghostZoneWidth,
                init));
                
                              

// fixme: remove dead code
//         maxSteps = 1500;
//         firstStep = 20;
//         firstNanoStep = 18;
//         firstCycle = firstStep * TestCell::nanoSteps() + firstNanoStep;
//         init.reset(new TestInitializer(testInit()));

//         weights.clear();
//         unsigned totalSize = dimensions.x * dimensions.y;
//         unsigned secondToLastNodeWeight = totalSize / 11;
//         unsigned lastNodeWeight = totalSize / 88;
//         unsigned numNormalNodes = 7;
//         unsigned normalNodeWeight = (totalSize / 2 - lastNodeWeight - secondToLastNodeWeight) / numNormalNodes;
//         for (int i = 0; i < numNormalNodes; ++i)
//             weights += normalNodeWeight;
//         weights += secondToLastNodeWeight, lastNodeWeight;
//         offset = totalSize / 7;

//         RegionAccumulator<Partition> regionAccu(partition, offset, weights);
//         partitionManager.resetRegions(
//             CoordRectangle(Coord(0, 0), dimensions.x, dimensions.y),
//             regionAccu,
//             rank,
//             nodeGhostZoneWidth);
//         boundingBoxes = genBoundingBoxes(partition, offset, weights);
//         partitionManager.resetGhostZones(boundingBoxes);

//         clusterRegion = Region(partition[offset], partition[offset + weights.sum()]);
//         Region outerClusterRim = clusterRegion.expand() - clusterRegion;
//         innerClusterRim = outerClusterRim.expand(clusterGhostZoneWidth) & clusterRegion;

//         mockPatchProvider.reset(new MockPatchProvider<TestCell>());
//         mockPatchAccepter.reset(new MockPatchAccepter<TestCell>());

//         clusterGhostZoneGroup.reset(
//             new UpdateGroup<TestCell, Partition>(
//                 innerClusterRim,
//                 partition, 
//                 weights, 
//                 offset, 
//                 CoordRectangle(Coord(0, 0), dimensions.x, dimensions.y),
//                 clusterGhostZoneWidth,
//                 &*init,
//                 &*mockPatchProvider,
//                 &*mockPatchAccepter));
        
//         // fixme: rename simulationAreaGroup and clusterGhostZoneGroup to "level x" or "node level", "cluster level" etc. and add a ascii draft
//         simulationAreaGroup.reset(
//             new UpdateGroup<TestCell, Partition, RegionAccumulator>(
//                 partition,
//                 weights, 
//                 offset, 
//                 CoordRectangle(Coord(0, 0), dimensions.x, dimensions.y),
//                 nodeGhostZoneWidth,
//                 &*init));
                
//         serialSim.reset(new SerialSimulator<TestCell>(new TestInitializer(testInit())));
//         patchRegion = simulationAreaGroup->partitionManager.getOuterOutgroupGhostZoneFragment() &
//             clusterGhostZoneGroup->partitionManager.ownRegion(nodeGhostZoneWidth);
            
    }

    void tearDown()
    {
//         clusterGhostZoneGroup.reset();
//         simulationAreaGroup.reset();
//         serialSim.reset();
    }

    void testBasic()
    {
//         int jumpLength = clusterGhostZoneWidth;
//         int numJumps = 5;
//         for (int jump = 0; jump < numJumps; ++jump) {
//             for (int i = 0; i < jumpLength; ++i) 
//                 serialSim->nanoStep((firstCycle + jump * jumpLength + i) % TestCell::nanoSteps());
//             mockPatchProvider->push(*serialSim->getGrid(), clusterGhostZoneGroup->partitionManager.getOuterOutgroupGhostZoneFragment(), firstCycle + (jump + 1) * jumpLength);
//         }

//         for (int jump = 0; jump < numJumps; ++jump) 
//             mockPatchAccepter->push(patchRegion, firstCycle + jump * jumpLength);

//         for (int jump = 0; jump < numJumps; ++jump) 
//             clusterGhostZoneGroup->nanoStep(jumpLength, jumpLength);

// //         for (int jump = 0; jump < numJumps; ++jump) 
//         simulationAreaGroup->nanoStep(4, 4);

//         // fixme: check events on mockpatchprovider/accepter
    }

private:
    unsigned rank;
    Coord<3> dimensions;
    SuperVector<unsigned> weights;
    Partition partition;
    unsigned ghostZoneWidth;
    boost::shared_ptr<Initializer<TestCell<3> > > init;
    boost::shared_ptr<UpdateGroup<TestCell<3>, Partition > > updateGroup;
};

}
}
