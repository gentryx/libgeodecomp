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

// fixme: remove this test

// template<class CELL_TYPE>
// class MockPatchProvider : public PatchProvider<CELL_TYPE>
// {
// public:
//     virtual void get(DisplacedGrid<CELL_TYPE> *destinationGrid, const Region& region, const unsigned& nanoStep)
//     {
//         if (storedRegions.front() != region) {
//             Region delta = (storedRegions.front() - region) + (region - storedRegions.front());
//             throw std::logic_error("requested region doesn't match stored region, delta is: " + delta.toString());
//         }
//         if (storedNanoSteps.front() != nanoStep) 
//             throw std::logic_error("requested time step doesn't match expected nano step");
//         GridVecConv::vectorToGrid(storage.front(), destinationGrid, region);
//         storage.pop_front();
//         storedRegions.pop_front();
//         storedNanoSteps.pop_front();
//     }

//     template<class GRID_TYPE>
//     void push(const GRID_TYPE& grid, const Region& region, const unsigned& nanoStep)
//     {
//         storedRegions.push_back(region);
//         storedNanoSteps.push_back(nanoStep);
//         storage.push_back(GridVecConv::gridToVector(grid, region));
//     }

// private:
//     std::deque<SuperVector<CELL_TYPE> > storage;
//     std::deque<Region> storedRegions;
//     std::deque<unsigned> storedNanoSteps;
// };

// template<class CELL_TYPE>
// class MockPatchAccepter : public PatchAccepter<CELL_TYPE>
// {
// public:
//     virtual void put(const DisplacedGrid<CELL_TYPE>& grid, const Region& validRegion, const unsigned& nanoStep)
//     {
//         if (storedNanoSteps.empty() || nanoStep < storedNanoSteps.front())
//             return;
//         if (nanoStep > storedNanoSteps.front()) 
//             throw std::logic_error("expected nano step was left out");
//         if (!(storedRegions.front() - validRegion).empty()) 
//             throw std::logic_error("validRegion is not a super set of the expected region");
//         storage.push_back(GridVecConv::gridToVector(grid, storedRegions.front()));
//         storedNanoSteps.pop_front();
//         storedRegions.pop_front();
//     }

//     void push(const Region& region, const unsigned& nanoStep)
//     {
//         storedRegions.push_back(region);
//         storedNanoSteps.push_back(nanoStep);
//     }

// private:
//     std::deque<SuperVector<CELL_TYPE> > storage;
//     std::deque<Region> storedRegions;
//     std::deque<unsigned> storedNanoSteps;
// };

class UpdateGroupTest : public CxxTest::TestSuite
{
public:
//     typedef ZCurvePartition Partition;
//     typedef PartitionManager<IntersectingRegionAccumulator<Partition> >::RegionVecMap RegionVecMap;
//     typedef PartitionManager<RegionAccumulator<Partition> > PartitionManager;
    
    void setUp()
    {
//         rank = MPILayer().rank();
//         dimensions = Coord(280, 200);
//         partition = Partition(Coord(0, 0), dimensions);
//         nodeGhostZoneWidth = 10;
//         clusterGhostZoneWidth = 33;

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

// private:
//     boost::shared_ptr<Initializer<TestCell> > init;
//     boost::shared_ptr<UpdateGroup<TestCell, Partition> > clusterGhostZoneGroup;   
//     boost::shared_ptr<UpdateGroup<TestCell, Partition, RegionAccumulator> > simulationAreaGroup;
//     boost::shared_ptr<SerialSimulator<TestCell> > serialSim;
//     boost::shared_ptr<MockPatchProvider<TestCell> > mockPatchProvider;
//     boost::shared_ptr<MockPatchAccepter<TestCell> > mockPatchAccepter;
//     Partition partition;
//     PartitionManager partitionManager;
//     Region clusterRegion;
//     Region innerClusterRim;
//     Region patchRegion;
//     Coord dimensions;
//     unsigned rank;
//     unsigned nodeGhostZoneWidth;
//     unsigned clusterGhostZoneWidth;
//     SuperVector<unsigned> weights;
//     SuperVector<CoordRectangle> boundingBoxes;
//     unsigned offset;
//     unsigned maxSteps;
//     unsigned firstStep;
//     unsigned firstNanoStep;
//     unsigned firstCycle;

//     TestInitializer testInit()
//     {
//         return TestInitializer(dimensions.x, dimensions.y, maxSteps, firstStep, firstNanoStep);
//     }

//     SuperVector<CoordRectangle> genBoundingBoxes(const Partition& partition, const unsigned& offset, const SuperVector<unsigned>& weights)
//     {
//         SuperVector<CoordRectangle> ret(weights.size());
//         unsigned currentOffset = offset;
//         for (int i = 0; i < weights.size(); ++i) {
//             Region s;
//             for (Partition::Iterator c = partition[currentOffset]; c != partition[currentOffset + weights[i]]; ++c)
//                 s << *c;
//             ret[i] = s.boundingBox();
//             currentOffset += weights[i];
//         }
//         return ret;
//     }
};

}
}
