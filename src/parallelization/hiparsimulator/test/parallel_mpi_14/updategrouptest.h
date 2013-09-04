#include <fstream>
#include <cerrno>
#include <boost/assign/std/vector.hpp>
#include <libgeodecomp/io/image.h>
#include <libgeodecomp/io/ioexception.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/zcurvepartition.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchbuffer.h>
#include <libgeodecomp/parallelization/hiparsimulator/updategroup.h>

using namespace boost::assign;
using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

// This is an old test code for setting up a hierarchical domain
// decomposition, suitable for multi-cluster runs with a high-latency
// link in betweeen them. It's being kept for reference.
class UpdateGroupTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
    //     rank = MPILayer().rank();
    //     dimensions = Coord(280, 200);
    //     partition = Partition(Coord(0, 0), dimensions);
    //     level0GhostZoneWidth = 9;
    //     level1GhostZoneWidth = 22;
    //     level2GhostZoneWidth = 30;

    //     maxSteps = 1500;
    //     firstStep = 20;
    //     firstNanoStep = 18;
    //     firstCycle = firstStep * APITraits::SelectNanoSteps<TestCell<2> >::VALUE + firstNanoStep;
    //     init.reset(new TestInitializer(buildTestInit()));


    //     // Rank tree structure: (goal)
    //     //
    //     //  level2        0-----2---------12
    //     //                |     |          |
    //     //  level1 (a-c)  0   1-2-11      12-13
    //     //                      |
    //     //  level0              2-3-4-5-6-7-8-9-10
    //     //

    //     SuperVector<int> level2Ranks;
    //     level2Ranks += 0, 2, 12;

    //     SuperVector<int> level1aRanks;
    //     level1aRanks += 0;

    //     SuperVector<int> level1bRanks;
    //     level1bRanks += 1, 2, 11;

    //     SuperVector<int> level1cRanks;
    //     level1cRanks += 12, 13;

    //     SuperVector<int> level0Ranks;
    //     level0Ranks += 2, 3, 4, 5, 6, 7, 8, 9, 10;

    //     level2MPIGroup = MPI_COMM_WORLD.Get_group().Incl(3, &level2Ranks[0]);
    //     level2MPIComm  = MPI_COMM_WORLD.Create(level2MPIGroup);

    //     level1aMPIGroup = MPI_COMM_WORLD.Get_group().Incl(1, &level1aRanks[0]);
    //     level1aMPIComm  = MPI_COMM_WORLD.Create(level1aMPIGroup);

    //     level1bMPIGroup = MPI_COMM_WORLD.Get_group().Incl(3, &level1bRanks[0]);
    //     level1bMPIComm  = MPI_COMM_WORLD.Create(level1bMPIGroup);

    //     level1cMPIGroup = MPI_COMM_WORLD.Get_group().Incl(2, &level1cRanks[0]);
    //     level1cMPIComm  = MPI_COMM_WORLD.Create(level1cMPIGroup);

    //     level0MPIGroup = MPI_COMM_WORLD.Get_group().Incl(9, &level0Ranks[0]);
    //     level0MPIComm  = MPI_COMM_WORLD.Create(level0MPIGroup);

    //     SuperVector<unsigned> weights;
    //     unsigned offset;

    //     genLevel2WeightsAndOffset(&weights, &offset);

    //     if (level2Ranks.contains(rank)) {
    //         patchBuffer.reset(new PatchBuffer<
    //                           DisplacedGrid<TestCell>,
    //                           DisplacedGrid<TestCell>,
    //                           TestCell>());

    //         level2CoordinationGroup.reset(
    //             new UpdateGroup<TestCell, Partition>(
    //                 partition,
    //                 weights,
    //                 offset,
    //                 CoordRectangle(Coord(0, 0), dimensions),
    //                 level2GhostZoneWidth,
    //                 init,
    //                 0,
    //                 0,
    //                 &level2MPIComm));

    //         SuperVector<unsigned> newWeights;
    //         newWeights += weights[level2MPIComm.Get_rank()];
    //         unsigned newOffset = offset;
    //         for (int i = 0; i < level2MPIComm.Get_rank(); ++i)
    //             newOffset += weights[i];

    //         level0Group.reset(
    //             new UpdateGroup<TestCell, Partition>(
    //                 partition,
    //                 newWeights,
    //                 newOffset,
    //                 CoordRectangle(Coord(0, 0), dimensions),
    //                 level0GhostZoneWidth,
    //                 init,
    //                 &*patchBuffer,
    //                 0,
    //                 &MPI_COMM_SELF));

    //         level2Group.reset(
    //             new UpdateGroup<TestCell, Partition>(
    //                 level2CoordinationGroup->partitionManager.rim(level2GhostZoneWidth),
    //                 partition,
    //                 newWeights,
    //                 newOffset,
    //                 CoordRectangle(Coord(0, 0), dimensions),
    //                 level2GhostZoneWidth,
    //                 init,
    //                 0,
    //                 &*patchBuffer,
    //                 &MPI_COMM_SELF));

    //         level2GhostZoneResolution.reset(
    //             new GhostZoneResolution(
    //                 level2Group->partitionManager.getOuterOutgroupGhostZoneFragment(),
    //                 Region(),
    //                 level0Group->partitionManager));
    //     }
    // }

    // void tearDown()
    // {
    //     level0Group.reset();
    //     level1Group.reset();
    //     level2Group.reset();
    //     level2CoordinationGroup.reset();
    }

    void testBasic()
    {
//         std::cout << "testBasic\n";
//         if (level0Group) {
//             long long start = Chronometer::timeUSec();
//             for (int superCycle = 0; superCycle < 3; ++superCycle) {
//                 std::cout << "superCycle: " << superCycle << "\n";

//                 // load events and update
//                 for (int i = 0; i < level2GhostZoneWidth; i += level0GhostZoneWidth)
//                     patchBuffer->pushRequest(&level0Group->partitionManager.getOuterOutgroupGhostZoneFragment(), firstCycle + superCycle * level2GhostZoneWidth + i);
//                 level2Group->nanoStep(level2GhostZoneWidth, level2GhostZoneWidth);
//                 level0Group->nanoStep(level2GhostZoneWidth, level2GhostZoneWidth);

//                 // collect level2 ghost zone
//                 level2CoordinationGroup->getNewGrid()->paste(*level2Group->getGrid(), level2Group->partitionManager.ownRegion());
//                 // transmit level2 ghost zone
//                 level2CoordinationGroup->stepper.sendGhostZones(level2GhostZoneWidth);
//                 level2CoordinationGroup->stepper.recvGhostZones(level2GhostZoneWidth);
//                 level2CoordinationGroup->stepper.waitForGhostZones();
//                 // distribute level2 ghost zone
//                 level2Group->getGrid()->paste(*level2CoordinationGroup->getNewGrid(), level2GhostZoneResolution->fromMaster);

//                 // paste from level0 group
//                 level2Group->getGrid()->paste(*level0Group->getGrid(),                level2GhostZoneResolution->fromHost);

// //                 //             DisplacedGrid<TestCell> level2GhostZone(level2CoordinationGroup->partitionManager.ownExpandedRegion().boundingBox());


// //                 //             DisplacedGrid<TestCell> patch(level2Group->partitionManager.getOuterOutgroupGhostZoneFragment().boundingBox());
// //                 //             patch.paste(*level0Group->getGrid(), level2GhostZoneResolution->fromMaster());

//                 long long end = Chronometer::timeUSec();
//                 if (rank == 0)
//                     std::cout << "time: " << (end-start) << "\n";


// //         int jumpLength = clusterGhostZoneWidth;
// //         int numJumps = 5;
// //         //             level2Group->nanoStep(1, 1);
// //         //             level0Group->nanoStep(10, 10);
// //         //}
//             }
//         }
// //     //         int jumpLength = clusterGhostZoneWidth;
// //     //         int numJumps = 5;

// //     //         for (int jump = 0; jump < numJumps; ++jump)
// //     //             clusterGhostZoneGroup->nanoStep(jumpLength, jumpLength);

// //     // //         for (int jump = 0; jump < numJumps; ++jump)
// //     //         simulationAreaGroup->nanoStep(4, 4);

// //     // fixme: check events on mockpatchprovider/accepter
//     std::cout << "testBasic2\n";
    }

// private:
//     boost::shared_ptr<Initializer<TestCell> > init;
//     boost::shared_ptr<UpdateGroup<TestCell, Partition> > level0Group; // node level
//     boost::shared_ptr<UpdateGroup<TestCell, Partition> > level1Group; // cluster level
//     boost::shared_ptr<UpdateGroup<TestCell, Partition> > level2Group; // meta cluster level
//     boost::shared_ptr<UpdateGroup<TestCell, Partition> > level2CoordinationGroup;
//     boost::shared_ptr<
//         PatchBuffer<DisplacedGrid<TestCell>,
//                     DisplacedGrid<TestCell>,
//                     TestCell> > patchBuffer;
//     boost::shared_ptr<GhostZoneResolution> level2GhostZoneResolution;
//     Partition partition;
//     Region clusterRegion;
//     Region innerClusterRim;
//     Coord dimensions;
//     unsigned rank;
//     unsigned level0GhostZoneWidth;
//     unsigned level1GhostZoneWidth;
//     unsigned level2GhostZoneWidth;
//     MPI_Group level2MPIGroup;
//     MPI_Group level1aMPIGroup;
//     MPI_Group level1bMPIGroup;
//     MPI_Group level1cMPIGroup;
//     MPI_Group level0MPIGroup;
//     MPI_Intracomm level2MPIComm;
//     MPI_Intracomm level1aMPIComm;
//     MPI_Intracomm level1bMPIComm;
//     MPI_Intracomm level1cMPIComm;
//     MPI_Intracomm level0MPIComm;
//     unsigned offset;
//     unsigned maxSteps;
//     unsigned firstStep;
//     unsigned firstNanoStep;
//     unsigned firstCycle;

//     TestInitializer buildTestInit()
//     {
//         return TestInitializer(dimensions.x, dimensions.y, maxSteps, firstStep, firstNanoStep);
//     }

//     unsigned totalSize()
//     {
//         return dimensions.x * dimensions.y;
//     }

//     void genLevel2WeightsAndOffset(SuperVector<unsigned> *weights, unsigned *offset)
//     {
//         weights->clear();
//         *weights +=
//             totalSize() / 9,
//             7 * totalSize() / 12;
//         fillinRemainder(weights, totalSize());
//         *offset = 0;
//     }

//     void genLevel1AWeightsAndOffset(SuperVector<unsigned> *weights, unsigned *offset)
//     {
//         SuperVector<unsigned> superLevelWeights;
//         unsigned superLevelOffset;
//         genLevel2WeightsAndOffset(&superLevelWeights, &superLevelOffset);

//         weights->clear();
//         *weights += superLevelWeights[0];
//         *offset = superLevelOffset;
//     }

//     void genLevel1BWeightsAndOffset(SuperVector<unsigned> *weights, unsigned *offset)
//     {
//         SuperVector<unsigned> superLevelWeights;
//         unsigned superLevelOffset;
//         genLevel2WeightsAndOffset(&superLevelWeights, &superLevelOffset);

//         weights->clear();
//         *weights +=
//             totalSize() / 7 - superLevelWeights[0],
//             totalSize() / 2;
//         fillinRemainder(weights, superLevelWeights[1]);
//         *offset = superLevelOffset + superLevelWeights[0];
//     }

//     void genLevel1CWeightsAndOffset(SuperVector<unsigned> *weights, unsigned *offset)
//     {
//         SuperVector<unsigned> superLevelWeights;
//         unsigned superLevelOffset;
//         genLevel2WeightsAndOffset(&superLevelWeights, &superLevelOffset);

//         weights->clear();
//         *weights += superLevelWeights[2] / 2;
//         fillinRemainder(weights, superLevelWeights[2]);
//         *offset = superLevelOffset + superLevelWeights[0] + superLevelWeights[1];
//     }

//     void genLevel0WeightsAndOffset(SuperVector<unsigned> *weights, unsigned *offset)
//     {
//         SuperVector<unsigned> superLevelWeights;
//         unsigned superLevelOffset;
//         genLevel1BWeightsAndOffset(&superLevelWeights, &superLevelOffset);

//         weights->clear();
//         unsigned secondToLastNodeWeight = totalSize() / 11;
//         unsigned lastNodeWeight = totalSize() / 88;
//         unsigned numNormalNodes = 7;
//         unsigned normalNodeWeight = (superLevelWeights[1] - lastNodeWeight - secondToLastNodeWeight) / numNormalNodes;
//         for (int i = 0; i < numNormalNodes; ++i)
//             *weights += normalNodeWeight;
//         *weights += secondToLastNodeWeight, lastNodeWeight;
//         *offset = superLevelOffset + superLevelWeights[0];
//     }

//     void fillinRemainder(SuperVector<unsigned> *vec, const unsigned& size)
//     {
//         vec->push_back(size - vec->sum());
//     }

};

}
}
