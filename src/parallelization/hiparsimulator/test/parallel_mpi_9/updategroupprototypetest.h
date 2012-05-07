#include <boost/assign/std/vector.hpp>
#include <cxxtest/TestSuite.h>
#include <cerrno>
#include <fstream>
#include "../../../../io/image.h"
#include "../../../../io/ioexception.h"
#include "../../../../io/testinitializer.h"
#include "../../../../misc/stringops.h"
#include "../../../../misc/testcell.h"
#include "../../../../mpilayer/mpilayer.h"
#include "../../updategroup.h"
#include "../../partitionmanager.h"
#include "../../partitions/hilbertpartition.h"
#include "../../partitions/hindexingpartition.h"
#include "../../partitions/stripingpartition.h"
#include "../../partitions/zcurvepartition.h"

using namespace boost::assign;
using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class UpdateGroupPrototypeTest : public CxxTest::TestSuite
{
public:
    // typedef ZCurvePartition Partition;
    // typedef PartitionManager::RegionVecMap RegionVecMap;

    void setUp()
    {
    } 
// void off1() {
//         // fixme
//         rank = MPILayer().rank();
//         dimensions = Coord(1400, 1000);
//         unsigned totalSize = dimensions.x * dimensions.y;
//         partition = Partition(Coord(0, 0), dimensions);
//         SuperVector<unsigned> weights;

//         int maxSteps = 1500;
//         int firstStep = 20;
//         int firstNanoStep = 18;
//         init.reset(new TestInitializer(dimensions.x, dimensions.y, maxSteps, firstStep, firstNanoStep));

//         weights.clear();
//         weights += 
//             totalSize / 9,
//             7 * totalSize / 12;
//         fillinRemainder(&weights, totalSize);
//         metaClusterGroup.init(partition, dimensions, weights, 0, 1, 30);

//         weights.clear();
//         weights +=
//             totalSize / 7 - metaClusterGroup.weights[0],
//             totalSize / 2;
//         fillinRemainder(&weights, metaClusterGroup.weights[1]);
//         clusterGroup.init(partition, dimensions, weights, metaClusterGroup.weights[0], 1, 22);

//         weights.clear();
//         unsigned secondToLastNodeWeight = totalSize / 11;
//         unsigned lastNodeWeight = totalSize / 88;
//         unsigned numNormalNodes = 7;
//         unsigned normalNodeWeight = (clusterGroup.weights[1] - lastNodeWeight - secondToLastNodeWeight) / numNormalNodes;
//         for (int i = 0; i < numNormalNodes; ++i)
//             weights += normalNodeWeight;
//         weights += secondToLastNodeWeight;
//         fillinRemainder(&weights, clusterGroup.weights[1]);
//         nodeGroup.init(partition, dimensions, weights, clusterGroup.offset + clusterGroup.weights[0], rank, 9);
//     }

    void testFoo()
    {
    }
 // void off2() {
//         // fixme
//         if (rank == 0) {
//             metaClusterGroup.print("test00MetaCluster.ppm");
//             clusterGroup.print("test01Cluster.ppm");
//         }
//         nodeGroup.print("test02Node" + StringConv::itoa(rank) + ".ppm");

//         Region intersectionRegion1 = metaClusterGroup.partitionManager.rim(metaClusterGroup.ghostZoneWidth);
//         UpdateGroup<TestCell, Partition> o1(
//             intersectionRegion1,
//             partition, 
//             nodeGroup.weights, 
//             nodeGroup.offset, 
//             CoordRectangle(Coord(0, 0), dimensions.x, dimensions.y),
//             metaClusterGroup.ghostZoneWidth,
//             &*init);
//         GhostZoneResolution res1(
//             o1.partitionManager.getOuterOutgroupGhostZoneFragment(),
//             Region(),
//             nodeGroup.partitionManager,
//             nodeGroup.boundingBoxes);
//         printGhostZoneResolution(o1, res1, clusterGroup.partitionManager.ownRegion(), "test03Level2fromLevel0");

//         Region intersectionRegion2 = clusterGroup.partitionManager.rim(clusterGroup.ghostZoneWidth);
//         UpdateGroup<TestCell, Partition> o2(
//             intersectionRegion2,
//             partition, 
//             nodeGroup.weights, 
//             nodeGroup.offset, 
//             CoordRectangle(Coord(0, 0), dimensions.x, dimensions.y),
//             clusterGroup.ghostZoneWidth,
//             &*init);
//         int delta2 = metaClusterGroup.ghostZoneWidth - clusterGroup.ghostZoneWidth;
//         GhostZoneResolution res2(
//             o2.partitionManager.getOuterOutgroupGhostZoneFragment(),
//             o1.partitionManager.rim(delta2),
//             nodeGroup.partitionManager,
//             nodeGroup.boundingBoxes);
//         printGhostZoneResolution(o2, res2, clusterGroup.partitionManager.ownRegion(), "test03Level1fromLevel0");

//         Region intersectionRegion3 = clusterGroup.partitionManager.ownRegion();
//         UpdateGroup<TestCell, Partition> o3(
//             intersectionRegion3,
//             partition, 
//             nodeGroup.weights, 
//             nodeGroup.offset, 
//             CoordRectangle(Coord(0, 0), dimensions.x, dimensions.y),
//             nodeGroup.ghostZoneWidth,
//             &*init);
//         int delta3 = clusterGroup.ghostZoneWidth - nodeGroup.ghostZoneWidth;
//         GhostZoneResolution res3(
//             o3.partitionManager.getOuterOutgroupGhostZoneFragment(),
//             o2.partitionManager.rim(delta3),
//             nodeGroup.partitionManager,
//             nodeGroup.boundingBoxes);
//         printGhostZoneResolution(o3, res3, clusterGroup.partitionManager.ownRegion(), "test03Level0fromLevel0");
//     }

// private:
//     template<class PARTITION>
//     class TestGroup
//     {
//     public:
//         typedef typename PARTITION::Iterator Iterator;
//         PARTITION partition;
//         Coord dimensions;
//         PartitionManager partitionManager;
//         SuperVector<unsigned> weights;
//         unsigned offset;
//         unsigned ghostZoneWidth;
//         unsigned rank;
//         SuperVector<CoordRectangle> boundingBoxes;

//         void init(const PARTITION& _partition, const Coord& _dimensions, const SuperVector<unsigned>& _weights, const unsigned& _offset, const unsigned& _rank, const unsigned& _ghostZoneWidth)
//         {
//             partition = _partition;
//             dimensions = _dimensions;
//             weights = _weights;
//             offset = _offset;
//             rank = _rank;
//             ghostZoneWidth = _ghostZoneWidth;
            
//             partitionManager.resetRegions(
//                 CoordRectangle(Coord(0, 0), dimensions.x, dimensions.y),
//                 new VanillaRegionAccumulator<Partition>(partition, offset, weights),
//                 rank,
//                 ghostZoneWidth);
//             boundingBoxes = genBoundingBoxes(partition, offset, weights);
//             partitionManager.resetGhostZones(boundingBoxes);
//         }

//         void print(const std::string& filename)
//         {
//             Grid<Color> img(dimensions.x, dimensions.y);
//             fillRegion(&img, partition[offset], partition[offset + weights.sum()], Color(0, 0, 100), addColor);
//             fillRegion(&img, partitionManager.ownRegion(), Color(0, 100, 100), addColor);
//             fillRegion(&img, partitionManager.ownExpandedRegion(), Color(150, 0, 0), addColor);
//             fillRegion(&img, partitionManager.getOuterOutgroupGhostZoneFragment(), Color(0, 100, 0), addColor);
//             printImageToFile(img, filename);
//         }

//     private:
//         SuperVector<CoordRectangle> genBoundingBoxes(const PARTITION& partition, const unsigned& offset, const SuperVector<unsigned>& weights)
//         {
//             SuperVector<CoordRectangle> ret(weights.size());
//             unsigned currentOffset = offset;
//             for (int i = 0; i < weights.size(); ++i) {
//                 Region s;
//                 for (Iterator c = partition[currentOffset]; c != partition[currentOffset + weights[i]]; ++c)
//                     s << *c;
//                 ret[i] = s.boundingBox();
//                 currentOffset += weights[i];
//             }
//             return ret;
//         }
//     };

//     class GhostZoneResolution
//     {
//     public:
//         template<class PARTITION_MANAGER>
//         GhostZoneResolution(const Region& outerGhostZone, const Region& patchRegion, PARTITION_MANAGER partitionManager, const SuperVector<CoordRectangle>& boundingBoxes)
//         {
//             fromHost = outerGhostZone & partitionManager.ownRegion();
//             fromMaster = outerGhostZone - fromHost;
//             fromPatches = fromMaster & patchRegion;
//             fromMaster -= fromPatches;

//             CoordRectangle fromMasterBoundingBox = fromMaster.boundingBox();

//             for (unsigned i = 0; i < boundingBoxes.size(); ++i) {
//                 if (fromMasterBoundingBox.intersects(boundingBoxes[i])) {
//                     Region intersection = fromMaster & partitionManager.getRegion(i, 0);
//                     if (!intersection.empty()) {
//                         fromNeighbors[i] = intersection;
//                         fromMaster -= intersection;
//                     }
//                 }
//             }
//         }

//         Region fromHost;
//         Region fromMaster;
//         Region fromPatches;
//         SuperMap<unsigned, Region> fromNeighbors;        
//     };

//     boost::shared_ptr<Initializer<TestCell> > init;
//     unsigned rank;
//     Coord dimensions;
//     Partition partition;
//     TestGroup<Partition> metaClusterGroup, clusterGroup, nodeGroup;

//     void fillinRemainder(SuperVector<unsigned> *vec, const unsigned& size)
//     {
//         vec->push_back(size - vec->sum());
//     }

//     static void printImageToFile(const Grid<Color>& img, const std::string& filename)
//     {
//         std::ofstream outfile(filename.c_str());
//         if (!outfile) 
//             throw FileOpenException("Cannot open output file", filename, errno);
//         outfile << "P6 " << img.width() << " " << img.height() << " 255\n";
//         for(unsigned y = 0; y < img.height(); ++y) 
//             for(unsigned x = 0; x < img.width(); ++x) 
//                 outfile << (char)img[Coord(x, y)].red() << (char)img[Coord(x, y)].green() << (char)img[Coord(x, y)].blue();
//     }

//     static void setColor(Color *col, const Color& newCol)
//     {
//         *col = newCol;
//     }

//     static void addColor(Color *col, const Color& addent)
//     {
//         *col = Color(col->red()   + addent.red(),
//                      col->green() + addent.green(),
//                      col->blue()  + addent.blue());
//     }

//     template <class ITERATOR>
//     static void fillRegion(
//         Grid<Color> *img, 
//         const ITERATOR& start, 
//         const ITERATOR& end, 
//         const Color& col,
//         void (*manipulator)(Color*, const Color&))
//     {
//         for (ITERATOR i = start; i != end; ++i) 
//             (*manipulator)(&(*img)[*i], col);
//     }

//     static void fillRegion(
//         Grid<Color> *img,
//         const Region& region,
//         const Color& col,
//         void (*manipulator)(Color*, const Color&))
//     {
//         for (Region::Iterator i = region.begin(); i != region.end(); ++i) 
//             (*manipulator)(&(*img)[*i], col);
//     }

//     template<class UPDATEGROUP>
//     void printGhostZoneResolution(UPDATEGROUP outgroup, const GhostZoneResolution& resolution, const Region& neighborLevelRegion, const std::string& filenamePrefix)
//     {
//         Grid<Color> img(dimensions.x, dimensions.y);

//         fillRegion(&img, neighborLevelRegion, Color(0, 0, 100), addColor);
//         fillRegion(&img, outgroup.partitionManager.ownRegion(), Color(0, 100, 100), addColor);

//         RegionVecMap outerGhostZoneFragments = outgroup.partitionManager.getOuterGhostZoneFragments();
//         for (RegionVecMap::iterator i = outerGhostZoneFragments.begin();
//              i != outerGhostZoneFragments.end();
//              ++i) {
//             if (i->first != PartitionManager::OUTGROUP)
//                 fillRegion(&img, i->second.back(), Color(0, 200, 0), setColor);
//         }

//         fillRegion(&img, resolution.fromHost,    Color(250, 250, 250), setColor);
//         fillRegion(&img, resolution.fromMaster,  Color(250, 200,   0), setColor);
//         fillRegion(&img, resolution.fromPatches, Color(  0, 250, 250), setColor);

//         for (SuperMap<unsigned, Region>::const_iterator i = resolution.fromNeighbors.begin();
//              i != resolution.fromNeighbors.end();
//              ++i) 
//             fillRegion(&img, i->second, Color(200, 0, 0), setColor);

//         printImageToFile(img, filenamePrefix + "Node" + StringConv::itoa(rank) + ".ppm");
    // }
};

}
}
