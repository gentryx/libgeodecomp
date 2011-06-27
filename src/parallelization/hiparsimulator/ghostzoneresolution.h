#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_ghostzoneresolution_h_
#define _libgeodecomp_parallelization_hiparsimulator_ghostzoneresolution_h_

#include <libgeodecomp/misc/region.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

// fixme: prune this class and the updategroup
// class GhostZoneResolution
// {
// public:
//     template<class PARTITION_MANAGER>
//     GhostZoneResolution(const Region& outerGhostZone, const Region& patchRegion, PARTITION_MANAGER partitionManager)
//     {
//         fromHost = outerGhostZone & partitionManager.ownRegion();
//         fromMaster = outerGhostZone - fromHost;
//         fromPatches = fromMaster & patchRegion;
//         fromMaster -= fromPatches;

//         CoordRectangle fromMasterBoundingBox = fromMaster.boundingBox();
//         const SuperVector<CoordRectangle>& boundingBoxes = partitionManager.getBoundingBoxes();
//         for (unsigned i = 0; i < boundingBoxes.size(); ++i) {
//             if (fromMasterBoundingBox.intersects(boundingBoxes[i])) {
//                 Region intersection = fromMaster & partitionManager.getRegion(i, 0);
//                 if (!intersection.empty()) {
//                     fromNeighbors[i] = intersection;
//                     fromMaster -= intersection;
//                 }
//             }
//         }
//     }

//     Region fromHost;
//     Region fromMaster;
//     Region fromPatches;
//     SuperMap<unsigned, Region> fromNeighbors;        
// };

}
}

#endif
#endif
