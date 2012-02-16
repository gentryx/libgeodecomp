#ifndef _libgeodecomp_parallelization_hiparsimulator_offsethelper_h_
#define _libgeodecomp_parallelization_hiparsimulator_offsethelper_h_

#include <libgeodecomp/misc/coord.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

// fixme: needs test
template<int INDEX, int DIM, typename TOPOLOGY>
class OffsetHelper
{
public:
    void operator()(
        Coord<DIM> *offset,
        Coord<DIM> *dimensions,
        const CoordBox<DIM>& ownBoundingBox, 
        const CoordBox<DIM>& simulationArea,
        const int& ghostZoneWidth)
    {
        offset->c[INDEX] = 0;
        if (TOPOLOGY::WrapEdges) {
            int enlargedWidth = 
                ownBoundingBox.dimensions.c[INDEX] + 2 * ghostZoneWidth;
            if (enlargedWidth < simulationArea.dimensions.c[INDEX]) {
                offset->c[INDEX] = 
                    ownBoundingBox.origin.c[INDEX] - ghostZoneWidth;
            } else {
                offset->c[INDEX] = 0;
            }
            dimensions->c[INDEX] = 
                std::min(enlargedWidth, simulationArea.dimensions.c[INDEX]);
        } else {
            offset->c[INDEX] = 
                std::max(0, ownBoundingBox.origin.c[INDEX] - ghostZoneWidth);
            int end = std::min(simulationArea.origin.c[INDEX] + 
                               simulationArea.dimensions.c[INDEX],
                               ownBoundingBox.origin.c[INDEX] + 
                               ownBoundingBox.dimensions.c[INDEX] + 
                               ghostZoneWidth);
            dimensions->c[INDEX] = end - offset->c[INDEX];
        } 

        OffsetHelper<INDEX - 1, DIM, typename TOPOLOGY::ParentTopology>()(
            offset, 
            dimensions, 
            ownBoundingBox, 
            simulationArea, 
            ghostZoneWidth);
    }
};

template<int DIM, typename TOPOLOGY>
class OffsetHelper<-1, DIM, TOPOLOGY>
{
public:
    void operator()(
        Coord<DIM> *offset,
        Coord<DIM> *dimensions,
        const CoordBox<DIM>& ownBoundingBox, 
        const CoordBox<DIM>& simulationArea,
        const int& ghostZoneWidth)
    {}
};

}
}

#endif
