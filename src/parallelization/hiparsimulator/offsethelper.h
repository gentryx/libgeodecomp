#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_OFFSETHELPER_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_OFFSETHELPER_H

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
        (*offset)[INDEX] = 0;
        if (TOPOLOGY::WRAP_EDGES) {
            int enlargedWidth = 
                ownBoundingBox.dimensions[INDEX] + 2 * ghostZoneWidth;
            if (enlargedWidth < simulationArea.dimensions[INDEX]) {
                (*offset)[INDEX] = 
                    ownBoundingBox.origin[INDEX] - ghostZoneWidth;
            } else {
                (*offset)[INDEX] = 0;
            }
            (*dimensions)[INDEX] = 
                std::min(enlargedWidth, simulationArea.dimensions[INDEX]);
        } else {
            (*offset)[INDEX] = 
                std::max(0, ownBoundingBox.origin[INDEX] - ghostZoneWidth);
            int end = std::min(simulationArea.origin[INDEX] + 
                               simulationArea.dimensions[INDEX],
                               ownBoundingBox.origin[INDEX] + 
                               ownBoundingBox.dimensions[INDEX] + 
                               ghostZoneWidth);
            (*dimensions)[INDEX] = end - (*offset)[INDEX];
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
