#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONS_CHECKERBOARDINGPARTITION_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONS_CHECKERBOARDINGPARTITION_H

#include <libgeodecomp/geometry/partitions/partition.h>

namespace LibGeoDecomp {

template<int DIM>
class CheckerboardingPartition : public Partition<DIM>
{
public:
    inline CheckerboardingPartition(
        const Coord<DIM>& origin = Coord<DIM>(),
        const Coord<DIM>& dimensions = Coord<DIM>(),
        const long& offset = 0,
        const std::vector<size_t>& weights = std::vector<std::size_t>(2)) :
        Partition<DIM>(offset, weights),
        origin(origin),
        dimensions(dimensions)
    {}

    Coord<DIM> getNodeGridDim(const std::size_t totalNodes) const
    {
        if (DIM == 2) {
            long factor = 1;
            for(long i = 2; i <= sqrt(totalNodes); ++i){
                if(totalNodes % i == 0){
                    factor = i;
                }
            }
            return Coord<DIM>(totalNodes/factor,factor);
        } else if (DIM == 3) {
            long factorX = 1;
            long factorY = 1;
            for(unsigned long i = 2; i <= sqrt(totalNodes); ++i){
                if(totalNodes % i == 0){
                    for(unsigned long j = 2; j <= (totalNodes / i); ++j){
                        if(totalNodes / i % j == 0 && j >= i){
                            factorX = i;
                            factorY = j;
                        }
                    }
                }
            }
            Coord<DIM> ret = Coord<DIM>::diagonal(1);
            ret[0] = factorX;
            ret[1] = factorY;
            ret[2] = totalNodes/factorX/factorY;
            return ret;
        }

    }

    Region<DIM> getRegion(const std::size_t node) const
    {

        Coord<DIM> nodeGridDim = getNodeGridDim(weights.size());
        Coord<DIM> logicalCoord(node % nodeGridDim.x(),
                                node / nodeGridDim.x());
        if (DIM > 2){
            logicalCoord[2] = node / (nodeGridDim.x() * nodeGridDim.y());
        }

        Coord<DIM> realStart;
        Coord<DIM> realEnd;
        for(int i = 0; i < DIM; ++i){
            realStart[i] = logicalCoord[i] * dimensions[i] / nodeGridDim[i];
            realEnd[i] = (logicalCoord[i]+1) * dimensions[i] / nodeGridDim[i];
        }

        Region<DIM> r;
        r << CoordBox<DIM>(origin + realStart, realEnd - realStart);

        return r;
    }

private:
    using Partition<DIM>::startOffsets;
    using Partition<DIM>::weights;

    Coord<DIM> origin;
    Coord<DIM> dimensions;
};

}

#endif
