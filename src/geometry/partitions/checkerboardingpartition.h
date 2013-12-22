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

    Region<DIM> getRegion(const std::size_t node) const
    {

        Coord<DIM> nodeGridDim = getNodeGridDim(weights.size());
        Coord<DIM> logicalCoord(node % nodeGridDim.x(),
                               (node % (nodeGridDim.x() * nodeGridDim.y()))/ nodeGridDim.x());
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
    Coord<DIM> getNodeGridDim(const std::size_t totalNodes) const
    {
        Coord<DIM> ret = Coord<DIM>::diagonal(1);
        if (DIM == 2) {
            for(long i = 2; i <= sqrt(totalNodes); ++i){
                if(totalNodes % i == 0){
                    ret[0] = i;
                }
            }
            ret[1] = totalNodes / ret[0];
            return ret;

        } else if (DIM == 3) {
            for(unsigned long i = 2; i <= sqrt(totalNodes); ++i){
                if(totalNodes % i == 0){
                    ret[0] = i;
                    ret[1] = totalNodes/i;
                    for(unsigned long j = 2; j <= sqrt(totalNodes/ret[0]); ++j){
                        if(totalNodes / i % j == 0){
                            ret[0] = i;
                            ret[1] = j;
                        }
                    }
                }
            }
            ret[2] = totalNodes/ret[0]/ret[1];
            return ret;
        }

    }


    using Partition<DIM>::startOffsets;
    using Partition<DIM>::weights;

    Coord<DIM> origin;
    Coord<DIM> dimensions;
};

}

#endif
