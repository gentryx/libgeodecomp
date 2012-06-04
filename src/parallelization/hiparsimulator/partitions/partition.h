#ifndef _libgeodecomp_parallelization_hiparsimulator_partition_h_
#define _libgeodecomp_parallelization_hiparsimulator_partition_h_

#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/misc/supervector.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<int DIM>
class Partition
{
public:
    inline Partition(
        // fixme: drop offset and bounding box from all partitions in favor of simulation region specifier?
        const long& offset,
        const SuperVector<long>& _weights) :
        weights(_weights)
    {
        startOffsets.resize(weights.size() + 1);
        startOffsets[0] = offset;
        for (long i = 0; i < weights.size(); ++i)
            startOffsets[i + 1] = startOffsets[i] + weights[i];        
    }

    virtual ~Partition()
    {}

    inline const SuperVector<long>& getWeights() const
    {
        return weights;
    }

    virtual Region<DIM> getRegion(const long& node) const = 0;

protected:
    SuperVector<long> weights;
    SuperVector<long> startOffsets;
};

}
}

#endif
