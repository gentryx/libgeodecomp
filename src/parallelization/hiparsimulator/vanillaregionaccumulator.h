#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_vanillaregionaccumulator_h_
#define _libgeodecomp_parallelization_hiparsimulator_vanillaregionaccumulator_h_

#include <boost/shared_ptr.hpp>
#include <libgeodecomp/parallelization/hiparsimulator/regionaccumulator.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<typename PARTITION>
class VanillaRegionAccumulator : public RegionAccumulator<PARTITION::DIM>
{
public:
    const static int DIM = PARTITION::DIM;

    inline VanillaRegionAccumulator(
        const PARTITION& _partition=PARTITION(), 
        const long& offset=0,
        const SuperVector<long>& weights=SuperVector<long>(2)) :
        partition(_partition)
    {
        startOffsets.resize(weights.size() + 1);
        startOffsets[0] = offset;
        for (long i = 0; i < weights.size(); ++i)
            startOffsets[i + 1] = startOffsets[i] + weights[i];        
    }

    inline virtual Region<DIM> getRegion(const long& node)
    {
        return Region<DIM>(
            partition[startOffsets[node]], 
            partition[startOffsets[node + 1]]);
    }

private:
    PARTITION partition;
    SuperVector<long> startOffsets;
};

}
}

#endif
#endif
