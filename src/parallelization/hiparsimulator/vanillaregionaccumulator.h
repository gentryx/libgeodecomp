#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_vanillaregionaccumulator_h_
#define _libgeodecomp_parallelization_hiparsimulator_vanillaregionaccumulator_h_

#include <boost/shared_ptr.hpp>
#include <libgeodecomp/parallelization/hiparsimulator/regionaccumulator.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

// fixme: deduce DIM from partition?!
template<typename PARTITION, int DIM>
class VanillaRegionAccumulator : public RegionAccumulator<DIM>
{
public:
    inline VanillaRegionAccumulator(
        const PARTITION& _partition=PARTITION(), 
        const unsigned& offset=0,
        const SuperVector<unsigned>& weights=SuperVector<unsigned>(2)) :
        partition(_partition)
    {
        startOffsets.resize(weights.size() + 1);
        startOffsets[0] = offset;
        for (unsigned i = 0; i < weights.size(); ++i)
            startOffsets[i + 1] = startOffsets[i] + weights[i];        
    }

    inline virtual Region<DIM> getRegion(const unsigned& node)
    {
        return Region<DIM>(
            partition[startOffsets[node]], 
            partition[startOffsets[node + 1]]);
    }

private:
    PARTITION partition;
    SuperVector<unsigned> startOffsets;
};

};
};

#endif
#endif
