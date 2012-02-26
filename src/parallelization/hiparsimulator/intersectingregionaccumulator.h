#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_intersectingregionaccumulator_h_
#define _libgeodecomp_parallelization_hiparsimulator_intersectingregionaccumulator_h_

#include <libgeodecomp/parallelization/hiparsimulator/vanillaregionaccumulator.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<typename PARTITION>
class IntersectingRegionAccumulator : public VanillaRegionAccumulator<PARTITION>
{
public:
    const static int DIM = PARTITION::DIM;

    inline IntersectingRegionAccumulator(
        const Region<DIM>& _intersectionRegion,
        const PARTITION& _partition, 
        const long& offset=0,
        const SuperVector<long>& weights=SuperVector<long>()) :
        VanillaRegionAccumulator<PARTITION>(_partition, offset, weights),
        intersectionRegion(_intersectionRegion)
    {}

    inline virtual Region<DIM> getRegion(const long& node)
    {
        return this->VanillaRegionAccumulator<PARTITION>::getRegion(node) & intersectionRegion;
    }

private:
    Region<DIM> intersectionRegion;
};

};
};

#endif
#endif
