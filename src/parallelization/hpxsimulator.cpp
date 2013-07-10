#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_HPX

#include <libgeodecomp/parallelization/hpxsimulator.h>

HPX_REGISTER_BASE_LCO_WITH_VALUE(
    LibGeoDecomp::HpxSimulator::StepPairType,
    LibGeoDecomp_BaseLcoStepPair
)

HPX_REGISTER_BASE_LCO_WITH_VALUE(
    LibGeoDecomp::CoordBox<2>,
    LibGeoDecomp_BaseLcoCoordBox2
)

#endif
