#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/config.hpp>
#include <boost/foreach.hpp>
#include <libgeodecomp/parallelization/hpxsimulator.h>

HPX_REGISTER_BASE_LCO_WITH_VALUE(
    LibGeoDecomp::HpxSimulator::StepPairType,
    LibGeoDecomp_BaseLcoStepPair
)

HPX_REGISTER_BASE_LCO_WITH_VALUE(
    LibGeoDecomp::CoordBox<2>,
    LibGeoDecomp_BaseLcoCoordBox2
)

HPX_REGISTER_BASE_LCO_WITH_VALUE(
    std::vector<double>,
    LibGeoDecomp_BaseLcovector_double
)

HPX_REGISTER_BASE_LCO_WITH_VALUE(
    LibGeoDecomp::Chronometer,
    LibGeoDecomp_BaseLcovector_Statistics
)

HPX_REGISTER_BASE_LCO_WITH_VALUE(
    std::vector<LibGeoDecomp::Chronometer>,
    LibGeoDecomp_BaseLcovector_StatisticsVector
)

#endif
