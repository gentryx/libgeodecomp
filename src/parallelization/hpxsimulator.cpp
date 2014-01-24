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

// FIXME: somehow, this needs to be defined under certain circumstances ...
namespace hpx { namespace naming {
    template <typename Archive>
    void id_type::load(Archive&, unsigned int)
    {
        HPX_ASSERT(false);
    }
    template void id_type::load<boost::archive::binary_iarchive>(boost::archive::binary_iarchive&, unsigned int);
    template <typename Archive>
    void id_type::save(Archive&, unsigned int) const
    {
        HPX_ASSERT(false);
    }
    template void id_type::save<boost::archive::binary_oarchive>(boost::archive::binary_oarchive&, unsigned int) const;
}}

#endif
