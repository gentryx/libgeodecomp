#include <cxxtest/TestSuite.h>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/misc/nonpodtestcell.h>
#include <libgeodecomp/parallelization/hpxsimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class HpxSimulator6Test : public CxxTest::TestSuite
{
public:
    void testNonPoDCellLittle()
    {
#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
        int scalingFactor = 1;

        typedef HpxSimulator<NonPoDTestCell, ZCurvePartition<2> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();

        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 2;
        SimulatorType sim(
            new NonPoDTestCell::Initializer(scalingFactor),
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulator6Test/testNonPoDCellLittle");

        sim.run();
#endif
    }

    void testNonPoDCellBig()
    {
#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
        int scalingFactor = 3;

        typedef HpxSimulator<NonPoDTestCell, ZCurvePartition<2> > SimulatorType;
        std::size_t rank = hpx::get_locality_id();
        std::vector<hpx::id_type> localities = hpx::find_all_localities();

        std::vector<double> updateGroupSpeeds(1 + rank, 10.0 / (rank + 10));
        int loadBalancingPeriod = 10;
        int ghostZoneWidth = 2;
        SimulatorType sim(
            new NonPoDTestCell::Initializer(scalingFactor),
            updateGroupSpeeds,
            new TracingBalancer(new OozeBalancer()),
            loadBalancingPeriod,
            ghostZoneWidth,
            false,
            "/HpxSimulator6Test/testNonPoDCellBig");

        sim.run();
#endif
    }

};

}
