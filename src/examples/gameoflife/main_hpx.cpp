
#include <libgeodecomp/parallelization/hiparsimulator/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/parallelization/hpxsimulator.h>

#include <hpx/hpx_init.hpp>

#include "common.h"

LIBGEDECOMP_REGISTER_HPX_SIMULATOR(
    ConwayCell
  , HiParSimulator::RecursiveBisectionPartition<2>
  , SimulatorType
)

BOOST_CLASS_EXPORT_GUID(CellInitializer, "CellInitializer");
typedef LibGeoDecomp::TracingWriter<ConwayCell> TracingWriterType;
BOOST_CLASS_EXPORT_GUID(TracingWriterType, "TracingWriterConwayCell");
typedef LibGeoDecomp::BOVWriterAlt<ConwayCell, StateSelector> BovWriterType;
BOOST_CLASS_EXPORT_GUID(BovWriterType, "BovWriterConwayCell");

int hpx_main()
{
    {
        int outputFrequency = 1;
        CellInitializer *init = new CellInitializer();

        SimulatorType sim(
            init,
            1, // overcommitFactor
            new TracingBalancer(new OozeBalancer()),
            10, // balancingPeriod
            1 // ghostZoneWidth
            );
        
        sim.addWriter(
            new BovWriterType(
                "game",
                outputFrequency));

        sim.addWriter(
            new TracingWriterType(
                1,
                init->maxSteps()));

        sim.run();
    }
    return hpx::finalize();
}

int main(int argc, char **argv)
{
    return hpx::init(argc, argv);
}
