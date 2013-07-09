
#include <libgeodecomp/parallelization/hiparsimulator/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/parallelization/hpxsimulator.h>
#include <libgeodecomp/io/hpxwritercollector.h>
#include <libgeodecomp/io/serialbovwriter.h>

#include <hpx/hpx_init.hpp>

#include "common.h"

typedef
    HpxSimulator::HpxSimulator<ConwayCell, HiParSimulator::RecursiveBisectionPartition<2> >
    SimulatorType;
LIBGEDECOMP_REGISTER_HPX_SIMULATOR_DECLARATION(
    SimulatorType,
    ConwayCellSimulator
)
LIBGEDECOMP_REGISTER_HPX_SIMULATOR(
    SimulatorType,
    ConwayCellSimulator
)

BOOST_CLASS_EXPORT_GUID(CellInitializer, "CellInitializer");

typedef LibGeoDecomp::TracingWriter<ConwayCell> TracingWriterType;
BOOST_CLASS_EXPORT_GUID(TracingWriterType, "TracingWriterConwayCell");

typedef LibGeoDecomp::SerialBOVWriter<ConwayCell, StateSelector> BovWriterType;
BOOST_CLASS_EXPORT_GUID(BovWriterType, "BovWriterConwayCell");

typedef 
    LibGeoDecomp::HpxWriterCollector<ConwayCell>
    HpxWriterCollectorType;
LIBGEODECOMP_REGISTER_HPX_WRITER_COLLECTOR_DECLARATION(
    HpxWriterCollectorType,
    ConwayCellWriterCollector
)
LIBGEODECOMP_REGISTER_HPX_WRITER_COLLECTOR(
    HpxWriterCollectorType,
    ConwayCellWriterCollector
)

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
 
        HpxWriterCollectorType::SinkType sink(
            new BovWriterType("game", outputFrequency),
            sim.numUpdateGroups());

        sim.addWriter(
            new HpxWriterCollectorType(
                outputFrequency,
                sink
            ));

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
