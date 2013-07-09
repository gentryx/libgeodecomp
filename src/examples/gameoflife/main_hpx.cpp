
#include <libgeodecomp/parallelization/hiparsimulator/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/parallelization/hpxsimulator.h>
#include <libgeodecomp/io/hpxwritercollector.h>
#include <libgeodecomp/io/serialbovwriter.h>

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

typedef LibGeoDecomp::HpxWriterCollector<ConwayCell> HpxWriterCollectorType;
BOOST_CLASS_EXPORT_GUID(HpxWriterCollectorType, "HpxWriterCollectorCell");

typedef LibGeoDecomp::SerialBOVWriter<ConwayCell, StateSelector> BovWriterType;
BOOST_CLASS_EXPORT_GUID(BovWriterType, "BovWriterConwayCell");

typedef LibGeoDecomp::HpxWriterSink<ConwayCell> HpxWriterSinkType;

HPX_REGISTER_ACTION_DECLARATION(
    HpxWriterSinkType::ComponentType::StepFinishedAction,
    HpxWriterSinkType_ComponentType_StepFinishedAction_ConwayCell
)

HPX_REGISTER_ACTION_DECLARATION(
    HpxWriterSinkType::ComponentWriterCreateActionType,
    HpxWriterSinkType_ComponentWriterCreateActionType_ConwayCell
)

HPX_REGISTER_ACTION_DECLARATION(
    HpxWriterSinkType::ComponentParallelWriterCreateActionType,
    HpxWriterSinkType_ComponentParallelWriterCreateActionType_ConwayCell
)

typedef
    hpx::components::managed_component<
        HpxWriterSinkType::ComponentType
    >
    HpxWriterSinkComponentType;
    
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    HpxWriterSinkComponentType,
    HpxWriterSinkComponentType
);

HPX_REGISTER_ACTION(
    HpxWriterSinkType::ComponentType::StepFinishedAction,
    HpxWriterSinkType_ComponentType_StepFinishedAction_ConwayCell
)

HPX_REGISTER_ACTION(
    HpxWriterSinkType::ComponentWriterCreateActionType,
    HpxWriterSinkType_ComponentWriterCreateActionType_ConwayCell
)

HPX_REGISTER_ACTION(
    HpxWriterSinkType::ComponentParallelWriterCreateActionType,
    HpxWriterSinkType_ComponentParallelWriterCreateActionType_ConwayCell
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
 
        HpxWriterSinkType sink(
            boost::shared_ptr<BovWriterType>(new BovWriterType("game", outputFrequency)),
            sim.numUpdateGroups());

        sim.addWriter(
            new HpxWriterCollectorType(
                outputFrequency,
                sink
            ));

        /*
        sim.addWriter(
            new TracingWriterType(
                1,
                init->maxSteps()));
        */

        sim.run();
    }
    return hpx::finalize();
}

int main(int argc, char **argv)
{
    return hpx::init(argc, argv);
}
