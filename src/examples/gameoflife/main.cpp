/**
 * We need to include typemaps first to avoid problems with Intel
 * MPI's C++ bindings (which may collide with stdio.h's SEEK_SET,
 * SEEK_CUR etc.).
 */
#include <libgeodecomp/mpilayer/typemaps.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

#include "common.h"

void runSimulation()
{
    int outputFrequency = 1;
    CellInitializer *init = new CellInitializer();

    StripingSimulator<ConwayCell> sim(
        init,
        MPILayer().rank() ? 0 : new TracingBalancer(new OozeBalancer()),
        10,
        1,
        MPI::BOOL);

    sim.addWriter(
        new BOVWriter<ConwayCell, StateSelector>(
            "game",
            outputFrequency));
    /*
    sim.addWriter(
        new TracingWriter<ConwayCell>(
            1,
            init->maxSteps()));
    */

    sim.run();
}

int main(int argc, char *argv[])
{
    MPI::Init(argc, argv);
    Typemaps::initializeMaps();

    runSimulation();

    MPI::Finalize();
    return 0;
}
