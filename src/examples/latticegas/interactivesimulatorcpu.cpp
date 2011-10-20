#include <libgeodecomp/examples/latticegas/interactivesimulatorcpu.h>


InteractiveSimulatorCPU::InteractiveSimulatorCPU(QObject *parent) :
    InteractiveSimulator(parent),
    gridOld(SimParams::modelSize),
    gridNew(SimParams::modelSize)
{
    // fixme: put this in the initializer
    // add initial cells
    for (int y = 5; y < SimParams::modelHeight - 5; y += 10) {
        for (int x = 5; x < SimParams::modelWidth - 5; x += 10) {
            gridOld[y  * SimParams::modelWidth + x][0] = Cell(Cell::liquid, Cell::R, 1);
        }
    }
}

InteractiveSimulatorCPU::~InteractiveSimulatorCPU()
{}

void InteractiveSimulatorCPU::loadStates()
{
    for (int y = 0; y < SimParams::modelHeight; ++y) {
        for (int x = 0; x < SimParams::modelWidth; ++x) {
            unsigned pos = y * SimParams::modelWidth + x;
            gridOld[pos][0].getState() = states[pos];
            gridOld[pos][1].getState() = states[pos];
        }
    }
}

void InteractiveSimulatorCPU::renderOutput()
{
    // fixme
}

void InteractiveSimulatorCPU::update()
{
    // fixme: use stepper here
   for (int y = 1; y < SimParams::modelHeight - 1; ++y) {
        for (int x = 1; x < SimParams::modelWidth - 1; ++x) {
            unsigned pos = y * SimParams::modelWidth + x;
            gridNew[pos].update(t,
                                Cell::simpleRand(pos + t),
                                &gridOld[pos - SimParams::modelWidth],
                                &gridOld[pos],
                                &gridOld[pos + SimParams::modelWidth]);
        }
    }
    std::swap(gridNew, gridOld);
}
