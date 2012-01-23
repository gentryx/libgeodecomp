#include <libgeodecomp/examples/latticegas/interactivesimulatorcpu.h>


InteractiveSimulatorCPU::InteractiveSimulatorCPU(QObject *parent) :
    InteractiveSimulator(parent),
    gridOld(simParamsHost.modelSize),
    gridNew(simParamsHost.modelSize),
    frame(simParamsHost.modelSize)
{
    // fixme: put this in the initializer
    // add initial cells
    for (int y = 5; y < simParamsHost.modelHeight - 5; y += 10) {
        for (int x = 5; x < simParamsHost.modelWidth - 5; x += 10) {
            gridOld[y  * simParamsHost.modelWidth + x][0] = Cell(Cell::liquid, Cell::C, 1);
        }
    }
}

InteractiveSimulatorCPU::~InteractiveSimulatorCPU()
{}

void InteractiveSimulatorCPU::loadStates()
{
    for (int y = 0; y < simParamsHost.modelHeight; ++y) {
        for (int x = 0; x < simParamsHost.modelWidth; ++x) {
            unsigned offset = y * simParamsHost.modelWidth + x;
            gridOld[offset][0].getState() = states[offset];
            gridOld[offset][1].getState() = states[offset];
            gridNew[offset][0].getState() = states[offset];
            gridNew[offset][1].getState() = states[offset];
        }
    }
}

void InteractiveSimulatorCPU::renderOutput()
{
    for (int y = 0; y < simParamsHost.modelHeight; ++y) {
        for (int x = 0; x < simParamsHost.modelWidth; ++x) {
            unsigned offset = y * simParamsHost.modelWidth + x;
            frame[offset] = gridOld[offset].toColor(&simParamsHost);
        }
    }

    for (int y = 0; y < outputFrameHeight; ++y) {
        for (int x = 0; x < outputFrameWidth; ++x) {
            int sourceX = x * simParamsHost.modelWidth  / outputFrameWidth;
            int sourceY = y * simParamsHost.modelHeight / outputFrameHeight;
            int sourceOffset = sourceY * simParamsHost.modelWidth + sourceX;
            unsigned offset = y * outputFrameWidth + x;
            outputFrame[offset] = frame[sourceOffset];
        }
    }
}

void InteractiveSimulatorCPU::update()
{
    // fixme: use stepper here
    for (int y = 1; y < simParamsHost.modelHeight - 1; ++y) {
        for (int x = 1; x < simParamsHost.modelWidth - 1; ++x) {
            unsigned offset = y * simParamsHost.modelWidth + x;
            gridNew[offset].update(
                &simParamsHost,
                t,
                &gridOld[offset - simParamsHost.modelWidth],
                &gridOld[offset],
                &gridOld[offset + simParamsHost.modelWidth]);
        }
    }
    std::swap(gridNew, gridOld);
}
