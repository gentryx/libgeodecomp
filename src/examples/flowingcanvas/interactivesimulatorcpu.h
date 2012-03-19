#ifndef _libgeodecomp_examples_flowingcanvas_interactivesimulatorcpu_h_
#define _libgeodecomp_examples_flowingcanvas_interactivesimulatorcpu_h_

#include <libgeodecomp/examples/flowingcanvas/interactivesimulator.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class InteractiveSimulatorCPU : public InteractiveSimulator
{
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef Grid<CELL_TYPE, Topology> GridType;
    InteractiveSimulatorCPU(QObject *parent, Initializer<CELL_TYPE> *initializer) :
        InteractiveSimulator(parent),
        sim(initializer)
    {}

    virtual ~InteractiveSimulatorCPU()
    {
        std::cout << "InteractiveSimulatorCPU dying\n";
    }

    virtual void loadStates()
    {
        std::cout << "loadStates()\n";
    }

    // fixme: move this out of the simulator!
    virtual void renderOutput()
    {
        Coord<2> dim = sim.getInitializer()->gridDimensions();
        const typename SerialSimulator<CELL_TYPE>::GridType *grid = sim.getGrid();
        int maxX = std::min((int)outputFrameWidth,  dim.x());
        int maxY = std::min((int)outputFrameHeight, dim.y());

        for (int y = 0; y < maxY; ++y) {
            for (int x = 0; x < maxX; ++x) {
                outputFrame[y * outputFrameWidth + x] = (*grid)[Coord<2>(x, y)].toColor();
            }
        }
    }

    virtual void update()
    {
        sim.step();
    }

private:
    SerialSimulator<CELL_TYPE> sim;
    std::vector<unsigned> frame;
};

}

#endif
