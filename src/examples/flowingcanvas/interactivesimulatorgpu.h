#ifndef _libgeodecomp_examples_flowingcanvas_interactivesimulatorgpu_h_
#define _libgeodecomp_examples_flowingcanvas_interactivesimulatorgpu_h_

#include <libgeodecomp/examples/flowingcanvas/interactivesimulator.h>
#include <libgeodecomp/misc/grid.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class GPUSimulator : public MonolithicSimulator<CELL_TYPE>
{
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef Grid<CELL_TYPE, Topology> GridType;
    static const int DIMENSIONS = Topology::DIMENSIONS;

    GPUSimulator(Initializer<CELL_TYPE> *initializer) :
        MonolithicSimulator<CELL_TYPE>(initializer)
    {
        gridHost.resize(this->initializer->gridBox().dimensions);
        this->initializer->grid(&gridHost);
    }

    virtual void step()
    {
        for (unsigned i = 0; i < CELL_TYPE::nanoSteps(); i++)
            nanoStep(i);

        this->stepNum++;    
        // call back all registered Writers
        for(unsigned i = 0; i < this->writers.size(); i++) 
            this->writers[i]->stepFinished();
    }

    virtual void run()
    {
        std::cout << "fixme InteractiveSimulatorGPU::run()\n";
    }

    virtual const GridType *getGrid()
    {
        return &gridHost;
    }

private:
    GridType gridHost;
    CELL_TYPE *curGridDevice;
    CELL_TYPE *newGridDevice;

    void nanoStep(const unsigned& nanoStep)
    {
        std::cout << "fixme InteractiveSimulatorGPU::nanoStep()\n";
    }
};

template<typename CELL_TYPE>
class InteractiveSimulatorGPU : public GPUSimulator<CELL_TYPE>, public InteractiveSimulator
{
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef Grid<CELL_TYPE, Topology> GridType;
    static const int DIMENSIONS = Topology::DIMENSIONS;

    InteractiveSimulatorGPU(QObject *parent, Initializer<CELL_TYPE> *initializer) :
        GPUSimulator<CELL_TYPE>(initializer),
        InteractiveSimulator(parent)
    {}

    virtual ~InteractiveSimulatorGPU()
    {}

    virtual void readCam()
    {
        std::cout << "fixme InteractiveSimulatorGPU::readCam()\n";
    }

    virtual void renderOutput()
    {
        std::cout << "fixme InteractiveSimulatorGPU::renderOutput()\n";
    }

    virtual void update()
    {
        GPUSimulator<CELL_TYPE>::step();
    }
};

}

#endif
