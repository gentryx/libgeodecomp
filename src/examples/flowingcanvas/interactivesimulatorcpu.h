#ifndef _libgeodecomp_examples_flowingcanvas_interactivesimulatorcpu_h_
#define _libgeodecomp_examples_flowingcanvas_interactivesimulatorcpu_h_

#include <libgeodecomp/examples/flowingcanvas/interactivesimulator.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class InteractiveSimulatorCPU : public SerialSimulator<CELL_TYPE>, public InteractiveSimulator
{
public:
    typedef typename CELL_TYPE::Topology Topology;
    typedef typename SerialSimulator<CELL_TYPE>::GridType GridType;
    typedef std::vector<boost::shared_ptr<Writer<CELL_TYPE> > > WriterVector;

    InteractiveSimulatorCPU(QObject *parent, Initializer<CELL_TYPE> *initializer) :
        SerialSimulator<CELL_TYPE>(initializer),
        InteractiveSimulator(parent)
    {}

    virtual ~InteractiveSimulatorCPU()
    {}

    virtual void readCam()
    {
        Coord<2> dim = this->getInitializer()->gridDimensions();
        float factorX = 1.0 * cameraFrameWidth  / dim.x();
        float factorY = 1.0 * cameraFrameHeight / dim.y();

        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                Coord<2> c(x, y);
                (*this->curGrid)[c].readCam(&cameraFrame[0], factorX, factorY, cameraFrameWidth, cameraFrameHeight);
            }
        }
    }

    virtual void renderOutput()
    {
        for(unsigned i = 0; i < this->writers.size(); i++) 
            this->writers[i]->stepFinished();
    }

    virtual void update()
    {
        SerialSimulator<CELL_TYPE>::step();
    }

    virtual void registerWriter(Writer<CELL_TYPE> *writer)
    {
        writers.push_back(boost::shared_ptr<Writer<CELL_TYPE> >(writer));
    }

protected:
    WriterVector writers;
};

}

#endif
