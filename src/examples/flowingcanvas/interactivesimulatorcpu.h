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
    typedef typename SerialSimulator<CELL_TYPE>::GridType GridType;
    InteractiveSimulatorCPU(QObject *parent, Initializer<CELL_TYPE> *initializer) :
        InteractiveSimulator(parent),
        sim(initializer)
    {}

    virtual ~InteractiveSimulatorCPU()
    {}

    virtual void readCam()
    {
        Coord<2> dim = sim.getInitializer()->gridDimensions();
        // fixme: ugly hack
        GridType *grid = (GridType*)sim.getGrid();
        float factorX = 1.0 * cameraFrameWidth  / dim.x();
        float factorY = 1.0 * cameraFrameHeight / dim.y();

        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                Coord<2> c(x, y);
                (*grid)[c].readCam(&cameraFrame[0], factorX, factorY, cameraFrameWidth, cameraFrameHeight);
            }
        }
    }

    // fixme: move this out of the simulator!
    virtual void renderOutput()
    {
        Coord<2> dim = sim.getInitializer()->gridDimensions();
        const GridType *grid = sim.getGrid();
       

        int spacingX = 10;
        int spacingY = 10;
        float factorX = 1.0 * outputFrame->width()  / dim.x();
        float factorY = 1.0 * outputFrame->height() / dim.y();

        QPainter p(outputFrame);
        p.setBrush(QBrush(Qt::black));
        p.drawRect(0, 0, outputFrame->width(), outputFrame->height());
        p.setBrush(QBrush(Qt::white));
        p.setPen(QPen(Qt::white));

        for (int y = 0; y < dim.y(); y += spacingY) {
            for (int x = 0; x < dim.x(); x += spacingX) {
                int startX = (x + 0.5) * factorX;
                int startY = (y + 0.5) * factorY;
                // float force0 = (*grid)[Coord<2>(x, y)].forceFixed[0];
                // float force1 = (*grid)[Coord<2>(x, y)].forceFixed[1];
                // float force0 = (*grid)[Coord<2>(x, y)].forceVario[0];
                // float force1 = (*grid)[Coord<2>(x, y)].forceVario[1];
                float force0 = (*grid)[Coord<2>(x, y)].forceVario[0] * 0.5 + 0.5 * (*grid)[Coord<2>(x, y)].forceFixed[0];
                float force1 = (*grid)[Coord<2>(x, y)].forceVario[1] * 0.5 + 0.5 * (*grid)[Coord<2>(x, y)].forceFixed[1];
                int offsetX = force0 * spacingX * factorX * 0.8;
                int offsetY = force1 * spacingY * factorY * 0.8;
                int endX = startX + offsetX;
                int endY = startY + offsetY;
                p.drawLine(startX, startY, endX, endY); 
                QRectF rec(endX - spacingX * 0.1, endY - spacingY * 0.1, spacingX * 0.2, spacingY * 0.2);
                p.drawPie(rec, 0, 5760);
            }
        }

        // for (int y = 0; y < dim.y(); ++y) {
        //     for (int x = 0; x < dim.x(); ++x) {
        //         unsigned col = (0xff << 24) + ((int)((*grid)[Coord<2>(x, y)].cameraLevel * 250) << 16);
        //         outputFrame->setPixel(x, y, col);
        //     }
        // }
        
        // for (int y = 0; y < dim.y(); ++y) {
        //     for (int x = 0; x < dim.x(); ++x) {
        //         outputFrame->setPixel(x, y, (*grid)[Coord<2>(x, y)].cameraPixel);
        //     }
        // }
    }

    virtual void update()
    {
        sim.step();
    }

private:
    SerialSimulator<CELL_TYPE> sim;

};

}

#endif
