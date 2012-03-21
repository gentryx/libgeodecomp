#ifndef _libgeodecomp_examples_flowingcanvas_canvaswriter_h_
#define _libgeodecomp_examples_flowingcanvas_canvaswriter_h_

#include <QImage>
#include <QPainter>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/examples/flowingcanvas/canvascell.h>

namespace LibGeoDecomp {

class CanvasWriter : public QObject, public Writer<CanvasCell>
{
    Q_OBJECT

public:
    class SelectForceFixed
    {
    public:
        float operator()(const CanvasCell& cell, const int& index) const
        {
            return cell.forceFixed[index];
        }
    };

    class SelectForceVario
    {
    public:
        float operator()(const CanvasCell& cell, const int& index) const
        {
            return cell.forceVario[index];
        }
    };

    class SelectForceTotal
    {
    public:
        float operator()(const CanvasCell& cell, const int& index) const
        {
            return cell.forceTotal[index];
        }
    };

    class SelectCameraLevel
    {
    public:
        unsigned operator()(const CanvasCell& cell) const
        {
            return (0xff << 24) + ((int)(cell.cameraLevel * 250) << 16);
        }
    };

    class SelectCameraPixel
    {
    public:
        unsigned operator()(const CanvasCell& cell) const
        {
            return cell.cameraPixel;
        }
    };

    CanvasWriter(QImage **_outputFrame,
                 MonolithicSimulator<CanvasCell> *_sim) :
        Writer<CanvasCell>("foo", _sim, 1),
        outputFrame(_outputFrame),
        mode(5)
    {}

    virtual void initialized()
    {}

    virtual void stepFinished()
    {
        switch (mode) {
        case 0:
            drawAttribute(SelectCameraPixel());
            break;
        case 1:
            drawAttribute(SelectCameraLevel());
            break;
        case 2:
            drawForce(SelectForceVario());
            break;
        case 3:
            drawForce(SelectForceFixed());
            break;
        case 4:
            drawForce(SelectForceTotal());
            break;
        case 5:
            drawParticles();
            break;
        default:
            break;
        }
    }

    virtual void allDone()
    {
    }

public slots:
    virtual void cycleViewMode()
    {
        mode = (mode + 1) % 6;
    }

private:
    QImage **outputFrame;
    int mode;

    template<typename SELECTOR>
    void drawForce(const SELECTOR& selector)
    {
        Coord<2> dim = sim->getInitializer()->gridDimensions();
        const typename Simulator<CanvasCell>::GridType *grid = sim->getGrid();
       
        int spacingX = 10;
        int spacingY = 10;
        float factorX = 1.0 * (*outputFrame)->width()  / dim.x();
        float factorY = 1.0 * (*outputFrame)->height() / dim.y();

        QPainter p(*outputFrame);
        p.setBrush(QBrush(Qt::black));
        p.drawRect(0, 0, (*outputFrame)->width(), (*outputFrame)->height());
        p.setBrush(QBrush(Qt::white));
        p.setPen(QPen(Qt::white));

        for (int y = 0; y < dim.y(); y += spacingY) {
            for (int x = 0; x < dim.x(); x += spacingX) {
                int startX = (x + 0.5) * factorX;
                int startY = (y + 0.5) * factorY;
                const CanvasCell& cell = grid->at(Coord<2>(x, y));
                float force0 = selector(cell, 0);
                float force1 = selector(cell, 1);
                int offsetX = force0 * spacingX * factorX * 0.8;
                int offsetY = force1 * spacingY * factorY * 0.8;
                int endX = startX + offsetX;
                int endY = startY + offsetY;
                p.drawLine(startX, startY, endX, endY); 
                QRectF rect(endX - spacingX * 0.1, endY - spacingY * 0.1, spacingX * 0.2, spacingY * 0.2);
                p.drawEllipse(rect);
            }
        }
    }

    // fixme: erase background
    // fixme: scale to image size
    template<typename SELECTOR>
    void drawAttribute(const SELECTOR& selector)
    {
        Coord<2> dim = sim->getInitializer()->gridDimensions();
        const typename Simulator<CanvasCell>::GridType *grid = sim->getGrid();
       
        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                (*outputFrame)->setPixel(x, y, selector(grid->at(Coord<2>(x, y))));
            }
        }
    }

    void drawParticles()
    {
        Coord<2> dim = sim->getInitializer()->gridDimensions();
        const typename Simulator<CanvasCell>::GridType *grid = sim->getGrid();

        float factorX = 1.0 * (*outputFrame)->width()  / dim.x();
        float factorY = 1.0 * (*outputFrame)->height() / dim.y();

        QPainter p(*outputFrame);
        p.setBrush(QBrush(Qt::black));
        p.drawRect(0, 0, (*outputFrame)->width(), (*outputFrame)->height());
        p.setBrush(QBrush(Qt::white));
        p.setPen(QPen(Qt::white));

        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                const CanvasCell& cell = grid->at(Coord<2>(x, y));
                for (int i = 0; i < cell.numParticles; ++i) {
                    const CanvasCell::Particle& particle = cell.particles[i];
                    QPoint origin(particle.pos[0] * factorX,
                                  particle.pos[1] * factorY);
                    QPoint direction(particle.vel[0] * 50,
                                     particle.vel[1] * 50);
                    QPoint end = origin + direction;
                    QPoint offset(2, 2);
                    QSize size(4, 4);
                    QRectF rect(origin - offset, size);
                    p.drawEllipse(rect);
                    p.drawLine(origin, end);
                }
            }
        }
    }
};

}

#endif
