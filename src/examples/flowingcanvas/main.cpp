#include <QtCore/QTimer>
#include <QtGui/QApplication>
#include <QThreadPool>
#include <libgeodecomp/examples/flowingcanvas/flowwidget.h>
#include <libgeodecomp/examples/flowingcanvas/framegrabber.h>
#include <libgeodecomp/examples/flowingcanvas/interactivesimulatorcpu.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/misc/floatcoord.h>

using namespace LibGeoDecomp;

// class SimParams
// {
// public:
//     int maxCameraFrames;
//     float gradientCutoff;
// };

class Particle
{
public:

private:
    float pos[2];
    float vel[2];
    
};

class CanvasCell
{
public:
    typedef Topologies::Cube<2>::Topology Topology;
    static const int TILE_WIDTH = 4;
    static const int TILE_HEIGHT = 4;
    
    static inline unsigned nanoSteps()
    {
        return 1;
    }

    CanvasCell(
        Coord<2> _pos = Coord<2>(), 
        bool _forceSet = false,
        FloatCoord<2> _forceFixed = FloatCoord<2>()) :
        cameraLevel(0)
    {
        pos[0] = _pos.x();
        pos[1] = _pos.y();
        forceFixed[0] = _forceFixed.c[0];
        forceFixed[1] = _forceFixed.c[1];
        forceSet = _forceSet;
    }

    template<typename COORD_MAP>
    void update(const COORD_MAP& hood, const unsigned& nanoStep)
    {
        const CanvasCell& oldSelf = hood[Coord<2>()];

        pos[0] = oldSelf.pos[0];
        pos[1] = oldSelf.pos[1];
        cameraPixel = oldSelf.cameraPixel;

        forceSet = oldSelf.forceSet;
        if (forceSet) {
            forceFixed[0] = oldSelf.forceFixed[0];
            forceFixed[1] = oldSelf.forceFixed[1];
        } else {
            forceFixed[0] = (hood[Coord<2>(0, -1)].forceFixed[0] +
                             hood[Coord<2>(-1, 0)].forceFixed[0] +
                             hood[Coord<2>(1,  0)].forceFixed[0] +
                             hood[Coord<2>(0,  1)].forceFixed[0]) * 0.25;
            forceFixed[1] = (hood[Coord<2>(0, -1)].forceFixed[1] +
                             hood[Coord<2>(-1, 0)].forceFixed[1] +
                             hood[Coord<2>(1,  0)].forceFixed[1] +
                             hood[Coord<2>(0,  1)].forceFixed[1]) * 0.25;
        }

        cameraLevel = (hood[Coord<2>(0, -1)].cameraLevel +
                       hood[Coord<2>(-1, 0)].cameraLevel +
                       hood[Coord<2>(1,  0)].cameraLevel +
                       hood[Coord<2>(0,  1)].cameraLevel) * 0.25;
        
        float gradientX = hood[Coord<2>(1, 0)].cameraLevel - hood[Coord<2>(-1, 0)].cameraLevel;
        float gradientY = hood[Coord<2>(0, 1)].cameraLevel - hood[Coord<2>(0, -1)].cameraLevel;
        forceVario[0] = 0;
        if ((gradientX > 0.011) || (gradientX < -0.011)) {
            forceVario[0] = 0.01 / gradientX;
        } else {
            forceVario[0] = 0;
        }

        if ((gradientY > 0.011) || (gradientY < -0.011)) {
            forceVario[1] = 0.01 / gradientY;
        } else {
            forceVario[1] = 0;
        }

//         float gradient[2];
//         gradient[0] = hood[Coord<2>(1, 0)].smoothCam - hood[Coord<2>(-1, 0)].smoothCam;
//         gradient[1] = hood[Coord<2>(0, 1)].smoothCam - hood[Coord<2>(0, -1)].smoothCam;

//         forceVario[0] = 0;
//         forceVario[1] = 0;
        
//         if ((gradient[0] * gradient[0]) > gradientCutoff) {
//             forceVario[0] = 1.0 / gradient[0];
//         }

//         if ((gradient[1] * gradient[1]) > gradientCutoff) {
//             forceVario[1] = 1.0 / gradient[1];
//         }

//         updateParticles(oldSelf.particle, 
//                         forceVario[0] + forceFixed[0],
//                         forceVario[1] + forceFixed[1]);
//         // moveParticles();
    }

    void readCam(unsigned char *frame, const float& factorX, const float& factorY, const int& width, const int& height)
    {
        int posX = pos[0] * factorX;
        int posY = pos[1] * factorY;
        
        int index = posY * width * 3 + posX * 3;
        cameraPixel = (0xff << 24) + 
            (frame[index + 0] << 16) +
            (frame[index + 1] <<  8) +
            (frame[index + 2] <<  0);

        int val = (int)frame[index + 0] + (int)frame[index + 1] + (int)frame[index + 2];
        if (val < 500) {
            cameraLevel = 1.0;
        } else {
            cameraLevel = std::max(0.0, cameraLevel - 0.05);
        }
    }

    // fixme:
// private:
    unsigned color[TILE_HEIGHT][TILE_WIDTH];
    unsigned cameraPixel;
    float pos[2];
    bool forceSet;

    float cameraLevel;
    float forceVario[2];
    float forceFixed[2];
    Particle particles;  

    void updateParticles()
    {}

    void moveParticles()
    {}
};

class CanvasInitializer : public SimpleInitializer<CanvasCell>
{
public:
    CanvasInitializer() :
        // SimpleInitializer<CanvasCell>(Coord<2>(240, 135), 100)
        SimpleInitializer<CanvasCell>(Coord<2>(320, 180), 100)
        // SimpleInitializer<CanvasCell>(Coord<2>(640, 360), 100)
    {}

    virtual void grid(GridBase<CanvasCell, 2> *ret)
    {
        CoordBox<2> box = ret->boundingBox();
        CoordBoxSequence<2> s = box.sequence();
        while (s.hasNext()) {
            Coord<2> c = s.next();
            bool setForce = false;
            FloatCoord<2> force;

            if ((c.x() == 100) && (c.y() >= 100) && (c.y() <= 200)) {
                setForce = true;
                force.c[1] = -1;
            }
            if ((c.x() == 200) && (c.y() >= 100) && (c.y() <= 200)) {
                setForce = true;
                force.c[1] = 1;
            }
            if ((c.y() == 100) && (c.x() >= 100) && (c.x() <= 200)) {
                setForce = true;
                force.c[0] = 1;
            }
            if ((c.y() == 200) && (c.x() >= 100) && (c.x() <= 200)) {
                setForce = true;
                force.c[0] = -1;
            }

            ret->at(c) = CanvasCell(c, setForce, force);
        }
    }
};

int runGUI(int argc, char **argv)
{
    QApplication app(argc, argv);
    FlowWidget flow;
    flow.resize(1200, 900);

    InteractiveSimulator *sim = new InteractiveSimulatorCPU<CanvasCell>(
        &flow,
        new CanvasInitializer());
    FrameGrabber *grabber = new FrameGrabber(true, &flow);

    QTimer *timerFlow = new QTimer(&flow);
    QTimer *timerGrab = new QTimer(&flow);
    QTimer *timerInfo = new QTimer(&flow);

    QObject::connect(timerInfo, SIGNAL(timeout()),           &flow,   SLOT(info()));
    QObject::connect(timerInfo, SIGNAL(timeout()),           grabber, SLOT(info()));
    QObject::connect(timerInfo, SIGNAL(timeout()),           sim,     SLOT(info()));
    QObject::connect(timerFlow, SIGNAL(timeout()),           &flow,   SLOT(ping()));
    QObject::connect(timerGrab, SIGNAL(timeout()),           grabber, SLOT(grab()));

    QObject::connect(grabber,   SIGNAL(newFrame(char*, unsigned, unsigned)), 
                     sim,       SLOT(updateCam( char*, unsigned, unsigned)));
    QObject::connect(&flow,     SIGNAL(updateImage(QImage*)),
                     sim,       SLOT(renderImage(QImage*)));
    QObject::connect(&app,      SIGNAL(lastWindowClosed()),  sim,       SLOT(quit()));
    QObject::connect(&app,      SIGNAL(lastWindowClosed()),  timerFlow, SLOT(stop()));
    QObject::connect(&app,      SIGNAL(lastWindowClosed()),  timerGrab, SLOT(stop()));
    QObject::connect(&app,      SIGNAL(lastWindowClosed()),  timerInfo, SLOT(stop()));

    QThreadPool *threadPool = QThreadPool::globalInstance();
    threadPool->start(sim);


    grabber->grab();
    timerFlow->start(50);
    timerGrab->start(500);
    timerInfo->start(5000);
    flow.show();
    int ret = app.exec();
    threadPool->waitForDone();
    return ret;
}

int main(int argc, char *argv[])
{
    runGUI(argc, argv);
}
