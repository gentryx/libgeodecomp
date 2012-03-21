#include <QTimer>
#include <QApplication>
#include <QThreadPool>
#include <libgeodecomp/examples/flowingcanvas/canvascell.h>
#include <libgeodecomp/examples/flowingcanvas/canvasinitializer.h>
#include <libgeodecomp/examples/flowingcanvas/canvaswriter.h>
#include <libgeodecomp/examples/flowingcanvas/flowwidget.h>
#include <libgeodecomp/examples/flowingcanvas/framegrabber.h>
#include <libgeodecomp/examples/flowingcanvas/interactivesimulatorcpu.h>
#include <libgeodecomp/examples/flowingcanvas/interactivesimulatorgpu.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/misc/floatcoord.h>

using namespace LibGeoDecomp;

// class SimParams
// {
// public:
//     int maxCameraFrames;
//     float gradientCutoff;
// };

int runGUI(int argc, char **argv)
{
    QApplication app(argc, argv);
    FlowWidget flow;
    flow.resize(1200, 900);

    InteractiveSimulatorGPU<CanvasCell> *sim = new InteractiveSimulatorGPU<CanvasCell>(
    // InteractiveSimulatorCPU<CanvasCell> *sim = new InteractiveSimulatorCPU<CanvasCell>(
        &flow,
        new CanvasInitializer());
    CanvasWriter *writer = new CanvasWriter(sim->getOutputFrame(), sim);
    FrameGrabber *grabber = new FrameGrabber(false, &flow);

    QTimer *timerFlow = new QTimer(&flow);
    QTimer *timerGrab = new QTimer(&flow);
    QTimer *timerInfo = new QTimer(&flow);

    QObject::connect(timerInfo, SIGNAL(timeout()),           &flow,   SLOT(info()));
    QObject::connect(timerInfo, SIGNAL(timeout()),           grabber, SLOT(info()));
    QObject::connect(timerInfo, SIGNAL(timeout()),           sim,     SLOT(info()));
    QObject::connect(timerFlow, SIGNAL(timeout()),           &flow,   SLOT(ping()));
    QObject::connect(timerGrab, SIGNAL(timeout()),           grabber, SLOT(grab()));

    QObject::connect(&flow,     SIGNAL(cycleViewModeParticle()), writer,  SLOT(cycleViewModeParticle()));
    QObject::connect(&flow,     SIGNAL(cycleViewModeCamera()),   writer,  SLOT(cycleViewModeCamera()));

    QObject::connect(grabber,   SIGNAL(newFrame(char*, unsigned, unsigned)), 
                     sim,       SLOT(updateCam( char*, unsigned, unsigned)));
    QObject::connect(&flow,     SIGNAL(updateImage(QImage*)),
                     sim,       SLOT(renderImage(QImage*)));

    QObject::connect(&app,      SIGNAL(lastWindowClosed()),  sim,       SLOT(quit()));
    QObject::connect(&app,      SIGNAL(lastWindowClosed()),  timerFlow, SLOT(stop()));
    QObject::connect(&app,      SIGNAL(lastWindowClosed()),  timerGrab, SLOT(stop()));
    QObject::connect(&app,      SIGNAL(lastWindowClosed()),  timerInfo, SLOT(stop()));

    QThreadPool *threadPool = QThreadPool::globalInstance();
    // fixme: sync_update_patch
    // threadPool->start(sim);
    // fixme: sync_update_patch
    QObject::connect(timerFlow, SIGNAL(timeout()),           sim,       SLOT(step()));

    grabber->grab();
    timerFlow->start(5);
    timerGrab->start(100);
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
