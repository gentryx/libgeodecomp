#ifndef _libgeodecomp_examples_flowingcanvas_interactivesimulator_h_
#define _libgeodecomp_examples_flowingcanvas_interactivesimulator_h_

#include <iostream>
#include <QObject>
#include <QRunnable>
#include <QSemaphore>
#include <libgeodecomp/examples/latticegas/fpscounter.h>

namespace LibGeoDecomp {

class InteractiveSimulator : public QObject, public QRunnable, protected FPSCounter
{
    Q_OBJECT
    
public:
    InteractiveSimulator(QObject *parent) :
        QObject(parent)
    {}

public slots:
    void updateCam(char *rawFrame, unsigned width, unsigned height)
    {
    }

    void step() 
    {
        if (newCameraFrame.tryAcquire()) 
            loadStates();
        if (newOutputFrameRequested.tryAcquire()) {
            renderOutput();
            newOutputFrameAvailable.release();
        }

        update();
        incFrames();
        ++t;
    }

    void renderImage(unsigned *image, unsigned width, unsigned height) 
    {
        if (!running) {
            return;
        }

        outputFrame = image;
        outputFrameWidth = width;
        outputFrameHeight = height;
        newOutputFrameRequested.release();
        newOutputFrameAvailable.acquire();
    }

    virtual void loadStates() = 0;
    virtual void renderOutput() = 0;
    virtual void update() = 0;
    
    void info()
    {
        std::cout << "InteractiveSimulator @ " << fps() << " FPS\n\n";
    }

    void run()
    {
        running = true;
        while (running) {
            step();
            std::cout << t << " " << fps() << " FPS\r";
        }
        std::cout << "run done\n";
        std::cout << "newOutputFrameRequested: " << newOutputFrameRequested.available() << "\n"
                  << "newOutputFrameAvailable: " << newOutputFrameAvailable.available() << "\n"
                  << "newCameraFrame:          " << newCameraFrame.available() << "\n";
    }

    void quit()
    {
        std::cout << "i've been told to quit\n";
        running = false;
    }

protected:
    int t;
    volatile bool running;
    QSemaphore newOutputFrameRequested;
    QSemaphore newOutputFrameAvailable;
    QSemaphore newCameraFrame;

    unsigned *outputFrame;
    volatile unsigned outputFrameWidth;
    volatile unsigned outputFrameHeight;
};

}

#endif
