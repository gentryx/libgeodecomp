#ifndef _libgeodecomp_examples_flowingcanvas_interactivesimulator_h_
#define _libgeodecomp_examples_flowingcanvas_interactivesimulator_h_

#include <iostream>
#include <QImage>
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

    QImage **getOutputFrame()
    {
        return &outputFrame;
    }

public slots:
    void updateCam(char *rawFrame, unsigned width, unsigned height)
    {
        // fixme: we could get raceconditions here as the simulator
        // may not have consumed the last camera frame
        int frameSize = 3 * width * height;
        cameraFrameWidth  = width;
        cameraFrameHeight = height;
        cameraFrame.resize(frameSize);
        std::copy(rawFrame, rawFrame + frameSize, &cameraFrame[0]);

        newCameraFrame.release();
    }

    virtual void step() 
    {
        if (newCameraFrame.tryAcquire()) {
            readCam();
        }
        if (newOutputFrameRequested.tryAcquire()) {
            renderOutput();
            // fixme: sync_update_patch
            // newOutputFrameAvailable.release();
        }

        update();
        incFrames();
        // std::cout << "\r" << getFrames() << " " << fps() << " FPS";
    }

    void renderImage(QImage *image) 
    {
        // fixme: sync_update_patch
        // fixme
        // if (!running) {
        //     return;
        // }
        
        outputFrame = image;
        newOutputFrameRequested.release();
        // fixme: sync_update_patch
        // newOutputFrameAvailable.acquire();
    }

    virtual void readCam() = 0;
    virtual void renderOutput() = 0;
    virtual void update() = 0;

    void info()
    {
        std::cout << "InteractiveSimulator @ " << fps() << " FPS\n\n";
    }

    virtual void run()
    {
        running = true;
        while (running) {
            step();
        }
    }

    void quit()
    {
        running = false;
    }

protected:
    volatile bool running;
    QSemaphore newOutputFrameRequested;
    QSemaphore newOutputFrameAvailable;
    QSemaphore newCameraFrame;

    QImage *outputFrame;
    SuperVector<unsigned char> cameraFrame;
    int cameraFrameWidth;
    int cameraFrameHeight;
};

}

#endif
