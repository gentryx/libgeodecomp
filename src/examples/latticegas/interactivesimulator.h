#ifndef _libgeodecomp_examples_latticegas_interactivesimulator_h_
#define _libgeodecomp_examples_latticegas_interactivesimulator_h_

#include <iostream>
#include <QObject>
#include <QRunnable>
#include <QSemaphore>
#include <libgeodecomp/examples/latticegas/bigcell.h>
#include <libgeodecomp/examples/latticegas/fpscounter.h>
#include <libgeodecomp/examples/latticegas/simparams.h>

class InteractiveSimulator : public QObject, public QRunnable, protected FPSCounter
{
    Q_OBJECT
    
public:
    InteractiveSimulator(QObject *parent) :
        QObject(parent),
        t(0),
        states(simParamsHost.modelSize, Cell::liquid)
    {}

    virtual ~InteractiveSimulator()
    {}

    static char pixelToState(unsigned char r, unsigned char g, unsigned char b)
    {
        float sum = 
            r * simParamsHost.weightR +
            g * simParamsHost.weightG +
            b * simParamsHost.weightB;
        return sum >= 1 ? Cell::liquid : Cell::solid1;
    }

public slots:
    void updateCam(char *rawFrame, unsigned width, unsigned height)
    {
        // std::cout << "  cam -> states\n";
        // fixme: move this to GPU?
        for (int y = 0; y < simParamsHost.modelHeight; ++y) {
            for (int x = 0; x < simParamsHost.modelWidth; ++x) {
                int sourceX = x * (width  - 1) / simParamsHost.modelWidth;
                int sourceY = y * (height - 1) / simParamsHost.modelHeight;
                int sourceOffset = sourceY * width + sourceX;
                char r = rawFrame[sourceOffset * 3 + 0];
                char g = rawFrame[sourceOffset * 3 + 1];
                char b = rawFrame[sourceOffset * 3 + 2];
                char state = pixelToState(r, g, b);

                // influx
                if (x <= 1)
                    state = Cell::source;

                // add right walls
                if (x >= simParamsHost.modelWidth - 2)
                    if ((y & 63) > simParamsHost.effluxSize)
                        state = Cell::solid1;

                // add upper and lower walls
                if (y <= 1 || y >= simParamsHost.modelHeight - 2)
                    state = Cell::slip;

                states[y * simParamsHost.modelWidth + x] = state;
            }
        }
        
        newCameraFrame.release();
    }

    void renderImage(unsigned *image, unsigned width, unsigned height) 
    {
        outputFrame = image;
        outputFrameWidth = width;
        outputFrameHeight = height;
        newOutputFrameRequested.release(1);
        newOutputFrameAvailable.acquire(1);
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
            // std::cout << "\r" << t << " " << fps() << " FPS";
        }
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

    std::vector<char> states;
};

#endif
