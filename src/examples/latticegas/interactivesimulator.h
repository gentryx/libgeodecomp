#ifndef _libgeodecomp_examples_latticegas_interactivesimulator_h_
#define _libgeodecomp_examples_latticegas_interactivesimulator_h_

#include <iostream>
#include <QObject>
#include <QRunnable>
#include <QSemaphore>
#include <libgeodecomp/examples/latticegas/bigcell.h>
#include <libgeodecomp/examples/latticegas/fpscounter.h>
#include <libgeodecomp/examples/latticegas/simparams.h>

class InteractiveSimulator : public QObject, public QRunnable, FPSCounter
{
    Q_OBJECT
    
public:
    InteractiveSimulator(QObject *parent);
    ~InteractiveSimulator();

    static char pixelToState(char r, char g, char b)
    {
        float sum = 
            r * SimParams::weightR +
            g * SimParams::weightG +
            b * SimParams::weightB;
        return sum <= 1 ? Cell::liquid : Cell::solid;
    }

public slots:
    void updateCam(char *rawFrame, unsigned width, unsigned height)
    {
        // std::cout << "  cam -> states\n";
        for (int y = 0; y < SimParams::modelHeight; ++y) {
            for (int x = 0; x < SimParams::modelWidth; ++x) {
                int sourceX = x * (width  - 1) / SimParams::modelWidth;
                int sourceY = y * (height - 1) / SimParams::modelHeight;
                int sourceOffset = sourceY * width + sourceX;
                char r = rawFrame[sourceOffset * 3 + 0];
                char g = rawFrame[sourceOffset * 3 + 1];
                char b = rawFrame[sourceOffset * 3 + 2];
                char state = pixelToState(r, g, b);

                states[y * SimParams::modelWidth + x] = state;
            }
        }
        
        newCameraFrame.release();
    }

    void renderImage(unsigned *image, unsigned width, unsigned height);

    void step();

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

private:
    int t;
    volatile bool running;
    QSemaphore newOutputFrameRequested;
    QSemaphore newOutputFrameAvailable;
    QSemaphore newCameraFrame;

    unsigned *outputFrame;
    volatile unsigned outputFrameWidth;
    volatile unsigned outputFrameHeight;

    std::vector<char> states;
    // fixme: get rid of these
    std::vector<BigCell> gridOld;
    std::vector<BigCell> gridNew;
    std::vector<unsigned> frame;
};

#endif
