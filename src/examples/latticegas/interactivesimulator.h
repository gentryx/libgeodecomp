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

    static char pixelToState(unsigned val)
    {
        float r = (val >> 16) & 0xff;
        float g = (val >>  8) & 0xff;
        float b = (val >>  0) & 0xff;
        float sum = 
            r * SimParams::weightR +
            g * SimParams::weightG +
            b * SimParams::weightB;
        return sum <= 1 ? Cell::liquid : Cell::solid;
    }

    // fixme: kill this
    static unsigned bigCellToColor(BigCell c)
    {
        unsigned r = 0;
        unsigned g = 0;
        unsigned b = 0;
   
        for (int y = 0; y < 2; ++y) {
            if (c[y].getState() != Cell::liquid) {
                r += 255;
                g += 255;
                b += 255;
            } else {
                for (int i = 0; i < 7; ++i) {
                    r += Cell::palette[c[y][i]][0];
                    g += Cell::palette[c[y][i]][1];
                    b += Cell::palette[c[y][i]][2];
                }
            }
        }

        if (r > 255)
            r = 255;
        if (g > 255)
            g = 255;
        if (b > 255)
            b = 255;

        return (0xff << 24) +
            (r << 16) +
            (g << 8) +
            (b << 0);
    }
    

public slots:
    void updateCam(unsigned *rawFrame, unsigned width, unsigned height)
    {
        // std::cout << "  cam -> states\n";
        for (int y = 0; y < SimParams::modelHeight; ++y) {
            for (int x = 0; x < SimParams::modelWidth; ++x) {
                int sourceX = x * (width  - 1) / SimParams::modelWidth;
                int sourceY = y * (height - 1) / SimParams::modelHeight;
                unsigned val = rawFrame[sourceY * width + sourceX];
                char state = pixelToState(val);

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
    std::vector<BigCell> gridOld;
    std::vector<BigCell> gridNew;
    std::vector<unsigned> frame;
};

#endif
