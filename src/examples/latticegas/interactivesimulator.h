#ifndef _libgeodecomp_examples_latticegas_interactivesimulator_h_
#define _libgeodecomp_examples_latticegas_interactivesimulator_h_

#include <iostream>
#include <QMutex>
#include <QObject>
#include <QRunnable>
#include <libgeodecomp/examples/latticegas/bigcell.h>
#include <libgeodecomp/examples/latticegas/fpscounter.h>
#include <libgeodecomp/examples/latticegas/simparams.h>

class InteractiveSimulator : public QObject, public QRunnable, FPSCounter
{
    Q_OBJECT
    
public:
    InteractiveSimulator(QObject *parent) :
        QObject(parent),
        t(0),
        mutex(QMutex::Recursive),
        states(SimParams::modelSize, Cell::liquid),
        cellsOld(SimParams::modelSize),
        cellsNew(SimParams::modelSize),
        frame(SimParams::modelSize)
    {
        cellsOld[10 * SimParams::modelWidth + 10][0] = Cell(Cell::liquid, Cell::R, 1);
    }

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

    static unsigned stateToPixel(char state)
    {
        switch (state) {
        case Cell::liquid:
            return 0xff000000;
        case Cell::solid:
            return 0xffffffff;
        default:
            return 0xffff0000;
        }
    }

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
        mutex.lock();

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

        // std::cout << "  states -> cells\n";
        for (int y = 0; y < SimParams::modelHeight; ++y) {
            for (int x = 0; x < SimParams::modelWidth; ++x) {
                unsigned pos = y * SimParams::modelWidth + x;
                cellsOld[pos][0].getState() = states[pos];
                cellsOld[pos][1].getState() = states[pos];
            }
        }

        mutex.unlock();
    }

    void renderImage(unsigned *image, unsigned width, unsigned height)
    {
        mutex.lock();

        // std::cout << "  cells -> frame\n";
        for (int y = 0; y < SimParams::modelHeight; ++y) {
            for (int x = 0; x < SimParams::modelWidth; ++x) {
                unsigned pos = y * SimParams::modelWidth + x;
                frame[pos] = bigCellToColor(cellsOld[pos]);
            }
        }

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int sourceX = x * (SimParams::modelWidth  - 1) / width;
                int sourceY = y * (SimParams::modelHeight - 1) / height;
                unsigned val = frame[sourceY * SimParams::modelWidth + sourceX];
                image[y * width + x] = val;
            }
        }

        mutex.unlock();
    }

    void step()
    {
        mutex.lock();

        for (int y = 1; y < SimParams::modelHeight - 1; ++y) {
            for (int x = 1; x < SimParams::modelWidth - 1; ++x) {
                unsigned pos = y * SimParams::modelWidth + x;
                cellsNew[pos].update(&cellsOld[pos - SimParams::modelWidth],
                                     &cellsOld[pos],
                                     &cellsOld[pos + SimParams::modelWidth]);
            }
        }
        std::swap(cellsNew, cellsOld);

        incFrames();
        ++t;
        mutex.unlock();
    }
     
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
    QMutex mutex;
    std::vector<char> states;
    std::vector<BigCell> cellsOld;
    std::vector<BigCell> cellsNew;
    std::vector<unsigned> frame;
};

#endif
