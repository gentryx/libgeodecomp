#ifndef _libgeodecomp_examples_latticegas_interactivesimulator_h_
#define _libgeodecomp_examples_latticegas_interactivesimulator_h_

#include <iostream>
#include <QMutex>
#include <QObject>
#include <QRunnable>
#include <libgeodecomp/examples/latticegas/fpscounter.h>

class InteractiveSimulator : public QObject, public QRunnable, FPSCounter
{
    Q_OBJECT

public:
    InteractiveSimulator(QObject *parent) :
        QObject(parent),
        t(0),
        mutex(QMutex::Recursive),
        frame(1),
        frameWidth(1),
        frameHeight(1)
    {}

public slots:
    void updateCam(unsigned *camFrame, unsigned width, unsigned height)
    {
        mutex.lock();
        std::cout << "updateCam()\n";
        unsigned size = width * height;
        frame.resize(size);
        std::copy(camFrame, camFrame + size, &frame[0]);
        frameWidth = width;
        frameHeight = height;
        mutex.unlock();
    }

    void renderImage(unsigned *image, unsigned width, unsigned height)
    {
        mutex.lock();

        if (frameWidth < 1 || frameHeight < 1) 
            throw std::logic_error("captured frame too small");

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int sourceX = x * (frameWidth - 1)  / width;
                int sourceY = y * (frameHeight - 1) / height;
                unsigned val = frame[sourceY * frameWidth + sourceX];
                image[y * width + x] = val;
            }
        }

        mutex.unlock();
    }

    void step()
    {
        mutex.lock();
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
    std::vector<unsigned> frame;
    unsigned frameWidth;
    unsigned frameHeight;
};

#endif
