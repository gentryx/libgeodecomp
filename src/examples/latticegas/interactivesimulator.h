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
        mutex(QMutex::Recursive)
    {}

public slots:
    void updateCam(unsigned *frame)
    {
        mutex.lock();
        std::cout << "\nupdateCam()\n";
        mutex.unlock();
    }

    void renderImage(unsigned *image, unsigned& width, unsigned& height)
    {
        std::cout << "\nrenderImage(" << width << ", " << height << ")\n";
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
};

#endif
