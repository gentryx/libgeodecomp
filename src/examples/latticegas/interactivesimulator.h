#ifndef _libgeodecomp_examples_latticegas_interactivesimulator_h_
#define _libgeodecomp_examples_latticegas_interactivesimulator_h_

#include <iostream>
#include <QObject>
#include <QRunnable>
#include <libgeodecomp/examples/latticegas/fpscounter.h>

class InteractiveSimulator : public QObject, public QRunnable, FPSCounter
{
    Q_OBJECT

public:
    InteractiveSimulator(QObject *parent) :
        QObject(parent),
        t(0)
    {}

public slots:
    void updateCam()
    {
        std::cout << "\nupdateCam()\n";
    }

    void renderImage(unsigned *image, unsigned& width, unsigned& height)
    {
        std::cout << "\nrenderImage(" << width << ", " << height << ")\n";
    }

    void step()
    {
        incFrames();
        ++t;
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
};

#endif
