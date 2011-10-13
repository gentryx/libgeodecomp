#ifndef _libgeodecomp_examples_latticegas_framegrabber_h_
#define _libgeodecomp_examples_latticegas_framegrabber_h_

#include <iostream>
#include <QObject>
#include <libgeodecomp/examples/latticegas/fpscounter.h>

class FrameGrabber : public QObject, FPSCounter
{
    Q_OBJECT

public:
    FrameGrabber(QObject *parent) :
        QObject(parent)
    {}

public slots:
    void grab()
    {
        incFrames();
    }

    void info()
    {
        std::cout << "FrameGrabber @ " << fps() << " FPS\n";
    }
};

#endif
