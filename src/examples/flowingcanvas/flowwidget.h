#ifndef _libgeodecomp_examples_flowingcanvas_flowwidget_h_
#define _libgeodecomp_examples_flowingcanvas_flowwidget_h_

#include <iostream>
#include <QtGui/QPainter>
#include <QtGui/QWidget>
#include <libgeodecomp/examples/latticegas/fpscounter.h>

class FlowWidget : public QWidget, FPSCounter
{
    Q_OBJECT

public:
    FlowWidget() :
        frameCounter(0),
        image(1024, 768, QImage::Format_ARGB32)
    {}

    // fixme: do we need to always draw the whole image?
    void paintEvent(QPaintEvent * /* event */)
    {
        QPainter painter(this);

        painter.drawImage(0, 0, image);

        painter.setPen(Qt::green);
        painter.drawText(32, 32, "Frame " + QString::number(frameCounter));
        ++frameCounter;
    }

public slots:
    void ping()
    {
        emit updateImage((unsigned*)image.scanLine(0), image.width(), image.height());        
        // if (simParamsHost.dumpFrames) 
        //     dumpFrame();
        update();
        incFrames();
    }

    void info()
    {
        std::cout << "FlowWidget @ " << fps() << " FPS\n";
    }

signals:
    void updateImage(unsigned *image, unsigned width, unsigned height);

private:
    int frameCounter;
    QImage image;
};

#endif
