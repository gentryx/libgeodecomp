#ifndef _libgeodecomp_examples_latticegas_flowwidget_h_
#define _libgeodecomp_examples_latticegas_flowwidget_h_

#include <iostream>
#include <QtGui/QColor>
#include <QtGui/QPainter>
#include <QtGui/QWidget>
#include <libgeodecomp/examples/latticegas/fpscounter.h>

class FlowWidget : public QWidget, FPSCounter
{
    Q_OBJECT

public:
    FlowWidget() :
        counter(0),
        image(1024, 768, QImage::Format_ARGB32)
    {}

    void paintEvent(QPaintEvent * /* event */)
    {
        QPainter painter(this);

        painter.drawImage(0, 0, image);

        painter.setPen(Qt::green);
        painter.drawText(32, 32, "Frame " + QString::number(counter));
        ++counter;
    }

public slots:
    void ping()
    {
        emit updateImage((unsigned*)image.scanLine(0), image.width(), image.height());
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
    int counter;
    QImage image;
};

#endif
