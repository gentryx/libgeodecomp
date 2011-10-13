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

        // for (int y = 0; y < 768; ++y) {
        //     QRgb *line = (QRgb*)image.scanLine(y);

        //     for (int x = 0; x < 1024; ++x) {
        //         line[x] = 
        //             (0xff << 24) + 
        //             ((counter & 0xff) << 16) +
        //             ((x & 0xff) << 8) +
        //             ((y & 0xff) << 0);
        //     }
        // }

        // slow!
        // painter.drawImage(rect(), image);
        painter.drawImage(0, 0, image);

        painter.setPen(Qt::red);
        painter.drawLine(10 + counter, 10, 500 + counter, 500);
        ++counter;
    }

    char *getImage()
    {
        return (char*)image.scanLine(0);
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
