#ifndef _libgeodecomp_examples_flowingcanvas_flowwidget_h_
#define _libgeodecomp_examples_flowingcanvas_flowwidget_h_

#include <iostream>
#include <QKeyEvent>
#include <QPainter>
#include <QWidget>
#include <libgeodecomp/examples/latticegas/fpscounter.h>

class FlowWidget : public QWidget, FPSCounter
{
    Q_OBJECT

public:
    FlowWidget() :
        frameCounter(0),
        image(1024, 768, QImage::Format_ARGB32)
    {
        setFocusPolicy(Qt::StrongFocus);
    }

    // fixme: do we need to always draw the whole image?
    virtual void paintEvent(QPaintEvent * /* event */)
    {
        QPainter painter(this);

        painter.drawImage(0, 0, image);

        painter.setPen(Qt::green);
        painter.drawText(32, 32, "Frame " + QString::number(frameCounter));
        ++frameCounter;
    }

    virtual void keyPressEvent(QKeyEvent *event)
    {
        std::cout << "got key " << event->key() << "\n";
        if (event->key() == Qt::Key_Space) {
            emit cycleViewModeParticle();
        }

        if (event->key() == Qt::Key_Enter) {
            emit cycleViewModeCamera();
        }
    }

public slots:
    void ping()
    {
        emit updateImage(&image);
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
    void updateImage(QImage*);
    void cycleViewModeParticle();
    void cycleViewModeCamera();

private:
    int frameCounter;
    QImage image;
};

#endif
